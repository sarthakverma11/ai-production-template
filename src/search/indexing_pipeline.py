"""Offline indexing pipeline for Lecture 3 Azure vector search."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config_loader import load_config
from src.embeddings.embedding_service import AzureOpenAIEmbeddingService
from src.logger import get_logger
from src.search.index_manager import SearchIndexManager
from src.storage.blob_service import load_json_blob, validate_containers_exist
from src.utils import ensure_dir, load_project_env


REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "source_file",
    "source_path",
    "document_version",
    "chunk_index",
    "chunk_text",
    "chunking_strategy",
    "chunk_size",
    "chunk_overlap",
}


class IndexingPipelineError(RuntimeError):
    """Raised when the Lecture 3 indexing pipeline cannot complete."""


def build_index_record(
    chunk: dict[str, Any],
    embedding: list[float],
    embedding_model: str,
    index_version: str,
    generated_at: str,
    expected_dimensions: int | None = None,
) -> dict[str, Any]:
    """Map one Lecture 2 chunk into one Azure AI Search document."""
    validate_chunk(chunk)

    if expected_dimensions is not None and len(embedding) != expected_dimensions:
        raise IndexingPipelineError(
            f"Embedding vector length mismatch for {chunk['chunk_id']}: "
            f"expected {expected_dimensions}, got {len(embedding)}."
        )

    return {
        "id": chunk["chunk_id"],
        "content": chunk["chunk_text"],
        "content_vector": [float(value) for value in embedding],
        "source_file": chunk["source_file"],
        "source_path": chunk["source_path"],
        "document_version": chunk["document_version"],
        "chunk_index": int(chunk["chunk_index"]),
        "chunking_strategy": chunk["chunking_strategy"],
        "chunk_size": int(chunk["chunk_size"]),
        "chunk_overlap": int(chunk["chunk_overlap"]),
        "embedding_model": embedding_model,
        "index_version": index_version,
        "generated_at": generated_at,
    }


def build_index_records(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    embedding_model: str,
    index_version: str,
    generated_at: str,
    expected_dimensions: int | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic index records for all chunks."""
    if len(chunks) != len(embeddings):
        raise IndexingPipelineError(
            f"Chunk and embedding count mismatch: {len(chunks)} chunks, {len(embeddings)} embeddings."
        )

    return [
        build_index_record(
            chunk=chunk,
            embedding=embedding,
            embedding_model=embedding_model,
            index_version=index_version,
            generated_at=generated_at,
            expected_dimensions=expected_dimensions,
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]


def validate_chunk(chunk: dict[str, Any]) -> None:
    """Validate required Lecture 2 metadata before indexing."""
    missing_fields = REQUIRED_CHUNK_FIELDS - set(chunk)
    if missing_fields:
        raise IndexingPipelineError(
            "Chunk is missing required fields: " + ", ".join(sorted(missing_fields))
        )

    if not str(chunk["chunk_text"]).strip():
        raise IndexingPipelineError(f"Chunk text is empty for {chunk['chunk_id']}.")


def run_indexing_pipeline(
    config_path: str = "configs/config.yaml",
    recreate_index: bool = False,
    search_client: Any | None = None,
    blob_client: Any | None = None,
    embedding_service: AzureOpenAIEmbeddingService | None = None,
) -> dict[str, Any]:
    """Run the complete Blob-to-Search indexing workflow."""
    load_project_env()
    config = load_config(config_path)
    runtime_config = config["runtime"]
    storage_config = config["azure_storage"]
    embedding_config = config["embedding"]
    search_config = config["search"]

    ensure_dir(runtime_config["log_dir"])
    ensure_dir(runtime_config["output_dir"])
    logger = get_logger(
        name="lecture-3-vector-search",
        log_dir=runtime_config["log_dir"],
        log_file="lecture_3_vector_search.log",
    )

    _verify_required_env_vars(
        [
            "AZURE_STORAGE_CONNECTION_STRING",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
            "AZURE_SEARCH_ENDPOINT",
            "AZURE_SEARCH_API_KEY",
            "AZURE_SEARCH_INDEX_NAME",
        ]
    )

    validate_containers_exist(
        [
            storage_config["raw_container"],
            storage_config["processed_container"],
            storage_config["metadata_container"],
        ],
        client=blob_client,
    )

    chunks_artifact = load_json_blob(
        storage_config["processed_container"],
        storage_config["chunks_blob"],
        client=blob_client,
    )
    lineage_artifact = load_json_blob(
        storage_config["metadata_container"],
        storage_config["lineage_blob"],
        client=blob_client,
    )

    chunks = _extract_chunks(chunks_artifact)
    _validate_lineage(lineage_artifact)
    for chunk in chunks:
        validate_chunk(chunk)

    embedding_service = embedding_service or AzureOpenAIEmbeddingService()
    expected_dimensions = int(embedding_config["dimensions"])
    actual_dimensions = embedding_service.verify_embedding_dimensions(expected_dimensions)

    index_name = os.getenv(search_config["index_name_env_var"], "")
    index_manager = SearchIndexManager(
        index_name=index_name,
        logger=logger,
    )
    index_manager.create_index_if_absent(
        dimensions=actual_dimensions,
        vector_profile_name=search_config["vector_profile_name"],
        vector_algorithm_name=search_config["vector_algorithm_name"],
        recreate=recreate_index,
    )

    texts = [chunk["chunk_text"] for chunk in chunks]
    embeddings = _embed_in_batches(
        texts=texts,
        embedding_service=embedding_service,
        batch_size=int(embedding_config["batch_size"]),
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    records = build_index_records(
        chunks=chunks,
        embeddings=embeddings,
        embedding_model=embedding_service.deployment,
        index_version=search_config["index_version"],
        generated_at=generated_at,
        expected_dimensions=actual_dimensions,
    )

    upload_results = _upload_records(
        records=records,
        index_name=index_name,
        search_client=search_client,
    )
    failed_ids = [
        result["id"]
        for result in upload_results
        if not result["succeeded"]
    ]

    summary = {
        "index_name": index_name,
        "index_version": search_config["index_version"],
        "embedding_deployment": embedding_service.deployment,
        "embedding_dimensions": actual_dimensions,
        "source_container": storage_config["processed_container"],
        "source_blob": storage_config["chunks_blob"],
        "source_document_count": chunks_artifact.get("document_count"),
        "chunk_count": len(chunks),
        "uploaded_count": len(chunks) - len(failed_ids),
        "failed_count": len(failed_ids),
        "failed_ids": failed_ids,
        "generated_at": generated_at,
    }

    summary_path = Path(runtime_config["output_dir"]) / "indexing_summary_v1.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Indexing summary saved to %s", summary_path)

    if failed_ids:
        logger.error("Indexing failed for chunk IDs: %s", failed_ids)
        raise IndexingPipelineError("Indexing run completed with failed uploads.")

    logger.info("Indexed %s chunks into %s.", len(chunks), index_name)
    return summary


def _verify_required_env_vars(env_var_names: list[str]) -> None:
    missing = [name for name in env_var_names if not os.getenv(name)]
    if missing:
        raise IndexingPipelineError("Missing required environment variables: " + ", ".join(missing))


def _extract_chunks(chunks_artifact: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = chunks_artifact.get("chunks")
    if not isinstance(chunks, list):
        raise IndexingPipelineError("chunks_v1.json must contain a list field named 'chunks'.")
    if not chunks:
        raise IndexingPipelineError("chunks_v1.json contains no chunks.")
    return chunks


def _validate_lineage(lineage_artifact: dict[str, Any]) -> None:
    required_fields = {"pipeline_name", "input_path", "processed_chunk_count"}
    missing = required_fields - set(lineage_artifact)
    if missing:
        raise IndexingPipelineError(
            "lineage_v1.json is missing required fields: " + ", ".join(sorted(missing))
        )


def _embed_in_batches(
    texts: list[str],
    embedding_service: AzureOpenAIEmbeddingService,
    batch_size: int,
) -> list[list[float]]:
    if batch_size <= 0:
        raise IndexingPipelineError("Embedding batch size must be positive.")

    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embeddings.extend(embedding_service.embed_texts(batch))
    return embeddings


def _upload_records(
    records: list[dict[str, Any]],
    index_name: str,
    search_client: Any | None = None,
) -> list[dict[str, Any]]:
    client = search_client or _create_search_client(index_name)
    try:
        results = client.upload_documents(documents=records)
    except Exception as error:
        raise IndexingPipelineError(f"Azure AI Search upload failed: {type(error).__name__}") from error

    normalized_results: list[dict[str, Any]] = []
    for result in results:
        key = getattr(result, "key", None) or getattr(result, "id", None)
        succeeded = bool(getattr(result, "succeeded", False))
        normalized_results.append({"id": key, "succeeded": succeeded})

    return normalized_results


def _create_search_client(index_name: str) -> Any:
    try:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
    except ModuleNotFoundError as error:
        raise IndexingPipelineError(
            "azure-search-documents is required. Install dependencies with pip install -r requirements.txt."
        ) from error

    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("AZURE_SEARCH_API_KEY")
    if not endpoint:
        raise IndexingPipelineError("Missing AZURE_SEARCH_ENDPOINT.")
    if not api_key:
        raise IndexingPipelineError("Missing AZURE_SEARCH_API_KEY.")

    return SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(api_key),
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Index Lecture 2 chunks into Azure AI Search.")
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Explicitly delete and recreate the index before uploading documents.",
    )
    args = parser.parse_args()

    try:
        summary = run_indexing_pipeline(recreate_index=args.recreate_index)
    except Exception as error:
        raise SystemExit(f"Indexing pipeline failed: {error}") from error

    print(f"Indexed {summary['uploaded_count']} chunks into {summary['index_name']}.")


if __name__ == "__main__":
    main()
