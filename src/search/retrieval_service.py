"""Semantic vector retrieval for Lecture 3."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config_loader import load_config
from src.embeddings.embedding_service import AzureOpenAIEmbeddingService
from src.utils import ensure_dir, load_project_env


SELECT_FIELDS = [
    "id",
    "content",
    "source_file",
    "source_path",
    "document_version",
    "chunk_index",
    "embedding_model",
    "index_version",
]


class RetrievalServiceError(RuntimeError):
    """Raised when semantic retrieval cannot complete."""


def retrieve_chunks(
    query: str,
    top_k: int | None = None,
    document_version: str | None = None,
    config_path: str = "configs/config.yaml",
    search_client: Any | None = None,
    embedding_service: AzureOpenAIEmbeddingService | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top-k evidence chunks using vector search."""
    load_project_env()
    if not isinstance(query, str) or not query.strip():
        raise RetrievalServiceError("Query must be a non-empty string.")

    config = load_config(config_path)
    search_config = config["search"]
    runtime_config = config["runtime"]
    resolved_top_k = top_k or int(search_config["top_k"])
    if resolved_top_k <= 0:
        raise RetrievalServiceError("top_k must be positive.")

    embedding_service = embedding_service or AzureOpenAIEmbeddingService()
    query_vector = embedding_service.embed_text(query)
    client = search_client or _create_search_client(
        index_name=os.getenv(search_config["index_name_env_var"], "")
    )
    filter_expression = _build_document_version_filter(document_version)

    results = _execute_vector_search(
        client=client,
        query_vector=query_vector,
        vector_field=search_config["vector_field"],
        top_k=resolved_top_k,
        filter_expression=filter_expression,
    )
    normalized_results = _normalize_results(results)

    ensure_dir(runtime_config["output_dir"])
    output_path = Path(runtime_config["output_dir"]) / "retrieval_results.json"
    output_payload = {
        "query": query,
        "top_k": resolved_top_k,
        "document_version": document_version,
        "retrieval_mode": search_config["mode"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": normalized_results,
    }
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    return normalized_results


def _execute_vector_search(
    client: Any,
    query_vector: list[float],
    vector_field: str,
    top_k: int,
    filter_expression: str | None,
) -> Any:
    try:
        from azure.search.documents.models import VectorizedQuery

        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields=vector_field,
        )
        return client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=SELECT_FIELDS,
            top=top_k,
            filter=filter_expression,
        )
    except ModuleNotFoundError:
        return client.search(
            search_text=None,
            vector_queries=[
                {
                    "vector": query_vector,
                    "k_nearest_neighbors": top_k,
                    "fields": vector_field,
                }
            ],
            select=SELECT_FIELDS,
            top=top_k,
            filter=filter_expression,
        )
    except Exception as error:
        raise RetrievalServiceError(f"Azure AI Search query failed: {type(error).__name__}") from error


def _normalize_results(results: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for rank, result in enumerate(results, start=1):
        score = result.get("@search.score") if isinstance(result, dict) else getattr(result, "@search.score", None)
        normalized.append(
            {
                "rank": rank,
                "chunk_id": _result_value(result, "id"),
                "content": _result_value(result, "content"),
                "source_file": _result_value(result, "source_file"),
                "source_path": _result_value(result, "source_path"),
                "document_version": _result_value(result, "document_version"),
                "chunk_index": _result_value(result, "chunk_index"),
                "embedding_model": _result_value(result, "embedding_model"),
                "index_version": _result_value(result, "index_version"),
                "score": score,
                "retrieval_mode": "vector",
            }
        )
    return normalized


def _result_value(result: Any, key: str) -> Any:
    if isinstance(result, dict):
        return result.get(key)
    return getattr(result, key, None)


def _build_document_version_filter(document_version: str | None) -> str | None:
    if not document_version:
        return None
    escaped_value = document_version.replace("'", "''")
    return f"document_version eq '{escaped_value}'"


def _create_search_client(index_name: str) -> Any:
    if not index_name:
        raise RetrievalServiceError("Missing AZURE_SEARCH_INDEX_NAME.")

    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("AZURE_SEARCH_API_KEY")
    if not endpoint:
        raise RetrievalServiceError("Missing AZURE_SEARCH_ENDPOINT.")
    if not api_key:
        raise RetrievalServiceError("Missing AZURE_SEARCH_API_KEY.")

    try:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
    except ModuleNotFoundError as error:
        raise RetrievalServiceError(
            "azure-search-documents is required. Install dependencies with pip install -r requirements.txt."
        ) from error

    return SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(api_key),
    )


def main() -> None:
    """CLI entry point for retrieval."""
    parser = argparse.ArgumentParser(description="Retrieve policy chunks from Azure AI Search.")
    parser.add_argument("query", help="Question or search query to embed and retrieve.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve.")
    parser.add_argument("--document-version", default=None, help="Optional document_version filter.")
    args = parser.parse_args()

    try:
        results = retrieve_chunks(
            query=args.query,
            top_k=args.top_k,
            document_version=args.document_version,
        )
    except Exception as error:
        raise SystemExit(f"Retrieval failed: {error}") from error

    for result in results:
        print(
            f"{result['rank']}. {result['chunk_id']} "
            f"score={result['score']} source={result['source_file']}"
        )


if __name__ == "__main__":
    main()
