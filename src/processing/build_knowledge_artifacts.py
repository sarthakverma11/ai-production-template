"""Build processed document chunks and lineage metadata."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config_loader import load_config
from src.ingestion.document_loader import load_documents
from src.ingestion.validate_documents import DocumentValidationError, validate_documents
from src.logger import get_logger
from src.processing.text_chunker import create_chunk_records, validate_chunk_settings
from src.processing.text_cleaner import clean_text
from src.utils import ensure_dir


def build_knowledge_artifacts(config_path: str = "configs/config.yaml") -> dict[str, Path]:
    """Run the Lecture 2 document-processing pipeline."""
    config = load_config(config_path)
    data_config = config["data"]
    processing_config = config["processing"]
    runtime_config = config["runtime"]

    raw_dir = Path(data_config["raw_dir"])
    processed_dir = Path(data_config["processed_dir"])
    metadata_dir = Path(data_config["metadata_dir"])
    log_dir = runtime_config["log_dir"]

    chunk_size = int(processing_config["chunk_size"])
    chunk_overlap = int(processing_config["chunk_overlap"])
    chunking_strategy = processing_config["strategy"]
    validate_chunk_settings(chunk_size, chunk_overlap)

    ensure_dir(str(processed_dir))
    ensure_dir(str(metadata_dir))
    ensure_dir(log_dir)

    logger = get_logger(
        name="lecture-2-document-pipeline",
        log_dir=log_dir,
        log_file="lecture_2_document_pipeline.log",
    )

    logger.info("Starting Lecture 2 document-processing pipeline.")
    logger.info("Loading documents from %s", raw_dir)
    documents = load_documents(
        folder_path=raw_dir,
        document_version=data_config["document_version"],
        supported_extensions=data_config["supported_extensions"],
    )

    logger.info("Validating %s loaded documents.", len(documents))
    try:
        validation_report = validate_documents(
            documents=documents,
            input_folder=raw_dir,
            supported_extensions=data_config["supported_extensions"],
            minimum_document_length=int(data_config["minimum_document_length"]),
        )
    except DocumentValidationError:
        logger.exception("Document validation failed. Processed artifacts were not created.")
        raise

    logger.info("Cleaning and chunking documents.")
    all_chunks: list[dict[str, Any]] = []
    processed_files: list[dict[str, Any]] = []

    for document in documents:
        cleaned_document = dict(document)
        cleaned_document["text"] = clean_text(document["text"])
        document_chunks = create_chunk_records(
            document=cleaned_document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy=chunking_strategy,
        )
        all_chunks.extend(document_chunks)
        processed_files.append(
            {
                "source_file": document["source_file"],
                "source_path": document["source_path"],
                "document_version": document["document_version"],
                "character_count": len(cleaned_document["text"]),
                "chunk_count": len(document_chunks),
            }
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    chunks_output_path = processed_dir / processing_config["chunks_output_file"]
    lineage_output_path = metadata_dir / processing_config["lineage_output_file"]

    chunks_artifact = {
        "artifact_type": "processed_document_chunks",
        "artifact_version": data_config["document_version"],
        "generated_at_utc": generated_at,
        "document_count": len(documents),
        "chunk_count": len(all_chunks),
        "processing_config": {
            "strategy": chunking_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        "chunks": all_chunks,
    }

    lineage_artifact = {
        "pipeline_name": "document_processing_pipeline",
        "pipeline_version": "1.0",
        "source_version": raw_dir.name,
        "input_path": raw_dir.as_posix(),
        "chunks_output_path": chunks_output_path.as_posix(),
        "lineage_output_path": lineage_output_path.as_posix(),
        "chunking_strategy": chunking_strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "generated_at_utc": generated_at,
        "validation_result": validation_report,
        "source_document_count": len(documents),
        "processed_chunk_count": len(all_chunks),
        "processed_files": processed_files,
        "processing_steps": [
            "load",
            "validate",
            "clean",
            "chunk",
            "save",
        ],
    }

    logger.info("Saving chunks artifact to %s", chunks_output_path)
    chunks_output_path.write_text(
        json.dumps(chunks_artifact, indent=2),
        encoding="utf-8",
    )

    logger.info("Saving lineage artifact to %s", lineage_output_path)
    lineage_output_path.write_text(
        json.dumps(lineage_artifact, indent=2),
        encoding="utf-8",
    )

    logger.info("Lecture 2 document-processing pipeline completed successfully.")
    return {
        "chunks_output_path": chunks_output_path,
        "lineage_output_path": lineage_output_path,
    }


def main() -> None:
    """CLI entry point for the document-processing pipeline."""
    output_paths = build_knowledge_artifacts()
    print(f"Chunks artifact created: {output_paths['chunks_output_path']}")
    print(f"Lineage artifact created: {output_paths['lineage_output_path']}")


if __name__ == "__main__":
    main()

