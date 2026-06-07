"""Fixed-size character chunking with overlap."""

import re
from pathlib import Path
from typing import Any


def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into deterministic fixed-size chunks."""
    validate_chunk_settings(chunk_size, chunk_overlap)

    if not text.strip():
        return []

    chunks: list[str] = []
    step = chunk_size - chunk_overlap

    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

        if start + chunk_size >= len(text):
            break

    return chunks


def validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    """Validate fixed-size chunking settings."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be zero or positive.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")


def create_chunk_records(
    document: dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
    chunking_strategy: str = "fixed_size",
) -> list[dict[str, Any]]:
    """Create chunk records for one document."""
    chunks = split_text_into_chunks(document["text"], chunk_size, chunk_overlap)
    source_file = document["source_file"]
    document_version = document["document_version"]
    source_slug = _make_source_slug(source_file)

    records: list[dict[str, Any]] = []
    for index, chunk_text in enumerate(chunks, start=1):
        records.append(
            {
                "chunk_id": f"{source_slug}_{document_version}_chunk_{index:03d}",
                "source_file": source_file,
                "source_path": document["source_path"],
                "document_version": document_version,
                "chunk_index": index,
                "chunk_text": chunk_text,
                "chunking_strategy": chunking_strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
        )

    return records


def _make_source_slug(source_file: str) -> str:
    """Create a readable stable slug from a filename."""
    stem = Path(source_file).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return slug or "document"

