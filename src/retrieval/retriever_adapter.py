"""Small adapter around the Lecture 3 retriever."""

import os
from typing import Any


class RetrieverAdapterError(RuntimeError):
    """Raised when the Lecture 3 retriever cannot be called."""


def retrieve_chunks(question: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Retrieve chunks through Lecture 3, or fake chunks only when explicitly enabled."""
    if os.getenv("USE_FAKE_RETRIEVER", "").lower() == "true":
        return _fake_retrieve_chunks(question, top_k)

    try:
        from src.search.retrieval_service import retrieve_chunks as lecture_3_retrieve_chunks
    except ModuleNotFoundError as error:
        raise RetrieverAdapterError(
            "Could not import the Lecture 3 retriever. Update "
            "src/retrieval/retriever_adapter.py with the correct import path."
        ) from error

    try:
        return lecture_3_retrieve_chunks(question, top_k=top_k)
    except TypeError as error:
        raise RetrieverAdapterError(
            "The Lecture 3 retriever signature was not compatible with "
            "retrieve_chunks(question, top_k=...). Update retriever_adapter.py."
        ) from error


def _fake_retrieve_chunks(question: str, top_k: int) -> list[dict[str, Any]]:
    """Return tiny sample chunks for offline wiring tests only."""
    sample_chunks = [
        {
            "rank": 1,
            "chunk_id": "leave_policy_v1_chunk_001",
            "content": "Employees receive 18 days of annual leave per calendar year.",
            "source_file": "leave_policy.txt",
            "document_version": "v1",
            "score": 0.99,
            "index_version": "fake-local",
        },
        {
            "rank": 2,
            "chunk_id": "travel_policy_v1_chunk_001",
            "content": "Travel reimbursement requires manager approval and receipts.",
            "source_file": "travel_policy.txt",
            "document_version": "v1",
            "score": 0.91,
            "index_version": "fake-local",
        },
        {
            "rank": 3,
            "chunk_id": "it_support_v1_chunk_001",
            "content": "For password resets, employees should contact the IT helpdesk.",
            "source_file": "it_support_policy.txt",
            "document_version": "v1",
            "score": 0.84,
            "index_version": "fake-local",
        },
    ]
    return sample_chunks[:top_k]
