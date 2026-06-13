import pytest

from src.search.indexing_pipeline import IndexingPipelineError, build_index_record


def _chunk():
    return {
        "chunk_id": "leave_policy_v1_chunk_001",
        "source_file": "leave_policy.txt",
        "source_path": "data/raw/policies_v1/leave_policy.txt",
        "document_version": "v1",
        "chunk_index": 1,
        "chunk_text": "Employees receive 12 casual leaves per calendar year.",
        "chunking_strategy": "fixed_size",
        "chunk_size": 500,
        "chunk_overlap": 50,
    }


def test_required_metadata_is_preserved():
    record = build_index_record(_chunk(), [0.1, 0.2], "text-embedding-3-small", "v1", "now")

    assert record["source_file"] == "leave_policy.txt"
    assert record["source_path"] == "data/raw/policies_v1/leave_policy.txt"
    assert record["document_version"] == "v1"
    assert record["chunk_index"] == 1


def test_ids_are_stable():
    record = build_index_record(_chunk(), [0.1, 0.2], "text-embedding-3-small", "v1", "now")

    assert record["id"] == "leave_policy_v1_chunk_001"


def test_empty_chunk_text_fails():
    chunk = _chunk()
    chunk["chunk_text"] = " "

    with pytest.raises(IndexingPipelineError, match="empty"):
        build_index_record(chunk, [0.1, 0.2], "text-embedding-3-small", "v1", "now")


def test_embedding_model_and_index_version_are_attached():
    record = build_index_record(_chunk(), [0.1, 0.2], "text-embedding-3-small", "v1", "now")

    assert record["embedding_model"] == "text-embedding-3-small"
    assert record["index_version"] == "v1"


def test_vector_length_is_validated():
    with pytest.raises(IndexingPipelineError, match="Embedding vector length mismatch"):
        build_index_record(
            _chunk(),
            [0.1, 0.2],
            "text-embedding-3-small",
            "v1",
            "now",
            expected_dimensions=3,
        )
