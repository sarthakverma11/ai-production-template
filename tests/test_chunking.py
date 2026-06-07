import pytest

from src.processing.text_chunker import create_chunk_records, split_text_into_chunks


def _sample_document():
    return {
        "source_file": "leave_policy.txt",
        "source_path": "data/raw/policies_v1/leave_policy.txt",
        "document_version": "v1",
        "text": "abcdefghijklmnopqrstuvwxyz" * 20,
    }


def test_valid_text_creates_chunks():
    chunks = split_text_into_chunks("abcdef", chunk_size=3, chunk_overlap=1)

    assert chunks == ["abc", "cde", "ef"]


def test_empty_text_does_not_create_meaningless_chunks():
    assert split_text_into_chunks("   ", chunk_size=10, chunk_overlap=2) == []


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        split_text_into_chunks("abcdef", chunk_size=5, chunk_overlap=5)


def test_chunk_ids_are_unique():
    records = create_chunk_records(_sample_document(), chunk_size=100, chunk_overlap=20)
    chunk_ids = [record["chunk_id"] for record in records]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_metadata_is_preserved():
    records = create_chunk_records(_sample_document(), chunk_size=100, chunk_overlap=20)

    assert records[0]["source_file"] == "leave_policy.txt"
    assert records[0]["source_path"] == "data/raw/policies_v1/leave_policy.txt"
    assert records[0]["document_version"] == "v1"
    assert records[0]["chunking_strategy"] == "fixed_size"


def test_chunk_output_is_deterministic():
    first_run = create_chunk_records(_sample_document(), chunk_size=100, chunk_overlap=20)
    second_run = create_chunk_records(_sample_document(), chunk_size=100, chunk_overlap=20)

    assert first_run == second_run

