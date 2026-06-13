import pytest

from src.search import retrieval_service
from src.search.retrieval_service import RetrievalServiceError, retrieve_chunks


class FakeEmbeddingService:
    def __init__(self):
        self.calls = []

    def embed_text(self, text):
        self.calls.append(text)
        return [0.1, 0.2, 0.3]


class FakeSearchClient:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return [
            {
                "id": "leave_policy_v1_chunk_001",
                "content": "Leave policy content",
                "source_file": "leave_policy.txt",
                "source_path": "data/raw/policies_v1/leave_policy.txt",
                "document_version": "v1",
                "chunk_index": 1,
                "embedding_model": "text-embedding-3-small",
                "index_version": "v1",
                "@search.score": 0.89,
            }
        ]


def test_empty_query_fails():
    with pytest.raises(RetrievalServiceError, match="non-empty"):
        retrieve_chunks(" ")


def test_top_k_is_passed_correctly(tmp_path):
    config_path = _write_config(tmp_path)
    fake_client = FakeSearchClient()

    retrieve_chunks(
        "leave balance",
        top_k=5,
        config_path=str(config_path),
        search_client=fake_client,
        embedding_service=FakeEmbeddingService(),
    )

    assert fake_client.calls[0]["top"] == 5


def test_query_embedding_is_generated_once(tmp_path):
    config_path = _write_config(tmp_path)
    fake_embedding = FakeEmbeddingService()

    retrieve_chunks(
        "leave balance",
        config_path=str(config_path),
        search_client=FakeSearchClient(),
        embedding_service=fake_embedding,
    )

    assert fake_embedding.calls == ["leave balance"]


def test_selected_result_fields_are_returned(tmp_path):
    config_path = _write_config(tmp_path)

    results = retrieve_chunks(
        "leave balance",
        config_path=str(config_path),
        search_client=FakeSearchClient(),
        embedding_service=FakeEmbeddingService(),
    )

    assert results[0]["chunk_id"] == "leave_policy_v1_chunk_001"
    assert results[0]["source_file"] == "leave_policy.txt"
    assert results[0]["score"] == 0.89


def test_document_version_filter_is_constructed():
    assert retrieval_service._build_document_version_filter("v1") == "document_version eq 'v1'"


def test_result_order_and_rank_are_deterministic():
    results = retrieval_service._normalize_results(
        [
            {"id": "a", "content": "A", "@search.score": 0.9},
            {"id": "b", "content": "B", "@search.score": 0.8},
        ]
    )

    assert [result["rank"] for result in results] == [1, 2]
    assert [result["chunk_id"] for result in results] == ["a", "b"]


def _write_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
runtime:
  output_dir: {tmp_path.as_posix()}
search:
  top_k: 3
  mode: vector
  vector_field: content_vector
  index_name_env_var: AZURE_SEARCH_INDEX_NAME
""",
        encoding="utf-8",
    )
    return config_path
