from types import SimpleNamespace

import pytest

from src.embeddings.embedding_service import AzureOpenAIEmbeddingService, EmbeddingServiceError


class FakeEmbeddingsClient:
    def __init__(self, vectors=None, fail=False):
        self.vectors = vectors or [[1.0, 2.0], [3.0, 4.0]]
        self.fail = fail
        self.calls = []

    def create(self, model, input):
        self.calls.append((model, input))
        if self.fail:
            raise RuntimeError("boom")
        data = [
            SimpleNamespace(index=index, embedding=vector)
            for index, vector in enumerate(self.vectors[: len(input)])
        ]
        return SimpleNamespace(data=data)


def _service(fake_embeddings):
    fake_client = SimpleNamespace(embeddings=fake_embeddings)
    return AzureOpenAIEmbeddingService(
        client=fake_client,
        deployment="text-embedding-3-small",
    )


def test_empty_text_is_rejected():
    service = _service(FakeEmbeddingsClient())

    with pytest.raises(EmbeddingServiceError, match="empty"):
        service.embed_text("   ")


def test_embedding_order_is_preserved():
    service = _service(FakeEmbeddingsClient(vectors=[[9.0], [8.0]]))

    vectors = service.embed_texts(["first", "second"])

    assert vectors == [[9.0], [8.0]]


def test_dimension_verification_succeeds():
    service = _service(FakeEmbeddingsClient(vectors=[[1.0, 2.0, 3.0]]))

    assert service.verify_embedding_dimensions(3) == 3


def test_dimension_mismatch_raises_error():
    service = _service(FakeEmbeddingsClient(vectors=[[1.0, 2.0]]))

    with pytest.raises(EmbeddingServiceError, match="dimension mismatch"):
        service.verify_embedding_dimensions(3)


def test_mocked_api_failure_is_handled():
    service = _service(FakeEmbeddingsClient(fail=True))

    with pytest.raises(EmbeddingServiceError, match="embedding request failed"):
        service.embed_text("hello")
