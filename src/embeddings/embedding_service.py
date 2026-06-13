"""Azure OpenAI embedding service used by Lecture 3."""

import os
from typing import Any

from src.utils import load_project_env


class EmbeddingServiceError(RuntimeError):
    """Raised when embedding generation fails."""


class AzureOpenAIEmbeddingService:
    """Small wrapper around Azure OpenAI embeddings."""

    def __init__(
        self,
        client: Any | None = None,
        deployment: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
    ) -> None:
        load_project_env()
        self.deployment = deployment or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        resolved_endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        resolved_api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")

        if not self.deployment:
            raise EmbeddingServiceError("Missing AZURE_OPENAI_EMBEDDING_DEPLOYMENT.")

        if client is not None:
            self.client = client
            return

        if not resolved_endpoint:
            raise EmbeddingServiceError("Missing AZURE_OPENAI_ENDPOINT.")
        if not resolved_api_key:
            raise EmbeddingServiceError("Missing AZURE_OPENAI_API_KEY.")

        try:
            from openai import AzureOpenAI
        except ModuleNotFoundError as error:
            raise EmbeddingServiceError(
                "openai is required. Install dependencies with pip install -r requirements.txt."
            ) from error

        self.client = AzureOpenAI(
            azure_endpoint=resolved_endpoint,
            api_key=resolved_api_key,
            api_version="2024-02-01",
        )

    def embed_text(self, text: str) -> list[float]:
        """Embed one non-empty text value."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text values while preserving input order."""
        if not texts:
            return []

        for index, text in enumerate(texts, start=1):
            if not isinstance(text, str) or not text.strip():
                raise EmbeddingServiceError(f"Embedding input {index} is empty.")

        try:
            response = self.client.embeddings.create(
                model=self.deployment,
                input=texts,
            )
        except Exception as error:
            raise EmbeddingServiceError(
                f"Azure OpenAI embedding request failed: {type(error).__name__}"
            ) from error

        data = getattr(response, "data", None)
        if not data:
            raise EmbeddingServiceError("Azure OpenAI returned an empty embedding response.")

        embeddings_by_index: dict[int, list[float]] = {}
        for item_index, item in enumerate(data):
            embedding = getattr(item, "embedding", None)
            response_index = getattr(item, "index", item_index)
            if not embedding:
                raise EmbeddingServiceError("Azure OpenAI returned an empty embedding vector.")
            embeddings_by_index[int(response_index)] = [float(value) for value in embedding]

        try:
            return [embeddings_by_index[index] for index in range(len(texts))]
        except KeyError as error:
            raise EmbeddingServiceError("Azure OpenAI embedding response was missing an item.") from error

    def verify_embedding_dimensions(self, expected_dimensions: int) -> int:
        """Generate one test embedding and verify the configured dimension."""
        if expected_dimensions <= 0:
            raise EmbeddingServiceError("Expected embedding dimensions must be positive.")

        vector = self.embed_text("dimension check")
        actual_dimensions = len(vector)
        if actual_dimensions != expected_dimensions:
            raise EmbeddingServiceError(
                "Embedding dimension mismatch: "
                f"expected {expected_dimensions}, got {actual_dimensions}."
            )

        return actual_dimensions


def embed_text(text: str) -> list[float]:
    """Convenience function for one-off embedding calls."""
    return AzureOpenAIEmbeddingService().embed_text(text)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convenience function for one-off batch embedding calls."""
    return AzureOpenAIEmbeddingService().embed_texts(texts)


def verify_embedding_dimensions(expected_dimensions: int) -> int:
    """Convenience function for one-off dimension verification."""
    return AzureOpenAIEmbeddingService().verify_embedding_dimensions(expected_dimensions)
