"""Azure AI Search index lifecycle helpers."""

import os
from urllib.parse import urlparse
from typing import Any

from src.search.index_schema import build_search_index
from src.utils import load_project_env


class SearchIndexManagerError(RuntimeError):
    """Raised when the Azure AI Search index cannot be managed."""


class SearchIndexManager:
    """Create or reuse the Lecture 3 vector index."""

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        index_name: str | None = None,
        client: Any | None = None,
        logger: Any | None = None,
    ) -> None:
        load_project_env()
        self.endpoint = endpoint or os.getenv("AZURE_SEARCH_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_SEARCH_API_KEY")
        self.index_name = index_name or os.getenv("AZURE_SEARCH_INDEX_NAME")
        self.logger = logger

        if not self.index_name:
            raise SearchIndexManagerError("Missing AZURE_SEARCH_INDEX_NAME.")

        if client is not None:
            self.client = client
            return

        if not self.endpoint:
            raise SearchIndexManagerError("Missing AZURE_SEARCH_ENDPOINT.")
        if not self.api_key:
            raise SearchIndexManagerError("Missing AZURE_SEARCH_API_KEY.")

        try:
            from azure.core.credentials import AzureKeyCredential
            from azure.search.documents.indexes import SearchIndexClient
        except ModuleNotFoundError as error:
            raise SearchIndexManagerError(
                "azure-search-documents is required. Install dependencies with pip install -r requirements.txt."
            ) from error

        self.client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key),
        )

    def index_exists(self) -> bool:
        """Return True when the configured index exists."""
        try:
            self.client.get_index(self.index_name)
            return True
        except Exception as error:
            if _looks_like_not_found(error):
                return False
            raise SearchIndexManagerError(
                f"Could not check search index existence: {type(error).__name__}"
            ) from error

    def create_index_if_absent(
        self,
        dimensions: int,
        vector_profile_name: str,
        vector_algorithm_name: str,
        recreate: bool = False,
    ) -> str:
        """Create the index only when absent, unless recreate=True is explicit."""
        index = build_search_index(
            index_name=self.index_name,
            dimensions=dimensions,
            vector_profile_name=vector_profile_name,
            vector_algorithm_name=vector_algorithm_name,
        )

        endpoint_host = urlparse(self.endpoint or "").netloc or "injected-client"
        _log(
            self.logger,
            "Preparing search index '%s' on '%s' with %s dimensions.",
            self.index_name,
            endpoint_host,
            dimensions,
        )

        exists = self.index_exists()
        if exists and not recreate:
            existing_index = self.client.get_index(self.index_name)
            self.validate_existing_index_schema(existing_index, dimensions)
            _log(self.logger, "Search index reused: %s", self.index_name)
            return "reused"

        if exists and recreate:
            self.client.delete_index(self.index_name)
            _log(self.logger, "Search index deleted for explicit recreation: %s", self.index_name)

        self.client.create_or_update_index(index)
        _log(self.logger, "Search index created or updated: %s", self.index_name)
        return "created" if not exists else "recreated"

    def validate_existing_index_schema(self, index: Any, dimensions: int) -> None:
        """Validate the existing index enough to catch common classroom mistakes."""
        fields = {field.name: field for field in getattr(index, "fields", [])}
        required_fields = {"id", "content", "content_vector"}
        missing_fields = sorted(required_fields - set(fields))
        if missing_fields:
            raise SearchIndexManagerError(
                "Existing index is missing required fields: " + ", ".join(missing_fields)
            )

        vector_field = fields["content_vector"]
        actual_dimensions = getattr(vector_field, "vector_search_dimensions", None)
        if actual_dimensions is not None and int(actual_dimensions) != dimensions:
            raise SearchIndexManagerError(
                "Existing index vector dimension mismatch: "
                f"expected {dimensions}, got {actual_dimensions}."
            )


def _looks_like_not_found(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code == 404:
        return True
    return type(error).__name__ in {"ResourceNotFoundError", "HttpResponseError"} and "not found" in str(error).lower()


def _log(logger: Any | None, message: str, *args: Any) -> None:
    if logger is not None:
        logger.info(message, *args)
