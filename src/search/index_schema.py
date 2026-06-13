"""Azure AI Search index schema for Lecture 3 vector retrieval."""

from dataclasses import dataclass
from typing import Any


@dataclass
class FallbackField:
    """Small field object used only when Azure SDK is not installed for local tests."""

    name: str
    type: str
    key: bool = False
    searchable: bool = False
    filterable: bool = False
    hidden: bool = False
    vector_search_dimensions: int | None = None
    vector_search_profile_name: str | None = None


@dataclass
class FallbackIndex:
    """Small index object used only when Azure SDK is not installed for local tests."""

    name: str
    fields: list[FallbackField]
    vector_search: Any


def build_search_index(
    index_name: str,
    dimensions: int,
    vector_profile_name: str,
    vector_algorithm_name: str,
) -> Any:
    """Build the policy knowledge index schema."""
    if not index_name:
        raise ValueError("Search index name is required.")
    if dimensions <= 0:
        raise ValueError("Vector dimensions must be positive.")
    if not vector_profile_name:
        raise ValueError("Vector profile name is required.")
    if not vector_algorithm_name:
        raise ValueError("Vector algorithm name is required.")

    try:
        return _build_azure_search_index(
            index_name=index_name,
            dimensions=dimensions,
            vector_profile_name=vector_profile_name,
            vector_algorithm_name=vector_algorithm_name,
        )
    except ModuleNotFoundError:
        return _build_fallback_index(
            index_name=index_name,
            dimensions=dimensions,
            vector_profile_name=vector_profile_name,
            vector_algorithm_name=vector_algorithm_name,
        )


def _build_azure_search_index(
    index_name: str,
    dimensions: int,
    vector_profile_name: str,
    vector_algorithm_name: str,
) -> Any:
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SearchableField,
        SimpleField,
        VectorSearch,
        VectorSearchProfile,
    )

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
            hidden=False,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            hidden=False,
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            hidden=True,
            vector_search_dimensions=dimensions,
            vector_search_profile_name=vector_profile_name,
        ),
    ]

    fields.extend(
        [
            SimpleField(name="source_file", type=SearchFieldDataType.String, filterable=True, hidden=False),
            SimpleField(name="source_path", type=SearchFieldDataType.String, filterable=True, hidden=False),
            SimpleField(name="document_version", type=SearchFieldDataType.String, filterable=True, hidden=False),
            SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True, hidden=False),
            SimpleField(name="chunking_strategy", type=SearchFieldDataType.String, filterable=True, hidden=False),
            SimpleField(name="chunk_size", type=SearchFieldDataType.Int32, filterable=True, hidden=False),
            SimpleField(name="chunk_overlap", type=SearchFieldDataType.Int32, filterable=True, hidden=False),
            SimpleField(name="embedding_model", type=SearchFieldDataType.String, filterable=True, hidden=False),
            SimpleField(name="index_version", type=SearchFieldDataType.String, filterable=True, hidden=False),
            SimpleField(name="generated_at", type=SearchFieldDataType.String, filterable=True, hidden=False),
        ]
    )

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name=vector_algorithm_name),
        ],
        profiles=[
            VectorSearchProfile(
                name=vector_profile_name,
                algorithm_configuration_name=vector_algorithm_name,
            ),
        ],
    )

    return SearchIndex(name=index_name, fields=fields, vector_search=vector_search)


def _build_fallback_index(
    index_name: str,
    dimensions: int,
    vector_profile_name: str,
    vector_algorithm_name: str,
) -> FallbackIndex:
    fields = [
        FallbackField("id", "Edm.String", key=True, filterable=True),
        FallbackField("content", "Edm.String", searchable=True),
        FallbackField(
            "content_vector",
            "Collection(Edm.Single)",
            searchable=True,
            hidden=True,
            vector_search_dimensions=dimensions,
            vector_search_profile_name=vector_profile_name,
        ),
        FallbackField("source_file", "Edm.String", filterable=True),
        FallbackField("source_path", "Edm.String", filterable=True),
        FallbackField("document_version", "Edm.String", filterable=True),
        FallbackField("chunk_index", "Edm.Int32", filterable=True),
        FallbackField("chunking_strategy", "Edm.String", filterable=True),
        FallbackField("chunk_size", "Edm.Int32", filterable=True),
        FallbackField("chunk_overlap", "Edm.Int32", filterable=True),
        FallbackField("embedding_model", "Edm.String", filterable=True),
        FallbackField("index_version", "Edm.String", filterable=True),
        FallbackField("generated_at", "Edm.String", filterable=True),
    ]
    vector_search = {
        "algorithm": vector_algorithm_name,
        "profile": vector_profile_name,
    }
    return FallbackIndex(name=index_name, fields=fields, vector_search=vector_search)
