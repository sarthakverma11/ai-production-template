from src.search.index_schema import build_search_index


def _fields_by_name(index):
    return {field.name: field for field in index.fields}


def test_expected_fields_exist():
    index = build_search_index("policy-knowledge-index-v1", 1536, "profile", "algo")

    fields = _fields_by_name(index)

    assert {
        "id",
        "content",
        "content_vector",
        "source_file",
        "source_path",
        "document_version",
        "chunk_index",
        "chunking_strategy",
        "chunk_size",
        "chunk_overlap",
        "embedding_model",
        "index_version",
        "generated_at",
    }.issubset(fields)


def test_id_is_key():
    index = build_search_index("policy-knowledge-index-v1", 1536, "profile", "algo")

    assert _fields_by_name(index)["id"].key is True


def test_content_is_searchable():
    index = build_search_index("policy-knowledge-index-v1", 1536, "profile", "algo")

    assert _fields_by_name(index)["content"].searchable is True


def test_vector_field_dimensions_and_profile_are_set():
    index = build_search_index("policy-knowledge-index-v1", 1536, "policy-vector-profile", "algo")

    vector_field = _fields_by_name(index)["content_vector"]

    assert vector_field.vector_search_dimensions == 1536
    assert vector_field.vector_search_profile_name == "policy-vector-profile"


def test_metadata_fields_are_filterable_and_retrievable():
    index = build_search_index("policy-knowledge-index-v1", 1536, "profile", "algo")

    fields = _fields_by_name(index)

    for name in ["source_file", "document_version", "chunk_index", "index_version"]:
        assert fields[name].filterable is True
        assert getattr(fields[name], "hidden", False) is False
