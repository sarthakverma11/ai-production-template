from pathlib import Path

from src.ingestion.document_loader import load_documents


def test_valid_files_load_successfully(tmp_path: Path):
    policy_file = tmp_path / "leave_policy.txt"
    policy_file.write_text("This is a valid policy document for testing.", encoding="utf-8")

    documents = load_documents(tmp_path, "v1", [".txt"])

    assert len(documents) == 1
    assert documents[0]["source_file"] == "leave_policy.txt"
    assert documents[0]["text"] == "This is a valid policy document for testing."


def test_metadata_fields_exist(tmp_path: Path):
    policy_file = tmp_path / "travel_policy.txt"
    policy_file.write_text("Travel policy content for metadata testing.", encoding="utf-8")

    document = load_documents(tmp_path, "v1", [".txt"])[0]

    assert document["source_path"].endswith("travel_policy.txt")
    assert document["document_version"] == "v1"
    assert document["file_extension"] == ".txt"
    assert "text" in document


def test_unsupported_files_are_not_treated_as_valid(tmp_path: Path):
    supported_file = tmp_path / "policy.txt"
    unsupported_file = tmp_path / "notes.md"
    supported_file.write_text("Supported text file.", encoding="utf-8")
    unsupported_file.write_text("Unsupported markdown file.", encoding="utf-8")

    documents = load_documents(tmp_path, "v1", [".txt"])

    assert [document["source_file"] for document in documents] == ["policy.txt"]

