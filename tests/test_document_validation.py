from pathlib import Path

import pytest

from src.ingestion.document_loader import load_documents
from src.ingestion.validate_documents import DocumentValidationError, validate_documents


def test_valid_documents_pass(tmp_path: Path):
    policy_file = tmp_path / "leave_policy.txt"
    policy_file.write_text("A valid document with enough useful policy text for testing.", encoding="utf-8")
    documents = load_documents(tmp_path, "v1", [".txt"])

    report = validate_documents(documents, tmp_path, [".txt"], minimum_document_length=20)

    assert report["status"] == "passed"
    assert report["valid_files"] == 1
    assert report["errors"] == []


def test_empty_document_fails(tmp_path: Path):
    policy_file = tmp_path / "empty_policy.txt"
    policy_file.write_text("", encoding="utf-8")
    documents = load_documents(tmp_path, "v1", [".txt"])

    with pytest.raises(DocumentValidationError) as error:
        validate_documents(documents, tmp_path, [".txt"], minimum_document_length=20)

    assert error.value.report["status"] == "failed"
    assert "empty" in " ".join(error.value.report["errors"]).lower()


def test_too_short_document_fails(tmp_path: Path):
    policy_file = tmp_path / "short_policy.txt"
    policy_file.write_text("Too short.", encoding="utf-8")
    documents = load_documents(tmp_path, "v1", [".txt"])

    with pytest.raises(DocumentValidationError) as error:
        validate_documents(documents, tmp_path, [".txt"], minimum_document_length=20)

    assert "shorter" in " ".join(error.value.report["errors"]).lower()


def test_missing_metadata_fails(tmp_path: Path):
    policy_file = tmp_path / "policy.txt"
    policy_file.write_text("A valid document exists so folder-level checks pass.", encoding="utf-8")
    documents = [
        {
            "source_file": "policy.txt",
            "document_version": "v1",
            "text": "A valid document with enough text for validation.",
        }
    ]

    with pytest.raises(DocumentValidationError) as error:
        validate_documents(documents, tmp_path, [".txt"], minimum_document_length=20)

    assert "source_path" in " ".join(error.value.report["errors"])

