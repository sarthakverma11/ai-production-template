"""Validate loaded documents before processing."""

from pathlib import Path
from typing import Any


REQUIRED_DOCUMENT_FIELDS = {
    "source_file",
    "source_path",
    "document_version",
    "text",
}


class DocumentValidationError(ValueError):
    """Raised when document validation fails."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        message = "Document validation failed: " + "; ".join(report["errors"])
        super().__init__(message)


def validate_documents(
    documents: list[dict[str, Any]],
    input_folder: str | Path,
    supported_extensions: list[str],
    minimum_document_length: int,
) -> dict[str, Any]:
    """Validate loaded documents and return a structured report."""
    errors: list[str] = []
    folder = Path(input_folder)
    normalized_extensions = {extension.lower() for extension in supported_extensions}

    if not folder.exists():
        errors.append(f"Input folder does not exist: {folder}")
    elif not folder.is_dir():
        errors.append(f"Input path is not a folder: {folder}")
    else:
        supported_files = [
            file_path
            for file_path in folder.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in normalized_extensions
        ]
        if not supported_files:
            errors.append(
                f"No supported documents found in {folder}. "
                f"Supported extensions: {sorted(normalized_extensions)}"
            )

    seen_source_files: set[str] = set()
    valid_files = 0

    for document_index, document in enumerate(documents, start=1):
        missing_fields = REQUIRED_DOCUMENT_FIELDS - set(document)
        if missing_fields:
            errors.append(
                f"Document {document_index} is missing required metadata: "
                f"{sorted(missing_fields)}"
            )

        source_file = str(document.get("source_file", ""))
        if source_file:
            if source_file in seen_source_files:
                errors.append(f"Duplicate source filename found: {source_file}")
            seen_source_files.add(source_file)

        file_extension = str(document.get("file_extension", Path(source_file).suffix)).lower()
        if file_extension and file_extension not in normalized_extensions:
            errors.append(
                f"Unsupported file extension for {source_file}: {file_extension}"
            )

        text = document.get("text", "")
        if not isinstance(text, str):
            errors.append(f"Document text must be a string for {source_file or document_index}")
            continue

        stripped_text = text.strip()
        if not stripped_text:
            errors.append(f"Document text is empty for {source_file or document_index}")
            continue

        if len(stripped_text) < minimum_document_length:
            errors.append(
                f"Document text is shorter than {minimum_document_length} characters "
                f"for {source_file or document_index}"
            )
            continue

        if not missing_fields and file_extension in normalized_extensions:
            valid_files += 1

    report = {
        "status": "failed" if errors else "passed",
        "files_checked": len(documents),
        "valid_files": valid_files if not errors else 0,
        "invalid_files": len(documents) - valid_files if errors else 0,
        "errors": errors,
    }

    if errors:
        raise DocumentValidationError(report)

    return report

