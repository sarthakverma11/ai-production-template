"""Load local text documents with basic metadata."""

from pathlib import Path
from typing import Any


def load_documents(
    folder_path: str | Path,
    document_version: str,
    supported_extensions: list[str],
) -> list[dict[str, Any]]:
    """Load supported UTF-8 documents from a folder."""
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Document folder not found: {folder}")

    if not folder.is_dir():
        raise NotADirectoryError(f"Document path is not a folder: {folder}")

    normalized_extensions = {extension.lower() for extension in supported_extensions}
    documents: list[dict[str, Any]] = []

    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file():
            continue

        file_extension = file_path.suffix.lower()
        if file_extension not in normalized_extensions:
            continue

        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise UnicodeDecodeError(
                error.encoding,
                error.object,
                error.start,
                error.end,
                f"Could not read {file_path} as UTF-8: {error.reason}",
            ) from error

        documents.append(
            {
                "source_file": file_path.name,
                "source_path": str(file_path.as_posix()),
                "document_version": document_version,
                "file_extension": file_extension,
                "text": text,
            }
        )

    return documents

