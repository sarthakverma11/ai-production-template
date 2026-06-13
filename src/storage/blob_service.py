"""Beginner-friendly Azure Blob Storage helpers for Lecture 3."""

import json
import os
from pathlib import Path
from typing import Any

from src.utils import load_project_env


class BlobStorageError(RuntimeError):
    """Raised when Blob Storage operations cannot be completed."""


def get_blob_service_client(connection_string: str | None = None) -> Any:
    """Create a BlobServiceClient using the configured connection string."""
    load_project_env()
    resolved_connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not resolved_connection_string:
        raise BlobStorageError("Missing AZURE_STORAGE_CONNECTION_STRING.")

    try:
        from azure.storage.blob import BlobServiceClient
    except ModuleNotFoundError as error:
        raise BlobStorageError(
            "azure-storage-blob is required. Install dependencies with pip install -r requirements.txt."
        ) from error

    return BlobServiceClient.from_connection_string(resolved_connection_string)


def validate_containers_exist(
    container_names: list[str],
    client: Any | None = None,
) -> None:
    """Validate that all required containers already exist."""
    blob_service_client = client or get_blob_service_client()
    missing_containers: list[str] = []

    for container_name in container_names:
        container_client = blob_service_client.get_container_client(container_name)
        if not container_client.exists():
            missing_containers.append(container_name)

    if missing_containers:
        raise BlobStorageError(
            "Missing Azure Blob containers: " + ", ".join(sorted(missing_containers))
        )


def upload_file(
    container_name: str,
    blob_name: str,
    local_path: str | Path,
    overwrite: bool = False,
    client: Any | None = None,
) -> None:
    """Upload a local file to Blob Storage without overwriting by default."""
    path = Path(local_path)
    if not path.exists():
        raise BlobStorageError(f"Local file not found: {path}")

    blob_service_client = client or get_blob_service_client()
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name,
    )

    if not overwrite and blob_client.exists():
        raise BlobStorageError(
            f"Blob already exists and overwrite=False: {container_name}/{blob_name}"
        )

    with path.open("rb") as file:
        blob_client.upload_blob(file, overwrite=overwrite)


def load_json_blob(
    container_name: str,
    blob_name: str,
    client: Any | None = None,
) -> dict[str, Any]:
    """Read a UTF-8 JSON blob and return it as a dictionary."""
    blob_service_client = client or get_blob_service_client()
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name,
    )

    if not blob_client.exists():
        raise BlobStorageError(f"Blob not found: {container_name}/{blob_name}")

    try:
        raw_bytes = blob_client.download_blob().readall()
        data = json.loads(raw_bytes.decode("utf-8"))
    except json.JSONDecodeError as error:
        raise BlobStorageError(f"Invalid JSON blob: {container_name}/{blob_name}") from error

    if not isinstance(data, dict):
        raise BlobStorageError(f"JSON blob must contain an object: {container_name}/{blob_name}")

    return data


def list_blobs(
    container_name: str,
    prefix: str | None = None,
    client: Any | None = None,
) -> list[str]:
    """List blob names in a container, optionally filtered by prefix."""
    blob_service_client = client or get_blob_service_client()
    container_client = blob_service_client.get_container_client(container_name)

    if not container_client.exists():
        raise BlobStorageError(f"Container not found: {container_name}")

    return sorted(blob.name for blob in container_client.list_blobs(name_starts_with=prefix))
