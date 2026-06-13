from types import SimpleNamespace

import pytest

from src.storage.blob_service import BlobStorageError, list_blobs, load_json_blob


class FakeBlobClient:
    def __init__(self, exists=True, payload=b'{"ok": true}'):
        self._exists = exists
        self._payload = payload

    def exists(self):
        return self._exists

    def download_blob(self):
        return SimpleNamespace(readall=lambda: self._payload)


class FakeContainerClient:
    def __init__(self, exists=True):
        self._exists = exists

    def exists(self):
        return self._exists

    def list_blobs(self, name_starts_with=None):
        names = ["policies_v1/leave_policy.txt", "other/file.txt"]
        return [
            SimpleNamespace(name=name)
            for name in names
            if name_starts_with is None or name.startswith(name_starts_with)
        ]


class FakeBlobServiceClient:
    def __init__(self, blob_client=None, container_client=None):
        self.blob_client = blob_client or FakeBlobClient()
        self.container_client = container_client or FakeContainerClient()
        self.requested = []

    def get_blob_client(self, container, blob):
        self.requested.append((container, blob))
        return self.blob_client

    def get_container_client(self, container):
        self.requested.append((container, None))
        return self.container_client


def test_load_json_blob_uses_container_and_blob_names():
    client = FakeBlobServiceClient()

    data = load_json_blob("processed-artifacts", "chunks_v1.json", client=client)

    assert data == {"ok": True}
    assert client.requested == [("processed-artifacts", "chunks_v1.json")]


def test_missing_blob_raises_clear_error():
    client = FakeBlobServiceClient(blob_client=FakeBlobClient(exists=False))

    with pytest.raises(BlobStorageError, match="Blob not found"):
        load_json_blob("processed-artifacts", "chunks_v1.json", client=client)


def test_invalid_json_raises_clear_error():
    client = FakeBlobServiceClient(blob_client=FakeBlobClient(payload=b"not-json"))

    with pytest.raises(BlobStorageError, match="Invalid JSON blob"):
        load_json_blob("processed-artifacts", "chunks_v1.json", client=client)


def test_list_blobs_filters_by_prefix():
    client = FakeBlobServiceClient()

    blobs = list_blobs("raw-documents", prefix="policies_v1", client=client)

    assert blobs == ["policies_v1/leave_policy.txt"]
