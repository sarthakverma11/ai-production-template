from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.evaluation.check_quality_gate import QualityGateError, check_quality_gate
from src.storage.upload_knowledge_artifacts import upload_knowledge_artifacts


class FakeUploadBlobClient:
    def __init__(self):
        self.uploads = []

    def exists(self):
        return False

    def upload_blob(self, file, overwrite=False):
        self.uploads.append(
            {
                "content": file.read().decode("utf-8"),
                "overwrite": overwrite,
            }
        )


class FakeUploadBlobServiceClient:
    def __init__(self):
        self.blob_clients = {}
        self.requested = []

    def get_blob_client(self, container, blob):
        self.requested.append((container, blob))
        key = (container, blob)
        self.blob_clients.setdefault(key, FakeUploadBlobClient())
        return self.blob_clients[key]


def test_upload_knowledge_artifacts_uploads_chunks_and_lineage(tmp_path):
    processed_dir = tmp_path / "processed"
    metadata_dir = tmp_path / "metadata"
    processed_dir.mkdir()
    metadata_dir.mkdir()
    (processed_dir / "chunks_v1.json").write_text('{"chunks": []}', encoding="utf-8")
    (metadata_dir / "lineage_v1.json").write_text('{"pipeline_name": "demo"}', encoding="utf-8")
    config_path = _write_test_config(tmp_path, processed_dir, metadata_dir)
    client = FakeUploadBlobServiceClient()

    result = upload_knowledge_artifacts(config_path=str(config_path), client=client)

    assert result["overwrite"] is True
    assert client.requested == [
        ("processed-artifacts", "chunks_v1.json"),
        ("metadata", "lineage_v1.json"),
    ]
    assert client.blob_clients[("processed-artifacts", "chunks_v1.json")].uploads == [
        {"content": '{"chunks": []}', "overwrite": True}
    ]
    assert client.blob_clients[("metadata", "lineage_v1.json")].uploads == [
        {"content": '{"pipeline_name": "demo"}', "overwrite": True}
    ]


def test_quality_gate_passes_at_threshold(tmp_path):
    summary_path = _write_summary(tmp_path, overall_pass_rate=0.80)

    result = check_quality_gate(summary_path=summary_path)

    assert result["passed"] is True
    assert result["overall_pass_rate"] == 0.80


def test_quality_gate_fails_below_threshold(tmp_path):
    summary_path = _write_summary(tmp_path, overall_pass_rate=0.79)

    with pytest.raises(QualityGateError, match="Quality gate failed"):
        check_quality_gate(summary_path=summary_path)


def test_quality_gate_missing_summary_file_gives_clear_error(tmp_path):
    missing_path = tmp_path / "missing_summary.csv"

    with pytest.raises(QualityGateError, match="Evaluation summary not found"):
        check_quality_gate(summary_path=missing_path)


def _write_summary(tmp_path: Path, overall_pass_rate: float) -> Path:
    summary_path = tmp_path / "comparison_summary.csv"
    summary_path.write_text(
        "prompt_version,total_questions,overall_pass_rate\n"
        f"qa_prompt_v2,5,{overall_pass_rate}\n",
        encoding="utf-8",
    )
    return summary_path


def _write_test_config(tmp_path: Path, processed_dir: Path, metadata_dir: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config = {
        "data": {
            "processed_dir": processed_dir.as_posix(),
            "metadata_dir": metadata_dir.as_posix(),
        },
        "processing": {
            "chunks_output_file": "chunks_v1.json",
            "lineage_output_file": "lineage_v1.json",
        },
        "azure_storage": {
            "processed_container": "processed-artifacts",
            "chunks_blob": "chunks_v1.json",
            "metadata_container": "metadata",
            "lineage_blob": "lineage_v1.json",
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path
