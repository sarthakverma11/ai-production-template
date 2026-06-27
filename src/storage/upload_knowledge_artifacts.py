"""Upload Lecture 2 knowledge artifacts for the Lecture 5 CI pipeline."""

import argparse
from pathlib import Path
from typing import Any

from src.config_loader import load_config
from src.storage.blob_service import upload_file


DEFAULT_CONFIG_PATH = "configs/config.yaml"


def upload_knowledge_artifacts(
    config_path: str = DEFAULT_CONFIG_PATH,
    overwrite: bool = True,
    client: Any | None = None,
) -> dict[str, Any]:
    """Upload processed chunks and lineage metadata to Azure Blob Storage."""
    config = load_config(config_path)
    data_config = config["data"]
    processing_config = config["processing"]
    storage_config = config["azure_storage"]

    artifacts = [
        {
            "name": "processed_chunks",
            "container_name": storage_config["processed_container"],
            "blob_name": storage_config["chunks_blob"],
            "local_path": Path(data_config["processed_dir"]) / processing_config["chunks_output_file"],
        },
        {
            "name": "lineage_metadata",
            "container_name": storage_config["metadata_container"],
            "blob_name": storage_config["lineage_blob"],
            "local_path": Path(data_config["metadata_dir"]) / processing_config["lineage_output_file"],
        },
    ]

    uploaded_artifacts = []
    for artifact in artifacts:
        upload_file(
            container_name=artifact["container_name"],
            blob_name=artifact["blob_name"],
            local_path=artifact["local_path"],
            overwrite=overwrite,
            client=client,
        )
        uploaded_artifacts.append(
            {
                "name": artifact["name"],
                "container_name": artifact["container_name"],
                "blob_name": artifact["blob_name"],
                "local_path": artifact["local_path"].as_posix(),
            }
        )

    return {"uploaded_artifacts": uploaded_artifacts, "overwrite": overwrite}


def main() -> None:
    """CLI entry point for uploading refreshed knowledge artifacts."""
    parser = argparse.ArgumentParser(description="Upload Lecture 2 artifacts to Azure Blob Storage.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.yaml.")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if the target blob already exists.",
    )
    args = parser.parse_args()

    result = upload_knowledge_artifacts(
        config_path=args.config,
        overwrite=not args.no_overwrite,
    )
    for artifact in result["uploaded_artifacts"]:
        print(
            "Uploaded "
            f"{artifact['local_path']} -> "
            f"{artifact['container_name']}/{artifact['blob_name']}"
        )


if __name__ == "__main__":
    main()
