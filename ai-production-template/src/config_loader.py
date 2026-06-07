"""Simple YAML configuration loader for the project."""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """Load a YAML config file and return it as a Python dictionary."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML config file: {path}") from error

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML dictionary at the top level.")

    return config
