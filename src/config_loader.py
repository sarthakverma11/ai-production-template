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

    _validate_optional_positive_int(config, "embedding", "dimensions")
    _validate_optional_positive_int(config, "embedding", "batch_size")
    _validate_optional_positive_int(config, "search", "top_k")

    return config


def _validate_optional_positive_int(
    config: dict[str, Any],
    section_name: str,
    key: str,
) -> None:
    """Validate optional Lecture 3 integer settings when present."""
    section = config.get(section_name)
    if section is None:
        return

    if not isinstance(section, dict):
        raise ValueError(f"Config section '{section_name}' must be a dictionary.")

    if key not in section:
        return

    value = section[key]
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Config value '{section_name}.{key}' must be a positive integer.")
