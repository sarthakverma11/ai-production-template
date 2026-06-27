"""Prompt registry and prompt text loading for Lecture 4."""

from pathlib import Path
from typing import Any

import yaml


DEFAULT_REGISTRY_PATH = Path("prompts/prompt_registry.yaml")


class PromptLoaderError(RuntimeError):
    """Raised when prompt registry or prompt text loading fails."""


def load_prompt(
    prompt_version: str | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> tuple[str, dict[str, Any]]:
    """Load prompt text and metadata for the requested or active prompt version."""
    registry = load_prompt_registry(registry_path)
    prompt_metadata = find_prompt_metadata(registry, prompt_version)
    prompt_path = Path(prompt_metadata["file_path"])

    if not prompt_path.exists():
        raise PromptLoaderError(f"Prompt file not found: {prompt_path}")

    prompt_text = prompt_path.read_text(encoding="utf-8")
    return prompt_text, prompt_metadata


def load_prompt_registry(registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    """Load prompt_registry.yaml as a dictionary."""
    path = Path(registry_path)
    if not path.exists():
        raise PromptLoaderError(f"Prompt registry not found: {path}")

    try:
        registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise PromptLoaderError(f"Invalid prompt registry YAML: {path}") from error

    if not isinstance(registry, dict):
        raise PromptLoaderError("Prompt registry must be a YAML dictionary.")
    if not isinstance(registry.get("prompts"), dict):
        raise PromptLoaderError("Prompt registry must contain a 'prompts' dictionary.")

    return registry


def find_prompt_metadata(
    registry: dict[str, Any],
    prompt_version: str | None = None,
) -> dict[str, Any]:
    """Return metadata for the requested prompt or the active prompt."""
    selected_version = prompt_version or registry.get("active_prompt")
    if not selected_version:
        raise PromptLoaderError("No prompt version requested and no active_prompt configured.")

    prompts = registry["prompts"]
    metadata = prompts.get(selected_version)
    if metadata is None:
        for candidate in prompts.values():
            if candidate.get("prompt_version") == selected_version:
                metadata = candidate
                break

    if not isinstance(metadata, dict):
        raise PromptLoaderError(f"Prompt version not found in registry: {selected_version}")
    if "file_path" not in metadata:
        raise PromptLoaderError(f"Prompt metadata missing file_path: {selected_version}")

    return dict(metadata)


def format_prompt(prompt_text: str, context: str, question: str) -> str:
    """Safely format a prompt with the two variables used in this demo."""
    if "{context}" not in prompt_text or "{question}" not in prompt_text:
        raise PromptLoaderError("Prompt text must include both {context} and {question}.")

    try:
        return prompt_text.format(context=context.strip(), question=question.strip())
    except KeyError as error:
        raise PromptLoaderError(f"Prompt contains an unknown placeholder: {error}") from error
