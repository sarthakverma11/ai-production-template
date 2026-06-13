"""Small utility functions used by the project."""

import random
from pathlib import Path

import numpy as np


def seed_everything(seed: int) -> None:
    """Set random seeds so repeated runs behave consistently."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_text_artifact(file_path: str, content: str) -> None:
    """Save text content to a file, creating parent folders when needed."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_project_env() -> None:
    """Load local environment variables from .env when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return

    load_dotenv()
