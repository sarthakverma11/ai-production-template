"""Shared pytest configuration for this project."""

import os
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PYTEST_TEMP_ROOT = PROJECT_ROOT / ".pytest_runtime" / f"session_{os.getpid()}"

PYTEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = str(PYTEST_TEMP_ROOT)
os.environ["TEMP"] = str(PYTEST_TEMP_ROOT)
tempfile.tempdir = str(PYTEST_TEMP_ROOT)
