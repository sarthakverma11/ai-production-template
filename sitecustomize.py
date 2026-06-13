"""Process startup tweaks for Windows classroom machines.

Pytest uses the operating-system temp directory for tmp_path fixtures. On some
Windows setups that folder, or a previous pytest temp folder, can be locked by
permissions, an IDE, or antivirus. Give each pytest process its own temp root.
"""

import os
import sys
import tempfile
from pathlib import Path


def _running_pytest() -> bool:
    argv_text = " ".join(sys.argv).lower()
    return "pytest" in argv_text


if _running_pytest():
    project_root = Path(__file__).resolve().parent
    temp_root = project_root / ".pytest_runtime" / f"session_{os.getpid()}"
    temp_root.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(temp_root)
    os.environ["TEMP"] = str(temp_root)
    tempfile.tempdir = str(temp_root)
