from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_folders_exist():
    required_folders = [
        "configs",
        "data",
        "notebooks",
        "src",
        "tests",
        "outputs",
        "logs",
        "models",
    ]

    for folder in required_folders:
        assert (PROJECT_ROOT / folder).is_dir(), f"Missing folder: {folder}"


def test_required_files_exist():
    required_files = [
        "configs/config.yaml",
        "main.py",
        "requirements.txt",
        "README.md",
    ]

    for file_path in required_files:
        assert (PROJECT_ROOT / file_path).is_file(), f"Missing file: {file_path}"
