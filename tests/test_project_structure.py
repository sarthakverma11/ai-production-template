from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_folders_exist():
    required_folders = [
        "app",
        "configs",
        "data",
        "data/raw",
        "data/raw/policies_v1",
        "data/processed",
        "data/metadata",
        "notebooks",
        "src",
        "src/ingestion",
        "src/processing",
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
        "app/streamlit_app.py",
        "data/raw/policies_v1/leave_policy.txt",
        "data/raw/policies_v1/travel_policy.txt",
        "data/raw/policies_v1/it_support_faq.txt",
        "docs/lecture_02_document_processing.md",
        "main.py",
        "pytest.ini",
        "requirements.txt",
        "README.md",
        ".dvcignore",
    ]

    for file_path in required_files:
        assert (PROJECT_ROOT / file_path).is_file(), f"Missing file: {file_path}"
