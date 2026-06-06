# AI Production Template

This repository is the starting point for a production-style AI, ML, and LLMOps project. In Lecture 1, the project does not call a real ML model or LLM API. Instead, it establishes the engineering foundation that later lectures will extend.

## Current Status

```text
Lecture 1 complete: Production AI Project Foundation
```

The project currently demonstrates:

- project structure
- YAML-based configuration
- config loading
- logging
- output artifact creation
- basic tests
- Git-ready structure

## Why this project exists

Production AI is not just about model building. Real AI systems need structure, configuration, logs, tests, reproducibility, and clear documentation. This repository will grow lecture by lecture into a more complete MLOps/LLMOps project.

Think of this repo as the foundation of a building. Future lectures will add the rooms, wiring, security, monitoring, and automation.

## Project Structure

```text
ai-production-template/
├── configs/
│   └── config.yaml
├── data/
│   └── .gitkeep
├── notebooks/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── config_loader.py
│   ├── logger.py
│   └── utils.py
├── tests/
│   └── test_project_structure.py
├── outputs/
│   └── .gitkeep
├── logs/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── docs/
│   └── lecture_01_foundation.md
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

Python 3.10 or above is recommended.

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the project

```bash
python main.py
```

This loads `configs/config.yaml`, initializes logging, sets the seed, simulates a project run, saves an output artifact, and writes logs.

Expected generated files:

- `outputs/run_summary.txt`
- `logs/lecture_1_demo.log`

## Run tests

```bash
pytest
```

The current tests validate that the required project folders and files exist.

## Lecture Documentation

Detailed lecture notes are available in:

```text
docs/lecture_01_foundation.md
```

[Lecture 1: Production AI Project Foundation](docs/lecture_01_foundation.md)

## Roadmap

| Future Topic             | How it will extend this project                    |
| ------------------------ | -------------------------------------------------- |
| Data validation          | Add validation scripts and data quality checks     |
| Data/document versioning | Add dataset or document version tracking           |
| ML training pipeline     | Add `src/training/` and model training code        |
| Experiment tracking      | Add MLflow or similar tracking                     |
| LLM API integration      | Add `src/llm/` for provider calls                  |
| Prompt management        | Add prompt templates and prompt versioning         |
| RAG pipeline             | Add ingestion, chunking, embeddings, and retrieval |
| Evaluation               | Add `src/evaluation/` for ML/LLM evaluation        |
| Docker                   | Add `Dockerfile` and container instructions        |
| CI/CD                    | Add `.github/workflows/`                           |
| Cloud deployment         | Add `deploy/` or `infra/`                          |
| Monitoring               | Add metrics, tracing, and observability            |

## Git workflow

```bash
git status
git add .
git commit -m "Create Lecture 1 production AI project foundation"
```

Each lecture should end with a meaningful commit so learners can track how the project evolves.

## Important note

There is no real LLM integration in Lecture 1. This is intentional. API keys, billing, authentication, and quota issues are avoided so learners can focus on production project foundations first.

## Summary

This repository will grow gradually through the course. Lecture 1 creates the clean foundation. Future lectures will add real ML, LLMOps, RAG, evaluation, deployment, CI/CD, and monitoring capabilities.
