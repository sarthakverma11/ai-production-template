# Lecture 1: Production AI Project Foundation

## 1. Purpose of this lecture

The goal of Lecture 1 is not to build a real AI model or LLM application. The goal is to create the engineering foundation needed for production AI projects.

Before adding ML models, LLM APIs, RAG pipelines, Docker, CI/CD, or monitoring, we need a clean project structure. This structure makes future work easier to explain, easier to test, and easier to maintain.

## 2. What we built

We created a base project with:

- configuration file
- reusable source code folder
- logging utility
- utility functions
- output folder
- logs folder
- tests folder
- README documentation
- Git-ready structure

## 3. Why no real LLM is used yet

Lecture 1 avoids real LLM API calls because API keys, billing, authentication, quotas, and network issues can distract from the core lesson.

The current project uses a simulated AI run. This allows learners to focus on structure, configuration, logging, reproducibility, and testing.

## 4. Key concepts demonstrated

### 4.1 Project structure

Folders are separated so each part of the project has a clear responsibility. Configuration lives in `configs/`, reusable Python code lives in `src/`, generated files go to `outputs/`, logs go to `logs/`, and tests live in `tests/`.

This separation helps learners avoid mixing notebooks, scripts, generated files, and source code in one place.

### 4.2 Configuration-driven execution

Settings are stored in `configs/config.yaml` instead of being hardcoded in `main.py`.

Changing a value in config should change program behavior without rewriting core code. For example, changing the project environment or placeholder model name in `configs/config.yaml` should change the generated run summary.

### 4.3 Logging

Production systems need logs because print statements are not enough.

Logs help answer what happened, when it happened, and where the run output went. In this project, logs are written to both the console and a log file under `logs/`.

### 4.4 Reproducibility

A fixed seed helps make runs more predictable.

In future ML lectures, reproducibility will matter more because training can involve randomness. Setting seeds early teaches learners the habit of making runs easier to repeat and debug.

### 4.5 Output artifacts

The project saves a generated result to `outputs/run_summary.txt`.

In future lectures, artifacts may include trained models, evaluation reports, processed data, embeddings, or prediction files.

### 4.6 Testing

The first test only validates project structure. This is intentionally simple.

Later tests will validate real logic such as data validation, model behavior, retrieval quality, and API behavior.

### 4.7 Git readiness

The repo is structured so it can be committed to Git from Lecture 1.

Git will help track how the project evolves lecture by lecture. Learners can look back and see what changed as each new production capability is added.

## 5. Folder-by-folder explanation

| Path               | Purpose                      | Future use                                            |
| ------------------ | ---------------------------- | ----------------------------------------------------- |
| `configs/`         | Stores configuration files   | environment configs, model configs, retrieval configs |
| `data/`            | Stores sample/local data     | raw data, processed data, documents                   |
| `notebooks/`       | Stores exploration notebooks | EDA, experiments                                      |
| `src/`             | Stores reusable code         | training, retrieval, evaluation, API modules          |
| `tests/`           | Stores automated tests       | unit tests, integration tests                         |
| `outputs/`         | Stores generated outputs     | reports, predictions, summaries                       |
| `logs/`            | Stores runtime logs          | execution logs, debugging logs                        |
| `models/`          | Stores local model artifacts | trained model files before registry                   |
| `main.py`          | Main entry point             | orchestrates project run                              |
| `requirements.txt` | Dependency list              | grows as libraries are added                          |

## 6. Demo walkthrough

1. Create virtual environment.
2. Install dependencies.
3. Review `configs/config.yaml`.
4. Run `python main.py`.
5. Open `outputs/run_summary.txt`.
6. Open the log file under `logs/`.
7. Run `pytest`.
8. Commit changes to Git.

## 7. Trainer speaking script

"Today we are not building a chatbot or training a model. We are building the foundation that will allow us to do those things properly later. A weak project structure leads to messy notebooks, hardcoded values, missing logs, and code that cannot be reproduced. This template solves that problem early."

"When we add LLM calls, RAG, MLflow, Docker, CI/CD, and monitoring later, they will not be random additions. They will fit into this same structure."

## 8. Learner questions

1. Why should configuration be outside Python code?
2. Why are logs better than only print statements?
3. Why should notebooks not contain final production logic?
4. What kind of files should not be committed to Git?
5. Why is it useful to have tests even in a simple project?
6. Why are we not using a real LLM API in Lecture 1?

## 9. Quick recap

In Lecture 1, we created a clean, configurable, logged, testable, and Git-ready project foundation. This is the base on which all future MLOps and LLMOps components will be added.

## 10. What comes next

In upcoming lectures, this same repository can be extended with:

- data validation
- document processing
- ML training
- experiment tracking
- LLM API calls
- RAG pipeline
- evaluation
- Docker
- CI/CD
- cloud deployment
- monitoring
