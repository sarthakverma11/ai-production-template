# AI Production Template

This repository is a progressive MLOps/LLMOps learning project. Each lecture extends the same production-style AI project instead of starting from scratch.

## Current status

```text
Lecture 1 complete: Production AI Project Foundation
Lecture 2 complete: Document Processing, Data Versioning, and Chatbot Shell
Lecture 3 complete: Azure Blob, Embeddings, and Vector Retrieval
Lecture 4 complete: Grounded RAG Generation, Evaluation, and MLflow Tracking
Lecture 5 complete: GitHub Actions Knowledge Refresh and RAG Quality Gate
```

## Learning progression

| Lecture | Main focus | Project additions |
| ------- | ---------- | ----------------- |
| Lecture 1 | Production AI foundations | Config, logging, seed, artifacts, tests, Git-ready structure |
| Lecture 2 | Document data operations | Raw documents, validation, cleaning, chunking, lineage, DVC preparation, Streamlit shell |
| Lecture 3 | Azure vector retrieval | Blob artifact loading, Azure OpenAI embeddings, Azure AI Search vector index, top-k retrieval display |
| Lecture 4 | Grounded RAG evaluation | Prompt versions, answer generation, golden dataset checks, comparison CSV files, MLflow tracking |
| Lecture 5 | RAG pipeline automation | GitHub Actions workflow, knowledge refresh, Azure indexing, evaluation gate, CI artifacts |

## Lecture boundary

The Lecture 3 application is not powered by chat completion.

It retrieves evidence chunks with Azure OpenAI embeddings and Azure AI Search.

Lecture 4 adds final answer generation separately through `src/generation/answer_generator.py`, so learners can see the clear jump from retrieval-only evidence to grounded RAG answers.

## Project structure

```text
ai-production-template/
|-- app/
|   `-- streamlit_app.py
|-- configs/
|   |-- config.yaml
|   `-- model_registry.yaml
|-- data/
|   |-- raw/
|   |   `-- policies_v1/
|   |-- processed/
|   `-- metadata/
|-- docs/
|   |-- lecture_01_foundation.md
|   |-- lecture_02_document_processing.md
|   |-- lecture_03_azure_vector_search.md
|   |-- lecture_04_rag_generation_evaluation.md
|   `-- lecture_05_github_actions_pipeline.md
|-- evaluations/
|   `-- eval_questions.json
|-- notebooks/
|-- prompts/
|   |-- qa_prompt_v1.txt
|   |-- qa_prompt_v2.txt
|   `-- prompt_registry.yaml
|-- src/
|   |-- config_loader.py
|   |-- logger.py
|   |-- utils.py
|   |-- chat_service.py
|   |-- ingestion/
|   |-- processing/
|   |-- storage/
|   |-- embeddings/
|   |-- search/
|   |-- retrieval/
|   |-- generation/
|   `-- evaluation/
|-- tests/
|-- outputs/
|-- logs/
|-- models/
|-- main.py
|-- pytest.ini
|-- requirements.txt
|-- README.md
|-- .env.example
|-- .gitignore
`-- .dvcignore
```

## Setup

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

Create a local `.env` from the example file and fill only local secrets:

```bash
copy .env.example .env
```

Do not commit `.env`.

## Run Lecture 1 baseline

```bash
python main.py
```

Expected generated files:

```text
outputs/run_summary.txt
logs/lecture_1_demo.log
```

## Run Lecture 2 document pipeline

```bash
python -m src.processing.build_knowledge_artifacts
```

Expected generated files:

```text
data/processed/chunks_v1.json
data/metadata/lineage_v1.json
logs/lecture_2_document_pipeline.log
```

## Run Lecture 3 Azure indexing pipeline

Before this step, upload the Lecture 2 artifacts to Azure Blob Storage:

```text
processed-artifacts/chunks_v1.json
metadata/lineage_v1.json
```

Then run:

```bash
python -m src.search.indexing_pipeline
```

Expected generated file:

```text
outputs/indexing_summary_v1.json
```

This command creates or reuses the Azure AI Search index named by `AZURE_SEARCH_INDEX_NAME` and uploads embedded chunk records.

## Run Lecture 3 retrieval from CLI

```bash
python -m src.search.retrieval_service "How much time off do employees receive?"
```

Expected generated file:

```text
outputs/retrieval_results.json
```

## Run Lecture 4 Streamlit RAG app

```bash
streamlit run app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

Ask questions about leave, travel, reimbursement, laptop, password, or helpdesk topics. The app retrieves evidence chunks, sends them through the active prompt version, calls the Azure OpenAI chat deployment, and displays the grounded answer with source metadata.

## Run Lecture 4 grounded RAG answer generation

Use this CLI command when you want to test the Lecture 4 backend without opening Streamlit.

```bash
python -m src.generation.answer_generator
```

Expected terminal output:

```text
Question
Prompt version
Model deployment
Retrieved sources
Generated grounded answer
```

## Run Lecture 4 evaluation pipeline

```bash
python -m src.evaluation.run_eval_pipeline
```

For a faster smoke test without LLM-as-judge scoring:

```bash
python -m src.evaluation.run_eval_pipeline --skip-llm-judge
```

Expected generated files:

```text
evaluations/results/results_qa_prompt_v1.csv
evaluations/results/results_qa_prompt_v2.csv
evaluations/results/comparison_summary.csv
```

The evaluation pipeline runs deterministic checks plus LLM-as-judge scoring. It also creates a local MLflow experiment:

```text
Lecture_4_RAG_Evaluation
```

Open MLflow:

```bash
mlflow ui
```

In MLflow, use **Model training** for prompt-version metrics and artifacts. Use **GenAI** -> **Traces** to inspect each evaluation question step by step.

Logged metrics include source pass rate, keyword pass rate, refusal pass rate, overall pass rate, latency, faithfulness, answer relevance, context relevance, context precision, context recall, hallucination score, and LLM judge score.

## Run tests

```bash
pytest
```

## Lecture documentation

Detailed lecture notes are available in:

- [Lecture 1: Production AI Project Foundation](docs/lecture_01_foundation.md)
- [Lecture 2: Document Processing, Data Versioning, and Chatbot Shell](docs/lecture_02_document_processing.md)
- [Lecture 3: Azure Vector Search Retrieval](docs/lecture_03_azure_vector_search.md)
- [Lecture 4: Grounded RAG Generation and Evaluation](docs/lecture_04_rag_generation_evaluation.md)
- [Lecture 5: GitHub Actions RAG Quality Gate](docs/lecture_05_github_actions_pipeline.md)

## Git workflow

```bash
git status
git add .github app configs docs evaluations prompts src tests .env.example .dvcignore .gitignore pytest.ini requirements.txt README.md
git commit -m "Add Lecture 5 GitHub Actions RAG quality gate"
```

DVC-tracked raw data should not be committed directly to GitHub after `dvc add`. Commit the `.dvc` pointer file generated by DVC instead.

## Summary

Lecture 1 built the production engineering foundation.

Lecture 2 added governed, validated, versioned document data and a visible application shell.

Lecture 3 added Azure-backed semantic retrieval while still avoiding final answer generation.

Lecture 4 added grounded answer generation, prompt versioning, deterministic evaluation, comparison CSV files, and local MLflow tracking.

## Lecture 5 addendum: GitHub Actions RAG quality gate

Lecture 5 is appended after the Lecture 4 RAG evaluation work. It does not rebuild the app or introduce Airflow, Docker, FastAPI, or a frontend.

The new automation story is:

```text
push to GitHub
-> install dependencies
-> run tests
-> rebuild Lecture 2 knowledge artifacts
-> upload chunks and lineage to Azure Blob Storage
-> refresh Azure AI Search index
-> run Lecture 4 evaluation for qa_prompt_v2
-> pass or fail the workflow based on quality
```

Run the same flow locally after configuring `.env`:

```bash
python -m src.processing.build_knowledge_artifacts
python -m src.storage.upload_knowledge_artifacts
python -m src.search.indexing_pipeline
python -m src.evaluation.run_eval_pipeline --prompt-version qa_prompt_v2 --skip-llm-judge
python -m src.evaluation.check_quality_gate
```

The quality gate reads:

```text
evaluations/results/comparison_summary.csv
```

It passes when `qa_prompt_v2` has:

```text
overall_pass_rate >= 0.80
```

The GitHub Actions workflow lives at:

```text
.github/workflows/rag-quality-gate.yml
```

It runs on pushes to `main` and can also be started manually from the GitHub Actions tab.

Add these repository secrets before running the workflow:

```text
AZURE_STORAGE_CONNECTION_STRING
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_API_VERSION
AZURE_OPENAI_EMBEDDING_DEPLOYMENT
AZURE_OPENAI_CHAT_DEPLOYMENT
AZURE_SEARCH_ENDPOINT
AZURE_SEARCH_API_KEY
AZURE_SEARCH_INDEX_NAME
```

Lecture 5 added GitHub Actions automation for knowledge refresh, Azure Search indexing, RAG evaluation, and a pass/fail quality gate.

Future lectures will add deployment, monitoring, and feedback loops.
