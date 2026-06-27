# Lecture 5: GitHub Actions RAG Quality Gate

## 1. Purpose of this lecture

Lecture 5 turns the Lecture 4 evaluation pipeline into an automated GitHub Actions workflow.

The story is:

```text
Lecture 3 retrieved top-k chunks from Azure AI Search.
Lecture 4 generated grounded answers, evaluated them, and tracked results in MLflow.
Lecture 5 automates the refresh and evaluation steps whenever code is pushed.
```

If the RAG quality score passes, GitHub Actions is green. If quality drops below the threshold, GitHub Actions is red.

## 2. What we built

Lecture 5 adds:

- a push-triggered GitHub Actions workflow
- a helper script to upload refreshed knowledge artifacts to Azure Blob Storage
- a helper script to check evaluation quality
- tests for the upload helper and quality gate
- CI artifact upload for logs, outputs, and evaluation CSV files

This lecture does not add Airflow, Docker, FastAPI, frontend code, or a rebuilt RAG application.

## 3. End-to-end flow

```text
push to master
-> install Python dependencies
-> run pytest
-> rebuild Lecture 2 chunks and lineage
-> upload chunks and lineage to Azure Blob Storage
-> refresh Azure AI Search index
-> run Lecture 4 evaluation for qa_prompt_v2
-> fail or pass the workflow based on overall_pass_rate
```

The workflow file lives at:

```text
.github/workflows/rag-quality-gate.yml
```

## 4. Required GitHub secrets

Add these secrets in the GitHub repository settings before running the workflow:

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

Do not put these values in code, YAML config files, docs, or committed logs.

## 5. Workflow commands

The workflow runs the same beginner-friendly commands that can also be run locally:

```bash
pytest
python -m src.processing.build_knowledge_artifacts
python -m src.storage.upload_knowledge_artifacts
python -m src.search.indexing_pipeline
python -m src.evaluation.run_eval_pipeline --prompt-version qa_prompt_v2 --skip-llm-judge
python -m src.evaluation.check_quality_gate
```

Lecture 5 evaluates only `qa_prompt_v2`, which is treated as the production prompt.

The first CI version skips LLM-as-judge scoring to reduce runtime, cost, and live-service variability.

## 6. Quality gate

The quality gate reads:

```text
evaluations/results/comparison_summary.csv
```

It checks:

```text
prompt_version = qa_prompt_v2
overall_pass_rate >= 0.80
```

With the current five-question golden dataset, this means at least four out of five evaluation cases must pass.

If the score is below `0.80`, `src.evaluation.check_quality_gate` exits with status code `1`, and GitHub Actions marks the workflow red.

## 7. Demo steps

1. Open `.github/workflows/rag-quality-gate.yml`.
2. Show the workflow trigger: push to `master` plus manual rerun.
3. Show the GitHub secrets list.
4. Push a small safe change to `master`.
5. Open the GitHub Actions run.
6. Point out each step: tests, artifact build, blob upload, indexing, evaluation, quality gate.
7. Open the uploaded workflow artifact and inspect `comparison_summary.csv`.
8. Explain that the workflow result is now a quality signal, not only a code signal.

## 8. Troubleshooting

| Problem | Check |
| ------- | ----- |
| Workflow cannot connect to Azure | Confirm all GitHub secrets are set |
| Upload step fails | Confirm Blob containers exist and the storage connection string is valid |
| Indexing step fails | Confirm Search endpoint, key, and index name are correct |
| Evaluation step fails | Confirm Azure OpenAI chat and embedding deployments are available |
| Quality gate fails | Open `comparison_summary.csv` and inspect `overall_pass_rate` |
| Results are hard to inspect | Download the `lecture-5-evaluation-results` workflow artifact |

## 9. Quick recap

Lecture 5 makes the RAG pipeline operational:

```text
knowledge refresh + search indexing + RAG evaluation + pass/fail quality gate
```

This is the first step from a local RAG demo toward production AI automation.
