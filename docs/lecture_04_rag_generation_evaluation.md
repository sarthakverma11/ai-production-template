# Lecture 4: Grounded RAG Generation and Evaluation

## 1. Purpose of this lecture

Lecture 4 continues from the Lecture 3 Azure vector retrieval workflow.

Lecture 3 could retrieve top-k evidence chunks from Azure AI Search. Lecture 4 uses those retrieved chunks inside prompt templates, calls an Azure OpenAI chat model, generates grounded answers, evaluates prompt versions, and tracks the evaluation run in MLflow.

The goal is to turn semantic retrieval into a complete classroom RAG loop:

```text
question -> retrieval -> context -> prompt -> grounded answer -> evaluation -> MLflow tracking
```

## 2. What we built

Lecture 4 adds:

- prompt files and a prompt registry
- chat model registry configuration
- grounded RAG answer generation
- a small retriever adapter around Lecture 3 retrieval
- a golden evaluation question dataset
- deterministic evaluation metrics
- LLM-as-judge evaluation metrics
- prompt comparison CSV outputs
- local MLflow experiment tracking
- MLflow GenAI traces for each evaluation question
- tests for the new prompt and evaluation helpers

## 3. What this is not

Lecture 4 does not rebuild the earlier lectures.

It does not:

- recreate the Azure AI Search index
- regenerate embeddings unless Lecture 3 retrieval needs a query embedding
- reprocess documents
- introduce FastAPI, Docker, Airflow, LangGraph, CI/CD, or deployment
- depend on RAGAS or DeepEval for the required classroom demo
- silently use fake retrieval

Fake retrieval is available only when `USE_FAKE_RETRIEVER=true` is set for local wiring tests.

## 4. End-to-end flow

```text
User question
-> src/retrieval/retriever_adapter.py
-> src/search/retrieval_service.py
-> Azure AI Search top-k chunks
-> prompts/qa_prompt_v1.txt or prompts/qa_prompt_v2.txt
-> Azure OpenAI chat completion
-> grounded answer with source metadata
-> deterministic evaluation checks
-> CSV files and MLflow run
```

## 5. Prompt registry

Prompt text lives under:

```text
prompts/
|-- qa_prompt_v1.txt
|-- qa_prompt_v2.txt
`-- prompt_registry.yaml
```

`qa_prompt_v1.txt` is intentionally simple. It asks the model to use the provided context and answer clearly.

`qa_prompt_v2.txt` is stricter. It acts as a company policy assistant, answers only from the provided context, mentions source documents when possible, and uses a fixed refusal when the context is insufficient:

```text
I do not have enough information in the provided documents.
```

`prompt_registry.yaml` tracks:

- active prompt
- prompt version
- file path
- description
- status
- owner
- created date

## 6. Model registry

Lecture 4 adds:

```text
configs/model_registry.yaml
```

This file stores non-secret chat model settings:

- provider
- deployment environment variable
- deployment name placeholder
- temperature
- max tokens
- purpose
- status

Real API keys are not stored in YAML config files.

## 7. Environment variables

Create a local `.env` file from `.env.example`.

Lecture 4 uses these Azure OpenAI chat variables:

```env
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_VERSION=
AZURE_OPENAI_CHAT_DEPLOYMENT=
```

Lecture 3 retrieval still uses:

```env
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_API_KEY=
AZURE_SEARCH_INDEX_NAME=policy-knowledge-index-v1
```

Secrets must stay in `.env`. They should not be placed in code, YAML configs, docs, logs, or Git.

## 8. New code paths

| Path | Purpose |
| ---- | ------- |
| `prompts/qa_prompt_v1.txt` | Simple RAG prompt |
| `prompts/qa_prompt_v2.txt` | Stricter company policy assistant prompt |
| `prompts/prompt_registry.yaml` | Tracks active and available prompt versions |
| `configs/model_registry.yaml` | Tracks chat model settings |
| `src/generation/prompt_loader.py` | Loads and formats registered prompts |
| `src/generation/answer_generator.py` | Runs retrieval, prompting, and chat completion |
| `src/retrieval/retriever_adapter.py` | Wraps the Lecture 3 retrieval import path |
| `evaluations/eval_questions.json` | Golden evaluation questions |
| `src/evaluation/eval_metrics.py` | Deterministic source, keyword, and refusal checks |
| `src/evaluation/llm_judge.py` | LLM-as-judge scoring for advanced RAG metrics |
| `src/evaluation/run_eval_pipeline.py` | Runs prompt comparison and logs MLflow runs |
| `tests/test_lecture_4_rag.py` | Lightweight Lecture 4 unit tests |

## 9. Run Lecture 3 first

Lecture 4 depends on Lecture 3 retrieval.

If the Azure AI Search index has not been created yet, run:

```bash
python -m src.search.indexing_pipeline
```

Then confirm retrieval works:

```bash
python -m src.search.retrieval_service "How much time off do employees receive?"
```

Expected local file:

```text
outputs/retrieval_results.json
```

## 10. Generate one grounded answer

Run:

```bash
python -m src.generation.answer_generator
```

Expected terminal output:

- question
- prompt version
- model deployment
- retrieved source files
- generated grounded answer

To choose a prompt version:

```bash
python -m src.generation.answer_generator "What is required for travel reimbursement?" --prompt-version qa_prompt_v2
```

## 11. Run the Streamlit RAG app

Run:

```bash
streamlit run app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

The app shows the Lecture 4 flow directly:

- required environment variable status
- active prompt version
- chat deployment name
- search index name
- generated grounded answer
- retrieved evidence chunks
- source files
- prompt version
- model deployment
- latency

This is the main live classroom demo after Lecture 4. The app no longer stops at displaying chunks only.

## 12. Run evaluation

Run:

```bash
python -m src.evaluation.run_eval_pipeline
```

This run calls Azure OpenAI for answer generation and again for LLM-as-judge scoring. It may take a minute or two depending on network latency.

For a faster smoke test, skip the judge model:

```bash
python -m src.evaluation.run_eval_pipeline --skip-llm-judge
```

To test only one prompt version:

```bash
python -m src.evaluation.run_eval_pipeline --prompt-version qa_prompt_v2
```

The pipeline evaluates both:

- `qa_prompt_v1`
- `qa_prompt_v2`

It checks:

- expected source retrieval
- expected answer keywords
- refusal behavior
- latency
- faithfulness
- answer relevance
- context relevance
- context precision
- context recall
- hallucination score
- LLM-as-judge score

Expected generated files:

```text
evaluations/results/results_qa_prompt_v1.csv
evaluations/results/results_qa_prompt_v2.csv
evaluations/results/comparison_summary.csv
```

## 13. MLflow tracking

Lecture 4 uses local MLflow by default.

The evaluation pipeline creates an experiment named:

```text
Lecture_4_RAG_Evaluation
```

For each prompt version, it logs:

- prompt version
- model deployment
- temperature
- top-k
- eval dataset path
- retriever index version
- pass-rate metrics
- average latency
- average LLM-as-judge metrics
- prompt, dataset, result CSV, and comparison CSV artifacts

The evaluation pipeline also logs MLflow GenAI traces. Each evaluation question creates one trace with spans for:

- evaluation case
- RAG answer-generation chain
- model config loading
- retrieval
- context construction
- prompt loading
- Azure OpenAI chat completion

Use the classic experiment run table to compare prompt-level metrics. Use the GenAI traces page to inspect what happened inside one question.

Open the UI:

```bash
mlflow ui
```

Then open the local URL shown in the terminal.

In the MLflow UI:

- Use **Model training** to inspect runs, parameters, metrics, and artifacts.
- Use **GenAI** -> **Traces** to inspect per-question RAG traces.

## 14. Evaluation dataset

The golden dataset lives at:

```text
evaluations/eval_questions.json
```

Each item contains:

- `id`
- `question`
- `expected_source`
- `expected_keywords`
- `expected_facts`
- `should_refuse`

The current dataset covers leave, travel reimbursement, IT support, password reset, and one unsupported question.

For teaching, explain the dataset like this:

```text
eval_questions.json = exam paper
evaluation pipeline = examiner
MLflow = report card
```

The dataset is not meant to contain every possible chatbot question. It is a fixed set of representative questions that lets learners compare prompt versions fairly.

## 15. Evaluation metrics

Lecture 4 intentionally starts with simple checks:

| Function | Purpose |
| -------- | ------- |
| `check_expected_source` | Confirms the expected source file was retrieved |
| `check_expected_keywords` | Confirms expected keywords appear in the answer |
| `check_refusal_behavior` | Confirms unsupported questions are refused |
| `calculate_pass_fail` | Combines checks into one row-level result |
| `calculate_context_precision` | Checks whether expected sources appear early in retrieval |
| `calculate_context_recall` | Checks whether retrieved context contains expected clues |
| `calculate_context_relevance` | Gives a simple lexical relevance score |

These metrics are easy to explain and do not require another model call.

Lecture 4 then adds LLM-as-judge scoring through `src/evaluation/llm_judge.py`.

The judge model receives:

- question
- retrieved context
- generated answer
- expected source
- expected keywords
- expected refusal behavior

It returns scores from `0.0` to `1.0`:

| Metric | Meaning |
| ------ | ------- |
| `faithfulness_score` | Whether the answer is supported by the retrieved context |
| `answer_relevance_score` | Whether the answer directly answers the question |
| `llm_context_relevance_score` | Whether the retrieved context is useful for the question |
| `context_precision_score` | Whether expected sources appear early in retrieved chunks |
| `context_recall_score` | Whether retrieved context contains expected clues |
| `hallucination_score` | Whether the answer contains unsupported claims; lower is better |
| `llm_judge_score` | Overall judge quality score; higher is better |

For this classroom demo, the same `gpt-4.1-mini` deployment can generate answers and judge answers. In production, teams may use a separate evaluator model.

Frameworks such as RAGAS, DeepEval, Evidently AI, Promptfoo, or Azure AI evaluation tools can be introduced later. Lecture 4 first shows the evaluation logic directly so learners understand what the tools are measuring.

## 16. Demo steps

1. Show Lecture 3 retrieval output.
2. Open `prompts/qa_prompt_v1.txt`.
3. Open `prompts/qa_prompt_v2.txt`.
4. Show `prompts/prompt_registry.yaml`.
5. Show `configs/model_registry.yaml`.
6. Run `streamlit run app/streamlit_app.py`.
7. Ask a policy question in the app.
8. Point out the final answer, retrieved sources, prompt version, model deployment, and latency.
9. Open `evaluations/eval_questions.json`.
10. Run `python -m src.evaluation.run_eval_pipeline`.
11. Open the result CSV files and show deterministic plus judge-score columns.
12. Start `mlflow ui`.
13. Compare prompt versions in the MLflow experiment.
14. Open **GenAI** -> **Traces** and inspect one failed or successful question.

## 17. Troubleshooting

| Problem | Check |
| ------- | ----- |
| Missing chat deployment | Set `AZURE_OPENAI_CHAT_DEPLOYMENT` |
| Missing API version | Set `AZURE_OPENAI_API_VERSION` |
| Retrieval fails | Confirm Lecture 3 retrieval works first |
| Search index has no records | Run the Lecture 3 indexing pipeline |
| Prompt file not found | Check `prompts/prompt_registry.yaml` file paths |
| Unknown prompt version | Use `qa_prompt_v1` or `qa_prompt_v2` |
| MLflow command not found | Run `pip install -r requirements.txt` |
| Result CSV shows errors | Read the `error` column for the failed question |
| Judge scores are zero | Read the `judge_error` column |
| Evaluation appears stuck | Watch the per-question progress output; set `AZURE_OPENAI_TIMEOUT_SECONDS=30` |
| Fake retriever does not generate answers | Fake retrieval only replaces search; Azure OpenAI chat is still required |

## 18. Git versus generated artifacts

Git tracks:

- prompt files
- prompt registry
- model registry
- source code
- tests
- evaluation dataset
- docs

Git should not track:

- `.env`
- `mlruns/`
- generated evaluation result CSV files
- generated logs
- generated outputs

## 19. Quick recap

Lecture 4 turns Lecture 3 retrieval into grounded RAG answer generation.

The project now retrieves evidence chunks, injects them into versioned prompts, calls an Azure OpenAI chat model, evaluates prompt behavior with a golden dataset, scores answers with deterministic checks and an LLM judge, writes comparison CSV files, and tracks metrics, artifacts, and traces in local MLflow.

## 20. What comes next

Future lectures can extend this same repository with:

- improved citation formatting
- richer retrieval and answer-quality evaluation
- orchestration
- deployment
- CI/CD
- monitoring
- feedback loops
