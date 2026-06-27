"""Run Lecture 4 RAG evaluation and track results in local MLflow."""

import json
import os
import argparse
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from src.config_loader import load_config
from src.evaluation.eval_metrics import (
    calculate_context_precision,
    calculate_context_recall,
    calculate_context_relevance,
    calculate_pass_fail,
)
from src.evaluation.llm_judge import ADVANCED_SCORE_KEYS, evaluate_with_llm_judge
from src.generation.answer_generator import build_context_block, generate_answer, load_model_config
from src.generation.prompt_loader import load_prompt
from src.utils import ensure_dir


EVAL_DATASET_PATH = Path("evaluations/eval_questions.json")
RESULTS_DIR = Path("evaluations/results")
PROMPT_VERSIONS = ["qa_prompt_v1", "qa_prompt_v2"]
EXPERIMENT_NAME = "Lecture_4_RAG_Evaluation"


def run_eval_pipeline(
    eval_dataset_path: str | Path = EVAL_DATASET_PATH,
    prompt_versions: list[str] | None = None,
    top_k: int = 3,
    enable_llm_judge: bool = True,
) -> dict[str, Any]:
    """Evaluate configured prompt versions and write CSV plus MLflow outputs."""
    eval_path = Path(eval_dataset_path)
    questions = load_eval_questions(eval_path)
    ensure_dir(str(RESULTS_DIR))
    mlflow.set_experiment(EXPERIMENT_NAME)

    selected_prompt_versions = prompt_versions or PROMPT_VERSIONS
    model_config = load_model_config()
    deployment_name = configured_model_deployment(model_config)
    retriever_index_version = configured_retriever_index_version()
    all_summaries: list[dict[str, Any]] = []
    run_ids: list[str] = []

    for prompt_version in selected_prompt_versions:
        rows = []
        result_path = RESULTS_DIR / f"results_{prompt_version}.csv"
        _prompt_text, prompt_metadata = load_prompt(prompt_version)

        with mlflow.start_run(run_name=f"lecture_4_{prompt_version}") as run:
            run_ids.append(run.info.run_id)
            mlflow.log_params(
                {
                    "prompt_version": prompt_version,
                    "model_deployment": deployment_name,
                    "temperature": model_config.get("temperature"),
                    "top_k": top_k,
                    "eval_dataset": str(eval_path),
                    "retriever_index_version": retriever_index_version,
                    "llm_judge_enabled": enable_llm_judge,
                }
            )

            for question_index, item in enumerate(questions, start=1):
                print(
                    f"[{prompt_version}] Evaluating {question_index}/{len(questions)}: "
                    f"{item.get('id', 'unknown')}",
                    flush=True,
                )
                row = evaluate_one_question(
                    item,
                    prompt_version,
                    top_k,
                    enable_tracing=True,
                    enable_llm_judge=enable_llm_judge,
                )
                rows.append(row)

            result_frame = pd.DataFrame(rows)
            result_frame.to_csv(result_path, index=False)
            summary = summarize_results(prompt_version, result_frame)
            all_summaries.append(summary)

            mlflow.log_metrics(
                {
                    "total_questions": summary["total_questions"],
                    "retrieval_source_pass_rate": summary["retrieval_source_pass_rate"],
                    "keyword_pass_rate": summary["keyword_pass_rate"],
                    "refusal_pass_rate": summary["refusal_pass_rate"],
                    "overall_pass_rate": summary["overall_pass_rate"],
                    "average_latency_ms": summary["average_latency_ms"],
                    "average_faithfulness_score": summary["average_faithfulness_score"],
                    "average_answer_relevance_score": summary["average_answer_relevance_score"],
                    "average_context_relevance_score": summary["average_context_relevance_score"],
                    "average_context_precision_score": summary["average_context_precision_score"],
                    "average_context_recall_score": summary["average_context_recall_score"],
                    "average_hallucination_score": summary["average_hallucination_score"],
                    "average_llm_judge_score": summary["average_llm_judge_score"],
                }
            )
            mlflow.log_artifact(prompt_metadata["file_path"])
            mlflow.log_artifact(str(eval_path))
            mlflow.log_artifact(str(result_path))

    summary_path = RESULTS_DIR / "comparison_summary.csv"
    summary_frame = pd.DataFrame(all_summaries)
    summary_frame.to_csv(summary_path, index=False)
    log_summary_to_runs(run_ids, summary_path)

    return {
        "result_files": [str(RESULTS_DIR / f"results_{version}.csv") for version in selected_prompt_versions],
        "comparison_summary": str(summary_path),
    }


def load_eval_questions(eval_dataset_path: Path) -> list[dict[str, Any]]:
    """Load golden evaluation questions from JSON."""
    if not eval_dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {eval_dataset_path}")

    questions = json.loads(eval_dataset_path.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError("Evaluation dataset must be a JSON list.")
    return questions


def evaluate_one_question(
    item: dict[str, Any],
    prompt_version: str,
    top_k: int,
    enable_tracing: bool = False,
    enable_llm_judge: bool = True,
) -> dict[str, Any]:
    """Run one eval case and return a CSV-ready row."""
    answer = ""
    source_files: list[str] = []
    chunk_ids: list[str] = []
    retrieved_chunks: list[dict[str, Any]] = []
    retrieved_context = ""
    model_deployment = ""
    latency_ms = 0.0
    error = ""
    judge_error = ""
    advanced_scores = {key: 0.0 for key in ADVANCED_SCORE_KEYS}
    advanced_scores["judge_reason"] = ""
    advanced_scores["judge_model_deployment"] = ""

    trace_context = _optional_eval_trace(
        enable_tracing=enable_tracing,
        item=item,
        prompt_version=prompt_version,
        top_k=top_k,
    )
    with trace_context as trace_span:
        try:
            result = generate_answer(
                item["question"],
                prompt_version=prompt_version,
                top_k=top_k,
                enable_tracing=enable_tracing,
            )
            answer = result["answer"]
            source_files = result["source_files"]
            chunk_ids = result["retrieved_chunk_ids"]
            retrieved_chunks = result.get("retrieved_chunks", [])
            retrieved_context = build_context_block(retrieved_chunks)
            model_deployment = result["model_deployment"]
            latency_ms = result["latency_ms"]
        except Exception as exc:
            error = str(exc)

        checks = calculate_pass_fail(
            retrieved_sources=source_files,
            expected_source=item.get("expected_source", ""),
            answer=answer,
            expected_keywords=item.get("expected_keywords", []),
            should_refuse=bool(item.get("should_refuse", False)),
        )
        context_precision_score = calculate_context_precision(
            retrieved_sources=source_files,
            expected_source=item.get("expected_source", ""),
        )
        context_recall_score = calculate_context_recall(
            retrieved_context=retrieved_context,
            expected_keywords=item.get("expected_keywords", []),
            should_refuse=bool(item.get("should_refuse", False)),
        )
        deterministic_context_relevance_score = calculate_context_relevance(
            question=item.get("question", ""),
            retrieved_context=retrieved_context,
        )
        if answer and not error and enable_llm_judge:
            try:
                advanced_scores = evaluate_with_llm_judge(
                    question=item["question"],
                    answer=answer,
                    retrieved_context=retrieved_context,
                    expected_source=item.get("expected_source", ""),
                    expected_keywords=item.get("expected_keywords", []),
                    should_refuse=bool(item.get("should_refuse", False)),
                )
                judge_error = advanced_scores.get("judge_error", "")
            except Exception as exc:
                judge_error = str(exc)
        elif not enable_llm_judge:
            judge_error = "LLM judge was skipped for this run."

        if trace_span is not None:
            trace_span.set_outputs(
                {
                    "answer_preview": answer[:500],
                    "retrieved_sources": source_files,
                    "retrieved_chunk_ids": chunk_ids,
                    "model_deployment": model_deployment,
                    "latency_ms": latency_ms,
                    "source_pass": checks["source_pass"],
                    "keyword_pass": checks["keyword_pass"],
                    "refusal_pass": checks["refusal_pass"],
                    "overall_pass": checks["overall_pass"],
                    "faithfulness_score": advanced_scores["faithfulness_score"],
                    "answer_relevance_score": advanced_scores["answer_relevance_score"],
                    "context_relevance_score": advanced_scores["llm_context_relevance_score"],
                    "context_precision_score": context_precision_score,
                    "context_recall_score": context_recall_score,
                    "hallucination_score": advanced_scores["hallucination_score"],
                    "llm_judge_score": advanced_scores["llm_judge_score"],
                    "error": error,
                    "judge_error": judge_error,
                }
            )

    return {
        "id": item.get("id"),
        "prompt_version": prompt_version,
        "question": item.get("question"),
        "answer": answer,
        "expected_source": item.get("expected_source", ""),
        "retrieved_sources": ";".join(source_files),
        "retrieved_chunk_ids": ";".join(chunk_ids),
        "expected_keywords": ";".join(item.get("expected_keywords", [])),
        "should_refuse": bool(item.get("should_refuse", False)),
        "source_pass": checks["source_pass"],
        "keyword_pass": checks["keyword_pass"],
        "refusal_pass": checks["refusal_pass"],
        "overall_pass": checks["overall_pass"],
        "faithfulness_score": advanced_scores["faithfulness_score"],
        "answer_relevance_score": advanced_scores["answer_relevance_score"],
        "llm_context_relevance_score": advanced_scores["llm_context_relevance_score"],
        "deterministic_context_relevance_score": deterministic_context_relevance_score,
        "context_precision_score": context_precision_score,
        "context_recall_score": context_recall_score,
        "hallucination_score": advanced_scores["hallucination_score"],
        "llm_judge_score": advanced_scores["llm_judge_score"],
        "judge_model_deployment": advanced_scores.get("judge_model_deployment", ""),
        "judge_reason": advanced_scores.get("judge_reason", ""),
        "judge_error": judge_error,
        "latency_ms": latency_ms,
        "model_deployment": model_deployment,
        "error": error,
    }


def summarize_results(prompt_version: str, result_frame: pd.DataFrame) -> dict[str, Any]:
    """Calculate pass rates for one prompt version."""
    total_questions = len(result_frame)
    if total_questions == 0:
        raise ValueError("Cannot summarize an empty evaluation result set.")

    return {
        "prompt_version": prompt_version,
        "total_questions": total_questions,
        "retrieval_source_pass_rate": float(result_frame["source_pass"].mean()),
        "keyword_pass_rate": float(result_frame["keyword_pass"].mean()),
        "refusal_pass_rate": float(result_frame["refusal_pass"].mean()),
        "overall_pass_rate": float(result_frame["overall_pass"].mean()),
        "average_latency_ms": float(result_frame["latency_ms"].mean()),
        "average_faithfulness_score": float(result_frame["faithfulness_score"].mean()),
        "average_answer_relevance_score": float(result_frame["answer_relevance_score"].mean()),
        "average_context_relevance_score": float(result_frame["llm_context_relevance_score"].mean()),
        "average_context_precision_score": float(result_frame["context_precision_score"].mean()),
        "average_context_recall_score": float(result_frame["context_recall_score"].mean()),
        "average_hallucination_score": float(result_frame["hallucination_score"].mean()),
        "average_llm_judge_score": float(result_frame["llm_judge_score"].mean()),
    }


def configured_model_deployment(model_config: dict[str, Any]) -> str:
    """Return the configured chat deployment without forcing Azure credentials to exist."""
    deployment_env_var = model_config.get("deployment_env_var", "AZURE_OPENAI_CHAT_DEPLOYMENT")
    return os.getenv(deployment_env_var) or str(model_config.get("deployment_name", "unknown"))


def configured_retriever_index_version() -> str:
    """Return the Lecture 3 index version from config.yaml when available."""
    try:
        config = load_config()
    except Exception:
        return "unknown"

    search_config = config.get("search", {})
    if not isinstance(search_config, dict):
        return "unknown"

    return str(search_config.get("index_version", "unknown"))


def log_summary_to_runs(run_ids: list[str], summary_path: Path) -> None:
    """Attach the final comparison CSV to each prompt-version MLflow run."""
    client = mlflow.tracking.MlflowClient()
    for run_id in run_ids:
        client.log_artifact(run_id, str(summary_path))


def _optional_eval_trace(
    enable_tracing: bool,
    item: dict[str, Any],
    prompt_version: str,
    top_k: int,
) -> Any:
    """Create an MLflow GenAI trace span for one evaluation case."""
    if not enable_tracing:
        from contextlib import nullcontext

        return nullcontext(None)

    return mlflow.start_span(
        name=f"rag_eval_case_{item.get('id', 'unknown')}_{prompt_version}",
        span_type="EVAL",
        attributes={
            "eval_id": item.get("id"),
            "prompt_version": prompt_version,
            "top_k": top_k,
            "expected_source": item.get("expected_source", ""),
            "expected_keywords": item.get("expected_keywords", []),
            "should_refuse": bool(item.get("should_refuse", False)),
        },
    )


def main() -> None:
    """CLI entry point for the full evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Run Lecture 4 RAG evaluation.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--prompt-version",
        action="append",
        dest="prompt_versions",
        help="Prompt version to evaluate. Use multiple times to compare prompts.",
    )
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Skip LLM-as-judge scoring for a faster smoke test.",
    )
    args = parser.parse_args()

    outputs = run_eval_pipeline(
        prompt_versions=args.prompt_versions,
        top_k=args.top_k,
        enable_llm_judge=not args.skip_llm_judge,
    )
    print("Evaluation complete.")
    print(f"Comparison summary: {outputs['comparison_summary']}")
    for result_file in outputs["result_files"]:
        print(f"Result file: {result_file}")


if __name__ == "__main__":
    main()
