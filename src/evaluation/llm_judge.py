"""LLM-as-judge scoring for Lecture 4 RAG evaluation."""

import json
import re
from typing import Any

from src.generation.answer_generator import (
    AnswerGenerationError,
    _create_chat_completion,
    _openai_timeout_seconds,
    load_model_config,
    resolve_chat_deployment,
)
from src.utils import load_project_env


ADVANCED_SCORE_KEYS = [
    "faithfulness_score",
    "answer_relevance_score",
    "llm_context_relevance_score",
    "hallucination_score",
    "llm_judge_score",
]


def evaluate_with_llm_judge(
    question: str,
    answer: str,
    retrieved_context: str,
    expected_source: str,
    expected_keywords: list[str],
    should_refuse: bool,
) -> dict[str, Any]:
    """Use the configured Azure OpenAI chat deployment to score one RAG answer."""
    load_project_env()
    if not answer:
        return _empty_scores("No answer was generated.")

    model_config = load_model_config()
    deployment = resolve_chat_deployment(model_config)
    prompt = build_judge_prompt(
        question=question,
        answer=answer,
        retrieved_context=retrieved_context,
        expected_source=expected_source,
        expected_keywords=expected_keywords,
        should_refuse=should_refuse,
    )

    try:
        from openai import AzureOpenAI
    except ModuleNotFoundError as error:
        raise AnswerGenerationError("openai is required. Install dependencies first.") from error

    import os

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    if not endpoint:
        raise AnswerGenerationError("Missing AZURE_OPENAI_ENDPOINT.")
    if not api_key:
        raise AnswerGenerationError("Missing AZURE_OPENAI_API_KEY.")
    if not api_version:
        raise AnswerGenerationError("Missing AZURE_OPENAI_API_VERSION.")

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        timeout=_openai_timeout_seconds(),
        max_retries=1,
    )
    response = _create_chat_completion(
        client=client,
        deployment=deployment,
        prompt=prompt,
        temperature=0.0,
        max_tokens=500,
    )
    judge_text = response.choices[0].message.content or ""
    parsed = parse_judge_response(judge_text)
    parsed["judge_model_deployment"] = deployment
    return parsed


def build_judge_prompt(
    question: str,
    answer: str,
    retrieved_context: str,
    expected_source: str,
    expected_keywords: list[str],
    should_refuse: bool,
) -> str:
    """Build a compact rubric prompt for the judge model."""
    return f"""
You are an evaluator for a Retrieval-Augmented Generation system.

Score the generated answer using the rubric below. Use values from 0.0 to 1.0.

Metric definitions:
- faithfulness_score: 1.0 means every important claim in the answer is supported by the retrieved context.
- answer_relevance_score: 1.0 means the answer directly addresses the user question.
- llm_context_relevance_score: 1.0 means the retrieved context is useful for answering the question.
- hallucination_score: 0.0 means no unsupported claims; 1.0 means many unsupported claims.
- llm_judge_score: overall quality score, where 1.0 is best.

Expected source:
{expected_source or "No specific source expected"}

Expected keywords:
{expected_keywords}

Should refuse:
{should_refuse}

Question:
{question}

Retrieved context:
{retrieved_context[:6000]}

Generated answer:
{answer}

Return only valid JSON with this exact shape:
{{
  "faithfulness_score": 0.0,
  "answer_relevance_score": 0.0,
  "llm_context_relevance_score": 0.0,
  "hallucination_score": 0.0,
  "llm_judge_score": 0.0,
  "judge_reason": "short reason"
}}
""".strip()


def parse_judge_response(judge_text: str) -> dict[str, Any]:
    """Parse and normalize a JSON response from the judge model."""
    json_text = _extract_json_object(judge_text)
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return _empty_scores(f"Judge returned non-JSON output: {judge_text[:200]}")

    normalized = {}
    for key in ADVANCED_SCORE_KEYS:
        normalized[key] = _clamp_score(parsed.get(key, 0.0))

    normalized["judge_reason"] = str(parsed.get("judge_reason", "")).strip()
    normalized["judge_error"] = ""
    return normalized


def _extract_json_object(text: str) -> str:
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced_match:
        return fenced_match.group(1)

    object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if object_match:
        return object_match.group(0)

    return text.strip()


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0

    return round(max(0.0, min(1.0, score)), 4)


def _empty_scores(reason: str) -> dict[str, Any]:
    scores = {key: 0.0 for key in ADVANCED_SCORE_KEYS}
    scores["judge_reason"] = reason
    scores["judge_error"] = reason
    scores["judge_model_deployment"] = ""
    return scores
