from src.evaluation.eval_metrics import (
    REFUSAL_MESSAGE,
    calculate_context_precision,
    calculate_context_recall,
    calculate_context_relevance,
    calculate_pass_fail,
)
from src.evaluation.llm_judge import parse_judge_response
from src.generation.answer_generator import build_context_block
from src.generation.prompt_loader import format_prompt, load_prompt


def test_active_prompt_can_be_loaded_and_formatted():
    prompt_text, metadata = load_prompt()
    formatted_prompt = format_prompt(
        prompt_text,
        context="Employees receive 18 days of annual leave.",
        question="How much leave do employees receive?",
    )

    assert metadata["prompt_version"] == "qa_prompt_v2"
    assert "Employees receive 18 days" in formatted_prompt
    assert "How much leave" in formatted_prompt


def test_context_block_includes_source_metadata():
    context = build_context_block(
        [
            {
                "chunk_id": "leave_policy_v1_chunk_001",
                "source_file": "leave_policy.txt",
                "document_version": "v1",
                "score": 0.89,
                "content": "Leave policy content",
            }
        ]
    )

    assert "chunk_id=leave_policy_v1_chunk_001" in context
    assert "source_file=leave_policy.txt" in context
    assert "Leave policy content" in context


def test_eval_metrics_pass_for_expected_refusal():
    checks = calculate_pass_fail(
        retrieved_sources=[],
        expected_source="",
        answer=REFUSAL_MESSAGE,
        expected_keywords=[],
        should_refuse=True,
    )

    assert checks == {
        "source_pass": True,
        "keyword_pass": True,
        "refusal_pass": True,
        "overall_pass": True,
    }


def test_context_metrics_are_deterministic():
    precision = calculate_context_precision(
        ["leave_policy.txt", "travel_policy.txt"],
        "leave_policy.txt",
    )
    recall = calculate_context_recall(
        "Sick leave requires employees to inform their manager.",
        ["sick leave", "manager"],
        should_refuse=False,
    )
    relevance = calculate_context_relevance(
        "How does sick leave work?",
        "Sick leave is available when employees are unwell.",
    )

    assert precision == 1.0
    assert recall == 1.0
    assert relevance > 0


def test_judge_response_parser_clamps_scores():
    parsed = parse_judge_response(
        """
        ```json
        {
          "faithfulness_score": 1.2,
          "answer_relevance_score": 0.8,
          "llm_context_relevance_score": 0.7,
          "hallucination_score": -0.2,
          "llm_judge_score": 0.9,
          "judge_reason": "Mostly grounded."
        }
        ```
        """
    )

    assert parsed["faithfulness_score"] == 1.0
    assert parsed["hallucination_score"] == 0.0
    assert parsed["llm_judge_score"] == 0.9
    assert parsed["judge_error"] == ""
