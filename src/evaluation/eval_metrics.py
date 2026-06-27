"""Beginner-friendly deterministic RAG evaluation checks."""


REFUSAL_MESSAGE = "I do not have enough information in the provided documents."


def check_expected_source(retrieved_sources: list[str], expected_source: str) -> bool:
    """Check whether the expected source document appears in retrieved sources."""
    if not expected_source:
        return True

    expected = expected_source.lower()
    return any(expected in source.lower() for source in retrieved_sources)


def check_expected_keywords(answer: str, expected_keywords: list[str]) -> bool:
    """Check whether all expected keywords appear in the answer."""
    if not expected_keywords:
        return True

    normalized_answer = answer.lower()
    return all(keyword.lower() in normalized_answer for keyword in expected_keywords)


def check_refusal_behavior(answer: str, should_refuse: bool) -> bool:
    """Check whether the answer refused only when the golden case expects refusal."""
    contains_refusal = REFUSAL_MESSAGE.lower() in answer.lower()
    return contains_refusal if should_refuse else not contains_refusal


def calculate_pass_fail(
    retrieved_sources: list[str],
    expected_source: str,
    answer: str,
    expected_keywords: list[str],
    should_refuse: bool,
) -> dict[str, bool]:
    """Return individual checks plus one overall pass/fail value."""
    source_pass = check_expected_source(retrieved_sources, expected_source)
    keyword_pass = check_expected_keywords(answer, expected_keywords)
    refusal_pass = check_refusal_behavior(answer, should_refuse)

    return {
        "source_pass": source_pass,
        "keyword_pass": keyword_pass,
        "refusal_pass": refusal_pass,
        "overall_pass": source_pass and keyword_pass and refusal_pass,
    }


def calculate_context_precision(retrieved_sources: list[str], expected_source: str) -> float:
    """Approximate whether expected context appears early in the retrieved ranking."""
    if not expected_source:
        return 1.0
    if not retrieved_sources:
        return 0.0

    expected = expected_source.lower()
    hits = 0
    precision_sum = 0.0
    for index, source in enumerate(retrieved_sources, start=1):
        if expected in source.lower():
            hits += 1
            precision_sum += hits / index

    return round(precision_sum / hits, 4) if hits else 0.0


def calculate_context_recall(retrieved_context: str, expected_keywords: list[str], should_refuse: bool) -> float:
    """Approximate whether retrieved context contains the expected answer clues."""
    if should_refuse:
        return 1.0
    if not expected_keywords:
        return 1.0
    if not retrieved_context:
        return 0.0

    normalized_context = retrieved_context.lower()
    matched_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in normalized_context)
    return round(matched_keywords / len(expected_keywords), 4)


def calculate_context_relevance(question: str, retrieved_context: str) -> float:
    """Simple lexical relevance score used beside the LLM judge score."""
    question_terms = {
        term.strip(".,?!:;").lower()
        for term in question.split()
        if len(term.strip(".,?!:;")) > 2
    }
    if not question_terms:
        return 1.0
    if not retrieved_context:
        return 0.0

    normalized_context = retrieved_context.lower()
    matched_terms = sum(1 for term in question_terms if term in normalized_context)
    return round(matched_terms / len(question_terms), 4)
