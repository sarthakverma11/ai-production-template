"""Policy lookup service for Lecture 2 keyword lookup and Lecture 3 retrieval."""

from pathlib import Path
from typing import Any

from src.ingestion.document_loader import load_documents
from src.search.retrieval_service import retrieve_chunks


KEYWORD_TO_SOURCE_FILE = {
    "leave": "leave_policy.txt",
    "holiday": "leave_policy.txt",
    "sick": "leave_policy.txt",
    "travel": "travel_policy.txt",
    "hotel": "travel_policy.txt",
    "cab": "travel_policy.txt",
    "reimbursement": "travel_policy.txt",
    "laptop": "it_support_faq.txt",
    "password": "it_support_faq.txt",
    "helpdesk": "it_support_faq.txt",
}

FALLBACK_ANSWER = (
    "No matching local policy was found using the current keyword-based lookup. "
    "Semantic retrieval will be added in a later lecture."
)

SEMANTIC_RETRIEVAL_MESSAGE = (
    "Semantic retrieval returned the evidence chunks below. "
    "No LLM-generated answer is used in Lecture 3."
)


def answer_question(
    question: str,
    raw_dir: str | Path = "data/raw/policies_v1",
    document_version: str = "v1",
) -> dict[str, Any]:
    """Answer a question using local keyword lookup only."""
    selected_source_file = find_matching_source_file(question)

    if selected_source_file is None:
        return {
            "answer": FALLBACK_ANSWER,
            "source_file": None,
            "retrieval_method": "keyword_lookup",
            "is_llm_response": False,
        }

    documents = load_documents(
        folder_path=raw_dir,
        document_version=document_version,
        supported_extensions=[".txt"],
    )
    documents_by_source = {document["source_file"]: document for document in documents}
    selected_document = documents_by_source.get(selected_source_file)

    if selected_document is None:
        return {
            "answer": FALLBACK_ANSWER,
            "source_file": None,
            "retrieval_method": "keyword_lookup",
            "is_llm_response": False,
        }

    excerpt = _select_excerpt(selected_document["text"], question)
    answer = (
        "This local keyword lookup matched your question to "
        f"`{selected_source_file}`.\n\n{excerpt}"
    )

    return {
        "answer": answer,
        "source_file": selected_source_file,
        "retrieval_method": "keyword_lookup",
        "is_llm_response": False,
    }


def answer_question_with_retrieval(
    question: str,
    mode: str = "semantic",
    raw_dir: str | Path = "data/raw/policies_v1",
    document_version: str = "v1",
    top_k: int | None = None,
    keyword_fallback_enabled: bool = False,
) -> dict[str, Any]:
    """Answer through the configured Lecture 3 retrieval mode."""
    normalized_mode = mode.lower().strip()
    if normalized_mode == "keyword":
        return answer_question(
            question=question,
            raw_dir=raw_dir,
            document_version=document_version,
        )

    if normalized_mode != "semantic":
        raise ValueError("Retrieval mode must be either 'keyword' or 'semantic'.")

    try:
        retrieved_chunks = retrieve_chunks(
            query=question,
            top_k=top_k,
            document_version=document_version,
        )
    except Exception as error:
        if keyword_fallback_enabled:
            fallback = answer_question(
                question=question,
                raw_dir=raw_dir,
                document_version=document_version,
            )
            fallback["error_type"] = type(error).__name__
            fallback["retrieval_error"] = str(error)
            fallback["fallback_used"] = True
            return fallback

        return {
            "answer": (
                "Semantic retrieval failed. Check Azure environment variables, "
                "the search index, and indexing status before retrying."
            ),
            "source_file": None,
            "retrieval_method": "semantic_vector_search",
            "is_llm_response": False,
            "retrieved_chunks": [],
            "error_type": type(error).__name__,
            "retrieval_error": str(error),
            "fallback_used": False,
        }

    return {
        "answer": SEMANTIC_RETRIEVAL_MESSAGE,
        "source_file": retrieved_chunks[0]["source_file"] if retrieved_chunks else None,
        "retrieval_method": "semantic_vector_search",
        "is_llm_response": False,
        "retrieved_chunks": retrieved_chunks,
        "error_type": None,
        "fallback_used": False,
    }


def find_matching_source_file(question: str) -> str | None:
    """Find the first policy file whose keyword appears in the question."""
    normalized_question = question.lower()

    for keyword, source_file in KEYWORD_TO_SOURCE_FILE.items():
        if keyword in normalized_question:
            return source_file

    return None


def list_available_policy_files(raw_dir: str | Path = "data/raw/policies_v1") -> list[str]:
    """List available local policy files."""
    folder = Path(raw_dir)
    if not folder.exists():
        return []

    return sorted(file_path.name for file_path in folder.glob("*.txt") if file_path.is_file())


def _select_excerpt(text: str, question: str, max_characters: int = 700) -> str:
    """Return a small matching paragraph or the beginning of the document."""
    normalized_question = question.lower()
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]

    for paragraph in paragraphs:
        if any(word in paragraph.lower() for word in normalized_question.split()):
            return paragraph[:max_characters]

    if paragraphs:
        return paragraphs[0][:max_characters]

    return text.strip()[:max_characters]
