"""Streamlit app for Lecture 4 grounded RAG answer generation."""

import html
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chat_service import list_available_policy_files  # noqa: E402
from src.config_loader import load_config  # noqa: E402
from src.generation.answer_generator import generate_answer  # noqa: E402
from src.generation.prompt_loader import load_prompt_registry  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.utils import load_project_env  # noqa: E402


SUGGESTED_QUESTIONS = [
    "How does sick leave work?",
    "Can I claim hotel reimbursement?",
    "What should I do if I lose my laptop?",
    "How do I reset my password?",
]

DOCUMENT_DETAILS = {
    "leave_policy.txt": {
        "title": "Leave Policy",
        "summary": "Casual leave, sick leave, approvals, carry-forward, and emergency leave.",
        "keywords": ["leave", "sick", "holiday"],
    },
    "travel_policy.txt": {
        "title": "Travel Policy",
        "summary": "Travel approval, booking, hotel, food, cab, and reimbursement rules.",
        "keywords": ["travel", "hotel", "cab", "reimbursement"],
    },
    "it_support_faq.txt": {
        "title": "IT Support FAQ",
        "summary": "Lost laptop, password reset, helpdesk, endpoint protection, and incidents.",
        "keywords": ["laptop", "password", "helpdesk"],
    },
}


def main() -> None:
    """Run the Streamlit app."""
    load_project_env()
    config = load_config(str(PROJECT_ROOT / "configs" / "config.yaml"))
    app_config = config["application"]
    data_config = config["data"]
    embedding_config = config["embedding"]
    search_config = config["search"]
    raw_dir = PROJECT_ROOT / data_config["raw_dir"]

    prompt_registry = load_prompt_registry(PROJECT_ROOT / "prompts" / "prompt_registry.yaml")

    logger = get_logger(
        name="lecture-4-streamlit-app",
        log_dir=str(PROJECT_ROOT / config["runtime"]["log_dir"]),
        log_file="lecture_4_rag_generation.log",
    )

    st.set_page_config(
        page_title=app_config["title"],
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    initialize_session_state()
    scroll_to_top_if_requested()

    available_files = list_available_policy_files(raw_dir)

    render_sidebar(data_config, embedding_config, search_config, prompt_registry, available_files)
    render_header(app_config, available_files)
    render_information_message()

    chat_column, context_column = st.columns([0.64, 0.36], gap="large")

    with chat_column:
        render_conversation()

    with context_column:
        render_cards(available_files)
        render_suggested_questions(available_files)

    question = st.chat_input(
        "Ask a company policy question",
    )
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None
        should_scroll_to_top = st.session_state.pending_question_should_scroll
        st.session_state.pending_question_should_scroll = False
    else:
        should_scroll_to_top = False

    if question:
        process_question(
            question=question,
            search_config=search_config,
            logger=logger,
            should_scroll_to_top=should_scroll_to_top,
        )


def initialize_session_state() -> None:
    """Create the Streamlit session keys used by the app."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "pending_question_should_scroll" not in st.session_state:
        st.session_state.pending_question_should_scroll = False
    if "scroll_to_top" not in st.session_state:
        st.session_state.scroll_to_top = False


def scroll_to_top_if_requested() -> None:
    """Scroll to the top after a suggested question has been processed."""
    if not st.session_state.scroll_to_top:
        return

    st.session_state.scroll_to_top = False
    components.html(
        """
        <script>
        const scrollTarget = window.parent || window;
        scrollTarget.scrollTo({ top: 0, behavior: "smooth" });
        </script>
        """,
        height=0,
    )


def process_question(
    question: str,
    search_config: dict[str, Any],
    logger: Any,
    should_scroll_to_top: bool = False,
) -> None:
    """Process manual input and suggested questions through Lecture 4 RAG."""
    st.session_state.messages.append({"role": "user", "content": question})
    logger.info("Received question for grounded RAG generation.")

    try:
        response = generate_answer(
            question=question,
            top_k=int(search_config["top_k"]),
        )
        answer = response["answer"]
        metadata = {
            "source_files": response["source_files"],
            "retrieval_method": "grounded_rag",
            "is_llm_response": True,
            "retrieved_chunks": response.get("retrieved_chunks", []),
            "retrieved_chunk_ids": response.get("retrieved_chunk_ids", []),
            "prompt_version": response["prompt_version"],
            "model_deployment": response["model_deployment"],
            "latency_ms": response["latency_ms"],
            "error_type": None,
            "generation_error": None,
        }
    except Exception as error:
        answer = (
            "Grounded RAG answer generation failed. Check the Lecture 4 required "
            "environment variables, Azure OpenAI chat deployment, and Lecture 3 retrieval setup."
        )
        metadata = {
            "source_files": [],
            "retrieval_method": "grounded_rag",
            "is_llm_response": False,
            "retrieved_chunks": [],
            "retrieved_chunk_ids": [],
            "prompt_version": None,
            "model_deployment": None,
            "latency_ms": None,
            "error_type": type(error).__name__,
            "generation_error": str(error),
        }

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "metadata": metadata,
        }
    )

    logger.info(
        "RAG response completed; prompt=%s; sources=%s; error=%s",
        metadata["prompt_version"],
        metadata["source_files"],
        metadata["error_type"],
    )
    if should_scroll_to_top:
        st.session_state.scroll_to_top = True
    st.rerun()


def inject_styles() -> None:
    """Apply the small design system used by the app."""
    st.markdown(
        """
        <style>
        :root {
            --page-bg: #f6f8fb;
            --card-bg: #ffffff;
            --text: #172033;
            --muted: #667085;
            --border: #e4e7ec;
            --primary: #2563eb;
            --primary-soft: #eff6ff;
            --success-soft: #ecfdf3;
            --success: #067647;
            --warning-soft: #fffaeb;
            --warning: #b54708;
            --shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
        }

        .stApp {
            background: var(--page-bg);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.35rem;
            padding-bottom: 5.5rem;
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--border);
        }

        .app-header,
        .info-callout,
        .content-card,
        .document-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            box-shadow: var(--shadow);
        }

        .app-header {
            padding: 1.05rem 1.2rem;
            margin-bottom: 0.75rem;
        }

        .header-row {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
        }

        .eyebrow {
            color: var(--primary);
            font-size: 0.78rem;
            font-weight: 750;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }

        .app-title {
            color: var(--text);
            font-size: 2rem;
            line-height: 1.16;
            font-weight: 800;
            margin: 0;
        }

        .app-subtitle {
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.45;
            margin: 0.42rem 0 0;
        }

        .status-badge {
            white-space: nowrap;
            border: 1px solid #bfdbfe;
            background: var(--primary-soft);
            color: #1d4ed8;
            border-radius: 999px;
            padding: 0.28rem 0.68rem;
            font-size: 0.82rem;
            font-weight: 700;
            margin-top: 0.15rem;
        }

        .info-callout {
            border-left: 4px solid var(--primary);
            padding: 0.72rem 0.9rem;
            margin-bottom: 0.95rem;
            color: #344054;
            font-size: 0.92rem;
            line-height: 1.45;
        }

        .content-card {
            padding: 1rem;
            margin-bottom: 0.9rem;
        }

        .section-title {
            color: var(--text);
            font-size: 1.08rem;
            font-weight: 800;
            margin: 0 0 0.2rem;
        }

        .section-help {
            color: var(--muted);
            font-size: 0.9rem;
            margin: 0 0 0.75rem;
        }

        .document-card {
            padding: 0.82rem 0.88rem;
            margin-bottom: 0.65rem;
        }

        .document-title {
            color: var(--text);
            font-weight: 800;
            font-size: 0.96rem;
            margin-bottom: 0.22rem;
        }

        .document-summary {
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.38;
        }

        .keyword {
            display: inline-block;
            margin: 0.5rem 0.25rem 0 0;
            padding: 0.14rem 0.45rem;
            border-radius: 999px;
            background: #f2f4f7;
            color: #475467;
            font-size: 0.75rem;
            font-weight: 700;
        }

        .empty-state {
            border: 1px dashed #cbd5e1;
            background: #f8fafc;
            border-radius: 8px;
            padding: 1rem;
            color: var(--muted);
            text-align: center;
        }

        .answer-details {
            color: var(--muted);
            font-size: 0.9rem;
        }

        .metadata-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }

        .metadata-table td {
            border-bottom: 1px solid var(--border);
            padding: 0.35rem 0;
            vertical-align: top;
        }

        .metadata-table td:first-child {
            color: var(--muted);
            width: 36%;
        }

        div.stButton > button {
            min-height: 2.8rem;
            border-radius: 8px;
            border: 1px solid #c7d7fe;
            background: #ffffff;
            color: #1849a9;
            font-weight: 650;
            white-space: normal;
        }

        div.stButton > button:hover {
            border-color: var(--primary);
            color: var(--primary);
            background: var(--primary-soft);
        }

        div[data-testid="stChatMessage"] {
            background: transparent;
        }

        div[data-testid="stChatInput"] {
            max-width: 1180px;
            margin: 0 auto;
        }

        @media (max-width: 900px) {
            .header-row {
                display: block;
            }

            .status-badge {
                display: inline-block;
                margin-top: 0.75rem;
            }

            .app-title {
                font-size: 1.65rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(app_config: dict[str, str], available_files: list[str]) -> None:
    """Render the compact application header."""
    status = "Grounded RAG active"
    st.markdown(
        f"""
        <div class="app-header">
            <div class="header-row">
                <div>
                    <div class="eyebrow">Lecture 4 internal demo</div>
                    <h1 class="app-title">{html.escape(app_config["title"])}</h1>
                    <p class="app-subtitle">Grounded answer generation with prompt versions, retrieved sources, and model metadata.</p>
                </div>
                <div class="status-badge">{html.escape(status)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_information_message() -> None:
    """Render a subtle scope callout."""
    st.markdown(
        """
        <div class="info-callout">
            <strong>Lecture 4 flow:</strong> retrieve evidence chunks, place them into a versioned prompt,
            call the Azure OpenAI chat deployment, and show a grounded answer with sources.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(
    data_config: dict[str, Any],
    embedding_config: dict[str, Any],
    search_config: dict[str, Any],
    prompt_registry: dict[str, Any],
    available_files: list[str],
) -> None:
    """Render clear sidebar sections for demo context."""
    st.sidebar.markdown("## Demo Guide")
    st.sidebar.caption("Use this panel to explain what learners are seeing.")

    st.sidebar.markdown("### What happens")
    st.sidebar.markdown(
        "1. Ask a policy question.\n"
        "2. Lecture 3 retrieves top-k chunks.\n"
        "3. Lecture 4 loads the active prompt.\n"
        "4. Azure OpenAI chat generates an answer.\n"
        "5. The app displays answer, sources, and metadata."
    )

    st.sidebar.markdown("### Required for Lecture 4")
    required_env_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_INDEX_NAME",
    ]
    for env_var in required_env_vars:
        st.sidebar.write(f"{env_var}: `{_env_status(env_var)}`")

    st.sidebar.markdown("### Data sources")
    if available_files:
        for file_name in available_files:
            label = DOCUMENT_DETAILS.get(file_name, {}).get("title", file_name)
            st.sidebar.write(f"- {label}")
    else:
        st.sidebar.warning("No `.txt` files found.")

    st.sidebar.markdown("### Settings")
    st.sidebar.write(f"Version: `{data_config['document_version']}`")
    st.sidebar.write(f"Raw folder: `{data_config['raw_dir']}`")
    st.sidebar.write(f"Active prompt: `{prompt_registry.get('active_prompt', 'unknown')}`")
    st.sidebar.write(f"Chat deployment: `{os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT', 'missing')}`")
    embedding_deployment = os.getenv(embedding_config["deployment_env_var"], "text-embedding-3-small")
    search_index = os.getenv(search_config["index_name_env_var"], "policy-knowledge-index-v1")
    st.sidebar.write("Answer mode: `Grounded RAG`")
    st.sidebar.write(f"Embedding deployment: `{embedding_deployment}`")
    st.sidebar.write(f"Search index: `{search_index}`")
    st.sidebar.write(f"Top-k: `{search_config['top_k']}`")

    with st.sidebar.expander("Lecture 4 scope", expanded=False):
        st.write("Uses the existing Lecture 3 Azure AI Search index")
        st.write("Does not recreate the index")
        st.write("Does not reprocess documents")
        st.write("Does not use LangChain or LlamaIndex")
        st.write("Evaluation is run separately from the app")


def render_cards(available_files: list[str]) -> None:
    """Render document cards in the right-side context panel."""
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Available documents</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-help">These documents produced the chunks indexed for retrieval.</div>',
        unsafe_allow_html=True,
    )

    if not available_files:
        st.markdown(
            """
            <div class="empty-state">
                No policy documents were found. Add `.txt` files under
                `data/raw/policies_v1`.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for file_name in available_files:
        details = DOCUMENT_DETAILS.get(
            file_name,
            {
                "title": file_name,
                "summary": "Local text document available for keyword lookup.",
                "keywords": [],
            },
        )
        keywords = "".join(
            f'<span class="keyword">{html.escape(keyword)}</span>'
            for keyword in details["keywords"]
        )
        st.markdown(
            f"""
            <div class="document-card">
                <div class="document-title">{html.escape(details["title"])}</div>
                <div class="document-summary">{html.escape(details["summary"])}</div>
                <div>{keywords}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_suggested_questions(available_files: list[str]) -> None:
    """Render suggested question buttons."""
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Suggested questions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-help">Use these for a quick classroom demo.</div>',
        unsafe_allow_html=True,
    )

    first_row = st.columns(2)
    second_row = st.columns(2)
    for index, (column, question) in enumerate(
        zip(first_row + second_row, SUGGESTED_QUESTIONS),
        start=1,
    ):
        with column:
            if st.button(
                question,
                key=f"suggested_question_{index}",
                use_container_width=True,
            ):
                st.session_state.pending_question = question
                st.session_state.pending_question_should_scroll = True

    st.markdown("</div>", unsafe_allow_html=True)


def render_conversation() -> None:
    """Render native Streamlit chat messages."""
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Chat</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-help">Ask a policy question. The app retrieves chunks and generates a grounded answer from them.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="empty-state">
                Start with a suggested question or type your own question in the chat box.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                render_answer_details(message.get("metadata", {}))

    st.markdown("</div>", unsafe_allow_html=True)


def render_answer_details(metadata: dict[str, Any]) -> None:
    """Render technical response details inside an expander."""
    source_files = metadata.get("source_files", [])
    source_text = ", ".join(source_files) if source_files else "No sources returned"
    retrieval_method = metadata.get("retrieval_method", "grounded_rag")
    is_llm_response = metadata.get("is_llm_response", False)
    error_type = metadata.get("error_type")
    generation_error = metadata.get("generation_error")
    retrieved_chunks = metadata.get("retrieved_chunks", [])
    prompt_version = metadata.get("prompt_version") or "None"
    model_deployment = metadata.get("model_deployment") or "None"
    latency_ms = metadata.get("latency_ms")

    if retrieved_chunks:
        render_retrieved_chunks(retrieved_chunks)

    title = "RAG answer details"
    if error_type:
        title = "Error details"

    with st.expander(title, expanded=False):
        st.markdown(
            f"""
            <table class="metadata-table">
                <tr><td>Source files</td><td>{html.escape(source_text)}</td></tr>
                <tr><td>Retrieval method</td><td>{html.escape(retrieval_method)}</td></tr>
                <tr><td>LLM response</td><td>{str(is_llm_response)}</td></tr>
                <tr><td>Prompt version</td><td>{html.escape(str(prompt_version))}</td></tr>
                <tr><td>Model deployment</td><td>{html.escape(str(model_deployment))}</td></tr>
                <tr><td>Latency ms</td><td>{html.escape(str(latency_ms))}</td></tr>
                <tr><td>Error type</td><td>{html.escape(error_type or "None")}</td></tr>
            </table>
            """,
            unsafe_allow_html=True,
        )
        if generation_error:
            st.caption(f"Generation error: {generation_error}")
        st.caption(
            "Lecture 4 uses retrieved evidence chunks inside a versioned prompt "
            "before calling the chat model."
        )


def _env_status(env_var: str) -> str:
    """Return a short display status for required environment variables."""
    return "set" if os.getenv(env_var) else "missing"


def render_retrieved_chunks(retrieved_chunks: list[dict[str, Any]]) -> None:
    """Render ranked chunks returned by Azure AI Search."""
    st.markdown("#### Retrieved evidence")
    for chunk in retrieved_chunks:
        rank = chunk.get("rank")
        chunk_id = chunk.get("chunk_id") or "unknown chunk"
        title = f"Rank {rank}: {chunk_id}"
        with st.expander(title, expanded=rank == 1):
            st.write(chunk.get("content", ""))
            score = chunk.get("score")
            if isinstance(score, float):
                score_text = f"{score:.4f}"
            else:
                score_text = str(score)
            st.markdown(
                f"""
                <table class="metadata-table">
                    <tr><td>Source file</td><td>{html.escape(str(chunk.get("source_file")))}</td></tr>
                    <tr><td>Document version</td><td>{html.escape(str(chunk.get("document_version")))}</td></tr>
                    <tr><td>Chunk ID</td><td>{html.escape(str(chunk_id))}</td></tr>
                    <tr><td>Chunk index</td><td>{html.escape(str(chunk.get("chunk_index")))}</td></tr>
                    <tr><td>Search score</td><td>{html.escape(score_text)}</td></tr>
                    <tr><td>Index version</td><td>{html.escape(str(chunk.get("index_version")))}</td></tr>
                    <tr><td>Retrieval mode</td><td>{html.escape(str(chunk.get("retrieval_mode")))}</td></tr>
                </table>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
