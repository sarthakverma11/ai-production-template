"""Grounded RAG answer generation for Lecture 4."""

import argparse
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import yaml

from src.generation.prompt_loader import format_prompt, load_prompt
from src.retrieval.retriever_adapter import retrieve_chunks
from src.utils import load_project_env


MODEL_REGISTRY_PATH = Path("configs/model_registry.yaml")


class AnswerGenerationError(RuntimeError):
    """Raised when the answer generator cannot complete."""


def generate_answer(
    question: str,
    prompt_version: str | None = None,
    top_k: int = 3,
    enable_tracing: bool = False,
) -> dict[str, Any]:
    """Retrieve context, format the selected prompt, call Azure OpenAI, and return metadata."""
    load_project_env()
    if not question.strip():
        raise AnswerGenerationError("Question must be a non-empty string.")

    started_at = time.perf_counter()
    with _optional_span(
        enable_tracing,
        name="rag_answer_generation",
        span_type="CHAIN",
        attributes={"top_k": top_k, "requested_prompt_version": prompt_version or "active"},
    ) as root_span:
        if root_span is not None:
            root_span.set_inputs({"question": question})

        with _optional_span(enable_tracing, name="load_model_config", span_type="PARSER") as span:
            model_config = load_model_config()
            deployment = resolve_chat_deployment(model_config)
            if span is not None:
                span.set_outputs(
                    {
                        "provider": model_config.get("provider"),
                        "deployment": deployment,
                        "temperature": model_config.get("temperature"),
                        "max_tokens": model_config.get("max_tokens"),
                    }
                )

        with _optional_span(enable_tracing, name="retrieve_chunks", span_type="RETRIEVER") as span:
            retrieved_chunks = retrieve_chunks(question, top_k=top_k)
            chunk_ids = _unique_values(retrieved_chunks, "chunk_id")
            source_files = _unique_values(retrieved_chunks, "source_file")
            if span is not None:
                span.set_inputs({"question": question, "top_k": top_k})
                span.set_outputs(
                    {
                        "chunk_ids": chunk_ids,
                        "source_files": source_files,
                        "scores": [chunk.get("score") for chunk in retrieved_chunks],
                    }
                )

        with _optional_span(enable_tracing, name="build_context", span_type="PARSER") as span:
            context = build_context_block(retrieved_chunks)
            if span is not None:
                span.set_outputs(
                    {
                        "context_characters": len(context),
                        "context_chunk_count": len(retrieved_chunks),
                    }
                )

        with _optional_span(enable_tracing, name="load_prompt", span_type="PROMPT") as span:
            prompt_text, prompt_metadata = load_prompt(prompt_version)
            final_prompt = format_prompt(prompt_text, context=context, question=question)
            if span is not None:
                span.set_outputs(
                    {
                        "prompt_version": prompt_metadata["prompt_version"],
                        "prompt_file": prompt_metadata.get("file_path"),
                        "formatted_prompt_characters": len(final_prompt),
                    }
                )

        with _optional_span(enable_tracing, name="call_azure_openai_chat", span_type="LLM") as span:
            answer = call_azure_openai_chat(
                prompt=final_prompt,
                deployment=deployment,
                temperature=float(model_config.get("temperature", 0.2)),
                max_tokens=int(model_config.get("max_tokens", 500)),
            )
            if span is not None:
                span.set_inputs(
                    {
                        "deployment": deployment,
                        "temperature": float(model_config.get("temperature", 0.2)),
                        "max_tokens": int(model_config.get("max_tokens", 500)),
                    }
                )
                span.set_outputs(
                    {
                        "answer_preview": answer[:500],
                        "answer_characters": len(answer),
                    }
                )

        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        result = {
            "answer": answer,
            "prompt_version": prompt_metadata["prompt_version"],
            "retrieved_chunk_ids": chunk_ids,
            "source_files": source_files,
            "model_deployment": deployment,
            "latency_ms": latency_ms,
            "retrieved_chunks": retrieved_chunks,
            "index_versions": _unique_values(retrieved_chunks, "index_version"),
        }
        if root_span is not None:
            root_span.set_outputs(
                {
                    "answer_preview": answer[:500],
                    "prompt_version": result["prompt_version"],
                    "source_files": source_files,
                    "chunk_ids": chunk_ids,
                    "latency_ms": latency_ms,
                }
            )

        return result


def load_model_config(config_path: str | Path = MODEL_REGISTRY_PATH) -> dict[str, Any]:
    """Load the active chat model settings from configs/model_registry.yaml."""
    path = Path(config_path)
    if not path.exists():
        raise AnswerGenerationError(f"Model registry not found: {path}")

    try:
        registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise AnswerGenerationError(f"Invalid model registry YAML: {path}") from error

    if not isinstance(registry, dict) or not isinstance(registry.get("active_chat_model"), dict):
        raise AnswerGenerationError("Model registry must contain active_chat_model settings.")

    return dict(registry["active_chat_model"])


def resolve_chat_deployment(model_config: dict[str, Any]) -> str:
    """Resolve deployment from env first, then from the registry placeholder."""
    env_var = model_config.get("deployment_env_var", "AZURE_OPENAI_CHAT_DEPLOYMENT")
    deployment = os.getenv(env_var) or model_config.get("deployment_name")
    if not deployment or deployment == "your-chat-deployment-name":
        raise AnswerGenerationError(
            "Missing Azure OpenAI chat deployment. Set AZURE_OPENAI_CHAT_DEPLOYMENT "
            "or update configs/model_registry.yaml."
        )
    return str(deployment)


def build_context_block(chunks: list[dict[str, Any]]) -> str:
    """Turn retrieved chunks into a readable context block with source metadata."""
    if not chunks:
        return "No retrieved context was available."

    context_parts: list[str] = []
    for chunk in chunks:
        source_file = chunk.get("source_file") or "unknown_source"
        chunk_id = chunk.get("chunk_id") or chunk.get("id") or "unknown_chunk"
        document_version = chunk.get("document_version") or "unknown_version"
        score = chunk.get("score")
        content = chunk.get("content") or chunk.get("chunk_text") or ""
        header = (
            f"[chunk_id={chunk_id}; source_file={source_file}; "
            f"document_version={document_version}; score={score}]"
        )
        context_parts.append(f"{header}\n{content}".strip())

    return "\n\n---\n\n".join(context_parts)


def call_azure_openai_chat(
    prompt: str,
    deployment: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call Azure OpenAI chat completions with environment-based credentials."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    if not endpoint:
        raise AnswerGenerationError("Missing AZURE_OPENAI_ENDPOINT.")
    if not api_key:
        raise AnswerGenerationError("Missing AZURE_OPENAI_API_KEY.")
    if not api_version:
        raise AnswerGenerationError("Missing AZURE_OPENAI_API_VERSION.")

    try:
        from openai import AzureOpenAI
    except ModuleNotFoundError as error:
        raise AnswerGenerationError("openai is required. Install dependencies first.") from error

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        timeout=_openai_timeout_seconds(),
        max_retries=1,
    )

    try:
        response = _create_chat_completion(
            client=client,
            deployment=deployment,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as error:
        raise AnswerGenerationError(f"Azure OpenAI chat completion failed: {type(error).__name__}") from error

    message = response.choices[0].message.content
    if not message:
        raise AnswerGenerationError("Azure OpenAI returned an empty answer.")

    return message.strip()


def _create_chat_completion(
    client: Any,
    deployment: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> Any:
    """Call chat completions using the current token parameter, with a fallback."""
    request = {
        "model": deployment,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    try:
        return client.chat.completions.create(
            **request,
            max_completion_tokens=max_tokens,
        )
    except TypeError:
        return client.chat.completions.create(
            **request,
            max_tokens=max_tokens,
        )


def _openai_timeout_seconds() -> float:
    """Return Azure OpenAI timeout in seconds with a classroom-friendly default."""
    raw_timeout = os.getenv("AZURE_OPENAI_TIMEOUT_SECONDS", "30")
    try:
        timeout = float(raw_timeout)
    except ValueError:
        return 30.0

    return max(5.0, timeout)


def _unique_values(chunks: list[dict[str, Any]], key: str) -> list[str]:
    values: list[str] = []
    for chunk in chunks:
        value = chunk.get(key)
        if value and str(value) not in values:
            values.append(str(value))
    return values


def _optional_span(enabled: bool, name: str, span_type: str, attributes: dict[str, Any] | None = None) -> Any:
    """Create an MLflow span only when tracing is explicitly enabled."""
    if not enabled:
        return nullcontext(None)

    import mlflow

    return mlflow.start_span(name=name, span_type=span_type, attributes=attributes)


def main() -> None:
    """CLI entry point for a single RAG answer."""
    parser = argparse.ArgumentParser(description="Generate a grounded Lecture 4 RAG answer.")
    parser.add_argument(
        "question",
        nargs="?",
        default="How many days of annual leave do employees receive?",
    )
    parser.add_argument("--prompt-version", default=None, help="Prompt version from prompt_registry.yaml.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve.")
    args = parser.parse_args()

    try:
        result = generate_answer(args.question, prompt_version=args.prompt_version, top_k=args.top_k)
    except Exception as error:
        raise SystemExit(f"Answer generation failed: {error}") from error

    print(f"Question: {args.question}")
    print(f"Prompt version: {result['prompt_version']}")
    print(f"Model deployment: {result['model_deployment']}")
    print(f"Retrieved sources: {', '.join(result['source_files'])}")
    print()
    print(result["answer"])


if __name__ == "__main__":
    main()
