"""Fail the Lecture 5 CI workflow when RAG evaluation quality is too low."""

import argparse
import csv
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY_PATH = Path("evaluations/results/comparison_summary.csv")
DEFAULT_PROMPT_VERSION = "qa_prompt_v2"
DEFAULT_MIN_OVERALL_PASS_RATE = 0.80


class QualityGateError(RuntimeError):
    """Raised when the RAG quality gate cannot pass."""


def check_quality_gate(
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    min_overall_pass_rate: float = DEFAULT_MIN_OVERALL_PASS_RATE,
) -> dict[str, Any]:
    """Validate the selected prompt version against the minimum pass rate."""
    path = Path(summary_path)
    if not path.exists():
        raise QualityGateError(f"Evaluation summary not found: {path}")

    rows = _load_summary_rows(path)
    matching_row = _find_prompt_row(rows, prompt_version)
    pass_rate = _parse_pass_rate(matching_row, path)
    passed = pass_rate >= min_overall_pass_rate

    result = {
        "summary_path": path.as_posix(),
        "prompt_version": prompt_version,
        "overall_pass_rate": pass_rate,
        "min_overall_pass_rate": min_overall_pass_rate,
        "passed": passed,
    }

    if not passed:
        raise QualityGateError(
            f"Quality gate failed for {prompt_version}: "
            f"overall_pass_rate={pass_rate:.4f}, "
            f"required>={min_overall_pass_rate:.4f}"
        )

    return result


def _load_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise QualityGateError(f"Evaluation summary is empty: {path}")
    return rows


def _find_prompt_row(rows: list[dict[str, str]], prompt_version: str) -> dict[str, str]:
    for row in rows:
        if row.get("prompt_version") == prompt_version:
            return row

    available_versions = sorted(row.get("prompt_version", "") for row in rows)
    raise QualityGateError(
        f"Prompt version {prompt_version!r} was not found in comparison summary. "
        f"Available versions: {', '.join(available_versions)}"
    )


def _parse_pass_rate(row: dict[str, str], path: Path) -> float:
    raw_value = row.get("overall_pass_rate")
    if raw_value is None or raw_value == "":
        raise QualityGateError(f"Missing overall_pass_rate column in {path}")

    try:
        return float(raw_value)
    except ValueError as error:
        raise QualityGateError(f"Invalid overall_pass_rate value in {path}: {raw_value}") from error


def main() -> None:
    """CLI entry point used by GitHub Actions."""
    parser = argparse.ArgumentParser(description="Check the Lecture 5 RAG quality gate.")
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--min-overall-pass-rate", type=float, default=DEFAULT_MIN_OVERALL_PASS_RATE)
    args = parser.parse_args()

    try:
        result = check_quality_gate(
            summary_path=args.summary_path,
            prompt_version=args.prompt_version,
            min_overall_pass_rate=args.min_overall_pass_rate,
        )
    except QualityGateError as error:
        print(error)
        raise SystemExit(1) from error

    print(
        f"Quality gate passed for {result['prompt_version']}: "
        f"overall_pass_rate={result['overall_pass_rate']:.4f}, "
        f"required>={result['min_overall_pass_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
