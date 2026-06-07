"""Simple text cleaning for local documents."""

import re


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph readability."""
    stripped_text = text.strip()
    stripped_text = re.sub(r"[ \t]+", " ", stripped_text)
    stripped_text = re.sub(r"\n\s*\n\s*\n+", "\n\n", stripped_text)
    lines = [line.strip() for line in stripped_text.splitlines()]
    return "\n".join(lines).strip()

