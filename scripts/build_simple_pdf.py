"""Build a simple text PDF from a Markdown handout using only the standard library."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path


PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LEFT_MARGIN = 54
TOP_MARGIN = 54
BOTTOM_MARGIN = 54
LINE_HEIGHT = 13
BODY_SIZE = 10
HEADING_SIZE = 15
TITLE_SIZE = 18
MONO_SIZE = 9


def markdown_to_pdf(markdown_path: Path, pdf_path: Path) -> None:
    """Convert a small Markdown handout to a readable PDF."""
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    pages = _layout_markdown(lines)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(_build_pdf_bytes(pages))


def _layout_markdown(lines: list[str]) -> list[list[tuple[str, int, str]]]:
    pages: list[list[tuple[str, int, str]]] = [[]]
    y = PAGE_HEIGHT - TOP_MARGIN
    in_code_block = False

    def add_line(text: str, size: int = BODY_SIZE, font: str = "F1") -> None:
        nonlocal y
        if y < BOTTOM_MARGIN:
            pages.append([])
            y = PAGE_HEIGHT - TOP_MARGIN
        pages[-1].append((text, size, font))
        y -= LINE_HEIGHT if size <= BODY_SIZE else LINE_HEIGHT + 5

    for raw_line in lines:
        line = raw_line.rstrip()

        if line.startswith("```"):
            in_code_block = not in_code_block
            add_line("", BODY_SIZE, "F1")
            continue

        if not line:
            add_line("", BODY_SIZE, "F1")
            continue

        if in_code_block:
            for wrapped in _wrap(line, width=84):
                add_line(wrapped, MONO_SIZE, "F2")
            continue

        if line.startswith("# "):
            add_line(line[2:], TITLE_SIZE, "F1")
            add_line("", BODY_SIZE, "F1")
            continue

        if line.startswith("## "):
            add_line(line[3:], HEADING_SIZE, "F1")
            continue

        if line.startswith("- "):
            for index, wrapped in enumerate(_wrap(line[2:], width=82)):
                prefix = "- " if index == 0 else "  "
                add_line(prefix + wrapped, BODY_SIZE, "F1")
            continue

        for wrapped in _wrap(line, width=88):
            add_line(wrapped, BODY_SIZE, "F1")

    return pages


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, replace_whitespace=False) or [""]


def _build_pdf_bytes(pages: list[list[tuple[str, int, str]]]) -> bytes:
    objects: list[bytes] = []
    page_object_numbers = []

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    for page_number, page_lines in enumerate(pages, start=1):
        content = _page_content(page_lines, page_number, len(pages))
        content_object_number = len(objects) + 1
        objects.append(
            b"<< /Length "
            + str(len(content)).encode("ascii")
            + b" >>\nstream\n"
            + content
            + b"endstream"
        )
        page_object_number = len(objects) + 1
        page_object_numbers.append(page_object_number)
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 "
                f"{PAGE_WIDTH} {PAGE_HEIGHT}] "
                "/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
                f"/Contents {content_object_number} 0 R >>"
            ).encode("ascii")
        )

    kids = " ".join(f"{number} 0 R" for number in page_object_numbers)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_object_numbers)} >>".encode(
        "ascii"
    )

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_number} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_position = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            "trailer\n"
            f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            "startxref\n"
            f"{xref_position}\n"
            "%%EOF\n"
        ).encode("ascii")
    )
    return bytes(pdf)


def _page_content(
    page_lines: list[tuple[str, int, str]],
    page_number: int,
    total_pages: int,
) -> bytes:
    commands: list[str] = []
    y = PAGE_HEIGHT - TOP_MARGIN
    for text, size, font in page_lines:
        escaped = _escape_pdf_text(text)
        commands.append(f"BT /{font} {size} Tf {LEFT_MARGIN} {y} Td ({escaped}) Tj ET")
        y -= LINE_HEIGHT if size <= BODY_SIZE else LINE_HEIGHT + 5

    footer = f"Lecture 5 RAG CI Handout - Page {page_number} of {total_pages}"
    commands.append(
        f"BT /F1 8 Tf {LEFT_MARGIN} 28 Td ({_escape_pdf_text(footer)}) Tj ET"
    )
    return ("\n".join(commands) + "\n").encode("latin-1", errors="replace")


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a simple handout PDF.")
    parser.add_argument("markdown_path")
    parser.add_argument("pdf_path")
    args = parser.parse_args()

    markdown_to_pdf(Path(args.markdown_path), Path(args.pdf_path))
    print(f"Created PDF: {args.pdf_path}")


if __name__ == "__main__":
    main()
