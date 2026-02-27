from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import fitz  # PyMuPDF


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


HEADER_FOOTER_LINE_RE = re.compile(r"^\s*(page\s*\d+|\d+\s*/\s*\d+)\s*$", re.IGNORECASE)


def extract_pdf_text(path: Path) -> list[PageText]:
    doc = fitz.open(path)
    pages: list[PageText] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        pages.append(PageText(page_number=i + 1, text=text))
    doc.close()
    return pages


def remove_common_headers_footers(pages: list[PageText]) -> list[PageText]:
    # Heuristic: remove first/last line if repeated across many pages
    first_lines = {}
    last_lines = {}

    def _first_line(t: str) -> str:
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        return lines[0] if lines else ""

    def _last_line(t: str) -> str:
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        return lines[-1] if lines else ""

    for p in pages:
        fl = _first_line(p.text)
        ll = _last_line(p.text)
        first_lines[fl] = first_lines.get(fl, 0) + 1
        last_lines[ll] = last_lines.get(ll, 0) + 1

    min_repeat = max(3, int(0.4 * len(pages))) if pages else 3
    common_first = {k for k, v in first_lines.items() if k and v >= min_repeat}
    common_last = {k for k, v in last_lines.items() if k and v >= min_repeat}

    cleaned: list[PageText] = []
    for p in pages:
        lines = p.text.splitlines()
        if lines:
            if lines[0].strip() in common_first:
                lines = lines[1:]
            if lines and (
                lines[-1].strip() in common_last or HEADER_FOOTER_LINE_RE.match(lines[-1].strip())
            ):
                lines = lines[:-1]
        cleaned.append(PageText(page_number=p.page_number, text="\n".join(lines)))
    return cleaned
