from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fitz  # PyMuPDF


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


HEADER_FOOTER_LINE_RE = re.compile(r"^\s*(page\s*\d+|\d+\s*/\s*\d+)\s*$", re.IGNORECASE)
PDF_DATE_RE = re.compile(
    r"^D:"
    r"(?P<year>\d{4})"
    r"(?P<month>\d{2})?"
    r"(?P<day>\d{2})?"
    r"(?P<hour>\d{2})?"
    r"(?P<minute>\d{2})?"
    r"(?P<second>\d{2})?"
    r"(?P<tz>Z|[+-]\d{2}'?\d{2}'?)?"
)


def extract_pdf_text(path: Path) -> list[PageText]:
    doc = fitz.open(path)
    pages: list[PageText] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        pages.append(PageText(page_number=i + 1, text=text))
    doc.close()
    return pages


def parse_pdf_datetime(value: str | None) -> datetime | None:
    if not value:
        return None

    value = value.strip()
    match = PDF_DATE_RE.match(value)
    if match:
        parts = match.groupdict()
        tz_raw = parts.get("tz")
        tzinfo = timezone.utc
        if tz_raw and tz_raw != "Z":
            sign = 1 if tz_raw.startswith("+") else -1
            digits = re.sub(r"[^0-9]", "", tz_raw)
            if len(digits) >= 4:
                hours = int(digits[:2])
                minutes = int(digits[2:4])
                tzinfo = timezone(sign * timedelta(hours=hours, minutes=minutes))

        try:
            return datetime(
                year=int(parts["year"]),
                month=int(parts.get("month") or 1),
                day=int(parts.get("day") or 1),
                hour=int(parts.get("hour") or 0),
                minute=int(parts.get("minute") or 0),
                second=int(parts.get("second") or 0),
                tzinfo=tzinfo,
            ).astimezone(timezone.utc)
        except ValueError:
            return None

    for fmt in [
        "%A, %B %d, %Y %I:%M:%S %p",
        "%B %d, %Y %I:%M:%S %p",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d",
    ]:
        try:
            parsed = datetime.strptime(value, fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            continue

    return None


def extract_pdf_metadata(path: Path) -> dict[str, str]:
    doc = fitz.open(path)
    try:
        metadata = doc.metadata or {}
    finally:
        doc.close()

    modified = parse_pdf_datetime(metadata.get("modDate"))
    created = parse_pdf_datetime(metadata.get("creationDate"))
    last_updated = modified or created

    result: dict[str, str] = {}
    for source_key, target_key in [
        ("title", "title"),
        ("author", "author"),
        ("subject", "subject"),
        ("keywords", "keywords"),
    ]:
        value = str(metadata.get(source_key) or "").strip()
        if value:
            result[target_key] = value

    if created:
        result["created_at"] = created.isoformat()
    if modified:
        result["modified_at"] = modified.isoformat()
    if last_updated:
        result["last_updated"] = last_updated.isoformat()

    return result


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
