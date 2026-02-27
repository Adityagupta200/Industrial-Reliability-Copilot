from __future__ import annotations

import re

WS_RE = re.compile(r"[ \t]+")


def clean_text(text: str) -> str:
    # Normalize whitespace, remove obvious noise
    text = text.replace("\x00", " ")
    text = WS_RE.sub(" ", text)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
