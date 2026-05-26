"""Load manuscript (PDF or txt) and reviews (txt) into plain strings."""
from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def load_manuscript(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def load_reviews(path: Path) -> str:
    return path.read_text(encoding="utf-8")
