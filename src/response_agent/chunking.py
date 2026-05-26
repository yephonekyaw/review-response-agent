"""Character-based chunker with overlap. Simple and good enough for one paper."""
from __future__ import annotations


def chunk_text(text: str, *, chunk_chars: int = 1800, overlap: int = 200) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    step = max(1, chunk_chars - overlap)
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_chars]
        if chunk.strip():
            chunks.append(chunk)
        if start + chunk_chars >= len(text):
            break
    return chunks
