"""Markdown → PDF using fpdf2 (pure Python, no system deps).

We convert markdown to HTML with the stdlib-style `markdown` lib, then hand
it to fpdf2's `write_html` which understands a useful subset (h1-h6, p,
b, i, ul, ol, blockquote, hr).
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import markdown as md
from fpdf import FPDF


# Smart-quote / dash / bullet replacements so fpdf2's Latin-1 Helvetica
# doesn't choke on LLM output. Anything left over after this map is
# pushed through NFKD + ASCII strip.
_UNICODE_FIXUPS = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "−": "-",
    "…": "...",
    "•": "*", "·": "*",
    " ": " ",
    "→": "->", "←": "<-", "⇒": "=>", "⇐": "<=",
    "×": "x",
})


def _sanitize_latin1(text: str) -> str:
    """Make `text` safe for fpdf2's default Helvetica font."""
    text = text.translate(_UNICODE_FIXUPS)
    # Decompose accents → ASCII (é → e), then drop anything still non-Latin-1.
    text = unicodedata.normalize("NFKD", text)
    return text.encode("latin-1", errors="ignore").decode("latin-1")


_CSS = """
<style>
  h1 { font-size: 20pt; }
  h2 { font-size: 14pt; }
  h3 { font-size: 12pt; }
  blockquote { color: #555; font-style: italic; }
  hr { color: #ccc; }
  p, li { font-size: 11pt; }
</style>
"""


def markdown_to_pdf(md_text: str, out_path: Path) -> Path:
    html = md.markdown(_sanitize_latin1(md_text), extensions=["extra"])
    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.write_html(_CSS + html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))
    return out_path
