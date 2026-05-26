"""Citation-grounding verifier.

The DRAFT prompt asks the Author to point to a manuscript section (e.g.
"see Section 3.2"). Nothing stops it from fabricating one. This module
extracts those pointers from a draft response and checks they appear in
the manuscript text. Any unverified pointer is returned as an issue
string the existing refine loop can act on.
"""
from __future__ import annotations

import re

# Matches "Section 3.2", "Figure 1", "Table 4", "Appendix A", "Equation (2)", etc.
# Captures the kind (Section/Figure/Table/Appendix/Equation/Eq) and the identifier.
_CITATION_RE = re.compile(
    r"\b(Section|Sections|Figure|Figures|Fig\.?|Table|Tables|Appendix|Appendices|Equation|Eq\.?)\s+"
    r"([A-Z]?\d+(?:\.\d+)*[a-z]?)",
    re.IGNORECASE,
)


def extract_citations(text: str) -> list[tuple[str, str]]:
    """Return (kind, id) tuples found in `text`, deduplicated, preserving order."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for m in _CITATION_RE.finditer(text):
        kind = m.group(1).rstrip(".").lower()
        # Normalize short forms so "Fig" → "figure", "Eq" → "equation".
        kind = {"fig": "figure", "eq": "equation"}.get(kind, kind)
        # Singularize so "figures" / "sections" / "tables" all collapse.
        if kind.endswith("s") and kind not in {"appendices"}:
            kind = kind[:-1]
        if kind == "appendices":
            kind = "appendix"
        ident = m.group(2)
        key = (kind, ident)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _manuscript_has(manuscript: str, kind: str, ident: str) -> bool:
    """Loose substring check, case-insensitive, allowing common short forms."""
    haystack = manuscript.lower()
    candidates = {kind, {"figure": "fig", "equation": "eq"}.get(kind, kind)}
    return any(f"{c} {ident.lower()}" in haystack for c in candidates)


def verify_citations(response: str, manuscript: str) -> str:
    """Return an empty string if all citations check out, else a single issue line."""
    cites = extract_citations(response)
    if not cites:
        return ""
    missing = [f"{kind.capitalize()} {ident}"
               for kind, ident in cites if not _manuscript_has(manuscript, kind, ident)]
    if not missing:
        return ""
    return ("The response references manuscript locations that do not appear in the manuscript: "
            f"{', '.join(missing)}. Either cite a location that exists or remove the pointer.")
