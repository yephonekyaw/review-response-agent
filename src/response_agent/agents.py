"""The four LLM-driven roles: PARSE, DRAFT, CRITIQUE, REFINE.

Prompts use <<PLACEHOLDER>> markers (not str.format) so that literal { } in the
prompt body — e.g. JSON schema examples — don't break string templating.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .llm import LLMProvider


PARSE_PROMPT = """You are preparing a manuscript rebuttal. Read the raw reviewer comments below and split them into atomic, individually-addressable points.

Rules:
- One concrete concern per item. Do NOT merge multiple concerns.
- Preserve the reviewer's wording where possible; lightly paraphrase only if needed for clarity.
- Skip pure pleasantries ("nice paper", "well written") unless they carry an actionable point.
- Identify the reviewer (e.g., "Reviewer 1", "Reviewer 2") from headings if present; otherwise label them sequentially.

Return a JSON array of objects with this shape:
[{"reviewer": "Reviewer 1", "comment": "..."}, ...]

RAW REVIEWS:
---
<<REVIEWS>>
---"""


DRAFT_PROMPT = """You are the author of the manuscript, writing a point-by-point rebuttal to a reviewer.

Write a response to the SINGLE reviewer comment provided. Use this structure (1-3 short paragraphs total):
1. Acknowledge the reviewer's point politely.
2. Clarify, agree, or describe the change you will make. Be specific and grounded in the manuscript excerpts below. Do NOT invent experiments, numbers, or citations not present in the manuscript.
3. Point to the relevant section of the manuscript (e.g., "see Section 3.2").

Tone: professional, respectful, concrete. Avoid filler.
<<REFINE_NOTE>>
RELEVANT MANUSCRIPT EXCERPTS:
---
<<CONTEXT>>
---

REVIEWER COMMENT:
<<COMMENT>>

Your response:"""


ADVERSARIAL_PROMPT = """You are a hostile, skeptical peer reviewer reading the author's rebuttal below.

Your job: identify the SINGLE weakest point in the response — an unsupported claim, an unaddressed sub-question, a hand-wave, or a promise without a concrete commitment — and write the short, pointed follow-up comment you would send in the next review round.

Constraints:
- One short paragraph (2-4 sentences). No preamble, no "Dear authors".
- Be specific. Reference the exact claim or omission. Do NOT just say "needs more detail".
- Tone: firm but professional. Not rude.
- If the response is genuinely airtight, reply with exactly: NO FOLLOW-UP.

ORIGINAL REVIEWER COMMENT:
<<COMMENT>>

AUTHOR RESPONSE:
<<RESPONSE>>

MANUSCRIPT EXCERPTS (ground truth):
---
<<CONTEXT>>
---

Your follow-up comment:"""


CRITIQUE_PROMPT = """You are a strict editor checking a rebuttal response before it is sent to the reviewer.

Evaluate the response against the comment on FOUR criteria:
(a) Does it directly address the specific concern raised?
(b) Is it grounded in the manuscript excerpts (no invented experiments, numbers, or citations)?
(c) Is the tone professional and non-defensive?
(d) Does it state a concrete action or clarification, not just agreement?

Return a JSON object with this exact shape:
{"ok": true, "issues": ""}
Set ok=true only if all four criteria pass. If ok=false, the issues string must be specific and actionable.

MANUSCRIPT EXCERPTS (ground truth):
---
<<CONTEXT>>
---

REVIEWER COMMENT:
<<COMMENT>>

DRAFT RESPONSE:
<<DRAFT>>"""


def _fill(template: str, **kwargs: str) -> str:
    out = template
    for key, value in kwargs.items():
        out = out.replace(f"<<{key}>>", value)
    return out


def parse_reviews(llm: LLMProvider, raw: str) -> list[dict]:
    prompt = _fill(PARSE_PROMPT, REVIEWS=raw)
    return json.loads(llm.generate(prompt, json_mode=True, temperature=0.2))


def draft_response(llm: LLMProvider, comment: str, context: str, *, refine_note: str = "") -> str:
    note = f"\nA previous draft had these issues; fix them: {refine_note}\n" if refine_note else ""
    prompt = _fill(DRAFT_PROMPT, REFINE_NOTE=note, CONTEXT=context, COMMENT=comment)
    return llm.generate(prompt, temperature=0.4)


def critique(llm: LLMProvider, comment: str, draft: str, context: str) -> dict:
    prompt = _fill(CRITIQUE_PROMPT, CONTEXT=context, COMMENT=comment, DRAFT=draft)
    return json.loads(llm.generate(prompt, json_mode=True, temperature=0.1))


def adversarial_followup(llm: LLMProvider, comment: str, response: str, context: str) -> str:
    """Return the next-round comment a hostile reviewer would write, or '' if airtight."""
    prompt = _fill(ADVERSARIAL_PROMPT, COMMENT=comment, RESPONSE=response, CONTEXT=context)
    text = llm.generate(prompt, temperature=0.6).strip()
    if text.upper().startswith("NO FOLLOW-UP"):
        return ""
    return text


@dataclass
class CommentResult:
    reviewer: str
    comment: str
    response: str
    refined: bool
    critique: dict
    adversarial: str = ""
