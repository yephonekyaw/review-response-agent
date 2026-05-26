"""Render the final rebuttal letter as Markdown."""
from __future__ import annotations

from collections import defaultdict

from .agents import CommentResult


def render_letter(results: list[CommentResult]) -> str:
    by_reviewer: dict[str, list[CommentResult]] = defaultdict(list)
    for r in results:
        by_reviewer[r.reviewer].append(r)

    lines = [
        "# Response to Reviewers",
        "",
        "We thank the reviewers for their thoughtful and constructive feedback. "
        "Below we address each comment in turn.",
        "",
    ]
    for reviewer in sorted(by_reviewer):
        lines.append("---")
        lines.append("")
        lines.append(f"## {reviewer}")
        lines.append("")
        for i, r in enumerate(by_reviewer[reviewer], 1):
            lines.append(f"### Comment {i}")
            lines.append("")
            lines.append(f"> {r.comment}")
            lines.append("")
            lines.append(f"**Response.** {r.response}")
            lines.append("")
    return "\n".join(lines)
