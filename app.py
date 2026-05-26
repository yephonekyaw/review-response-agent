"""Gradio UI for the reviewer-response agent.

Run:
    uv run python app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import gradio as gr

from response_agent.agents import (
    CommentResult,
    adversarial_followup,
    critique,
    draft_response,
    parse_reviews,
)
from response_agent.assemble import render_letter
from response_agent.chunking import chunk_text
from response_agent.export import markdown_to_pdf
from response_agent.ingest import load_manuscript, load_reviews
from response_agent.llm import get_provider
from response_agent.retrieval import build_index, retrieve
from response_agent.verifier import verify_citations


def _status(text: str) -> str:
    return text


def _render_card(
    idx: int,
    total: int,
    c: dict,
    draft: str,
    verdict: dict,
    final: str,
    refined: bool,
    followup: str = "",
) -> str:
    badge = "refined" if refined else "ok"
    if followup:
        badge += " · adversarial"
    issues = verdict.get("issues") or "—"
    follow_md = f"\n\n**Anticipated follow-up:** {followup}\n" if followup else ""
    return (
        f"**{idx}/{total} · {c['reviewer']}** · `{badge}`\n\n"
        f"> {c['comment']}\n\n"
        f"{final}\n"
        f"{follow_md}\n"
        f"<sub>Critic: ok={verdict.get('ok')} · {issues}</sub>\n\n"
        f"---\n\n"
    )


def run_pipeline(
    manuscript_file: str | None,
    reviews_file: str | None,
    reviews_text: str,
    top_k: int,
    refine_passes: int,
    adversarial: bool,
) -> Generator[tuple, None, None]:
    if not manuscript_file:
        yield "Upload a manuscript first.", "", "", None
        return
    if not reviews_file and not reviews_text.strip():
        yield "Upload a reviews file or paste the text.", "", "", None
        return

    yield "Loading inputs…", "", "", None
    manuscript = load_manuscript(Path(manuscript_file))
    reviews = load_reviews(Path(reviews_file)) if reviews_file else reviews_text

    yield "Chunking + embedding manuscript…", "", "", None
    llm = get_provider()
    chunks = chunk_text(manuscript)
    index = build_index(chunks, llm)

    yield f"{len(chunks)} chunks indexed. Parsing reviews…", "", "", None
    comments = parse_reviews(llm, reviews)

    trace = ""
    results: list[CommentResult] = []
    total = len(comments)
    for i, c in enumerate(comments, 1):
        comment = c["comment"]
        reviewer = c.get("reviewer", f"Reviewer {i}")

        yield f"[{i}/{total}] {reviewer} — drafting…", trace, "", None
        ctx_chunks = retrieve(index, comment, llm, k=top_k)
        context = "\n\n---\n\n".join(ctx_chunks)
        draft = draft_response(llm, comment, context)

        yield f"[{i}/{total}] {reviewer} — critiquing…", trace, "", None
        verdict = critique(llm, comment, draft, context)
        cite_issues = verify_citations(draft, manuscript)
        if cite_issues:
            verdict = {
                "ok": False,
                "issues": " ".join(
                    filter(None, [verdict.get("issues", ""), cite_issues])
                ),
            }

        refined = False
        passes = 0
        final = draft
        while not verdict.get("ok", False) and passes < refine_passes:
            yield f"[{i}/{total}] {reviewer} — refining (pass {passes + 1})…", trace, "", None
            final = draft_response(
                llm, comment, context, refine_note=verdict.get("issues", "")
            )
            verdict = critique(llm, comment, final, context)
            cite_issues = verify_citations(final, manuscript)
            if cite_issues:
                verdict = {
                    "ok": False,
                    "issues": " ".join(
                        filter(None, [verdict.get("issues", ""), cite_issues])
                    ),
                }
            refined = True
            passes += 1

        followup = ""
        if adversarial:
            yield f"[{i}/{total}] {reviewer} — adversarial review…", trace, "", None
            followup = adversarial_followup(llm, comment, final, context)

        results.append(
            CommentResult(
                reviewer=reviewer,
                comment=comment,
                response=final,
                refined=refined,
                critique=verdict,
                adversarial=followup,
            )
        )
        trace += _render_card(i, total, c, draft, verdict, final, refined, followup)
        yield f"[{i}/{total}] {reviewer} — done.", trace, "", None

    letter = render_letter(results)
    out_dir = Path(tempfile.gettempdir())
    md_path = out_dir / "rebuttal.md"
    pdf_path = out_dir / "rebuttal.pdf"
    md_path.write_text(letter, encoding="utf-8")
    markdown_to_pdf(letter, pdf_path)

    n_refined = sum(r.refined for r in results)
    yield (
        f"Done — {len(results)} responses, {n_refined} refined.",
        trace,
        letter,
        [str(md_path), str(pdf_path)],
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("## Reviewer-Response Agent")
        gr.Markdown(
            "Drafts a point-by-point rebuttal from a manuscript and its reviewer comments."
        )

        with gr.Row():
            manuscript_in = gr.File(
                label="Manuscript (PDF or TXT)",
                file_types=[".pdf", ".txt"],
                type="filepath",
            )
            reviews_file_in = gr.File(
                label="Reviews (TXT)", file_types=[".txt"], type="filepath"
            )

        reviews_text_in = gr.Textbox(
            label="Or paste reviewer comments", lines=4, placeholder="Reviewer 1: …"
        )

        with gr.Row():
            top_k = gr.Slider(1, 8, value=4, step=1, label="Top-k chunks")
            refine_passes = gr.Slider(0, 3, value=1, step=1, label="Max refine passes")
            adversarial = gr.Checkbox(value=False, label="Adversarial follow-up")
            run_btn = gr.Button("Run", variant="primary")

        status = gr.Textbox(label="Status", interactive=False, value="Idle.")

        with gr.Tab("Trace"):
            trace_out = gr.Markdown()
        with gr.Tab("Rebuttal letter"):
            letter_file = gr.Files(label="Download (Markdown + PDF)", interactive=False)
            letter_md = gr.Markdown()

        run_btn.click(
            run_pipeline,
            inputs=[
                manuscript_in,
                reviews_file_in,
                reviews_text_in,
                top_k,
                refine_passes,
                adversarial,
            ],
            outputs=[status, trace_out, letter_md, letter_file],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch()
