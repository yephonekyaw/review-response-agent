"""End-to-end orchestration: inputs in, rebuttal letter out."""
from __future__ import annotations

from pathlib import Path

from .agents import CommentResult, adversarial_followup, critique, draft_response, parse_reviews
from .assemble import render_letter
from .chunking import chunk_text
from .export import markdown_to_pdf
from .ingest import load_manuscript, load_reviews
from .llm import LLMProvider, get_provider
from .retrieval import build_index, retrieve
from .verifier import verify_citations


def run(
    manuscript_path: Path,
    reviews_path: Path,
    output_path: Path,
    *,
    llm: LLMProvider | None = None,
    top_k: int = 4,
    max_refine_passes: int = 1,
    adversarial: bool = False,
) -> Path:
    llm = llm or get_provider()

    print(f"[1/5] Loading manuscript: {manuscript_path}")
    manuscript = load_manuscript(manuscript_path)
    reviews_raw = load_reviews(reviews_path)
    print(f"      manuscript: {len(manuscript):,} chars   reviews: {len(reviews_raw):,} chars")

    print("[2/5] Chunking + embedding manuscript")
    chunks = chunk_text(manuscript)
    index = build_index(chunks, llm)
    print(f"      {len(chunks)} chunks indexed (dim={index.embeddings.shape[1]})")

    print("[3/5] Parsing reviews into atomic comments")
    comments = parse_reviews(llm, reviews_raw)
    print(f"      {len(comments)} comments extracted")

    print("[4/5] Drafting + critiquing each comment")
    results: list[CommentResult] = []
    for i, c in enumerate(comments, 1):
        comment = c["comment"]
        reviewer = c.get("reviewer", f"Reviewer {i}")
        context_chunks = retrieve(index, comment, llm, k=top_k)
        context = "\n\n---\n\n".join(context_chunks)

        draft = draft_response(llm, comment, context)
        verdict = critique(llm, comment, draft, context)
        cite_issues = verify_citations(draft, manuscript)
        if cite_issues:
            verdict = {"ok": False, "issues": " ".join(
                filter(None, [verdict.get("issues", ""), cite_issues]))}

        refined = False
        passes = 0
        while not verdict.get("ok", False) and passes < max_refine_passes:
            draft = draft_response(llm, comment, context, refine_note=verdict.get("issues", ""))
            verdict = critique(llm, comment, draft, context)
            cite_issues = verify_citations(draft, manuscript)
            if cite_issues:
                verdict = {"ok": False, "issues": " ".join(
                    filter(None, [verdict.get("issues", ""), cite_issues]))}
            refined = True
            passes += 1

        followup = ""
        if adversarial:
            followup = adversarial_followup(llm, comment, draft, context)

        tag = f"refined x{passes}" if refined else "ok first pass"
        if followup:
            tag += " · adversarial flagged"
        print(f"      [{i:>2}/{len(comments)}] {reviewer} — {tag}")
        results.append(CommentResult(
            reviewer=reviewer, comment=comment, response=draft,
            refined=refined, critique=verdict, adversarial=followup,
        ))

    print(f"[5/5] Writing letter to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    letter = render_letter(results)
    output_path.write_text(letter, encoding="utf-8")
    pdf_path = output_path.with_suffix(".pdf")
    markdown_to_pdf(letter, pdf_path)
    print(f"      PDF: {pdf_path}")
    return output_path
