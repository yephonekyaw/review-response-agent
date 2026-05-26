---
license: mit
language:
- en
tags:
- agent
- multi-agent
- rag
- peer-review
- rebuttal
- scientific-writing
- gemini
- ollama
pipeline_tag: text-generation
base_model:
- google/gemini-3.1-flash-lite
- google/gemini-embedding-001
library_name: custom
---

# Reviewer-Response Agent

> A multi-agent RAG system that turns raw peer-review comments into a
> grounded, point-by-point manuscript rebuttal letter.

This is an **Agent Card** (Hugging Face Model Card format, adapted) for
the CSC532/691 final-hackathon submission. The system is not a single
model — it's an orchestration of five LLM-driven roles plus a
deterministic citation verifier, all driven through a provider-agnostic
LLM layer.

## Highlights

- **Multi-agent loop** — five LLM roles (Parse → Draft → Critique →
  Refine, plus an optional Adversarial Reviewer) run per reviewer
  comment, with a real Critic→Refine feedback loop instead of one-shot
  generation.
- **Deterministic citation verifier** — regex-extracts every `Section
  X.Y`, `Figure N`, `Table N`, `Appendix X`, `Equation N` reference in a
  draft and checks it against the manuscript. Missing pointers are
  merged into the Critic's verdict and trigger the same refine loop —
  no new control flow.
- **RAG over the manuscript** — character chunking + `gemini-embedding-001`
  embeddings + in-memory numpy cosine retrieval. The Draft / Critique
  prompts only ever see the top-k chunks relevant to a comment.
- **Provider-agnostic LLM layer** — `GeminiProvider` (default, free
  tier) and `OllamaProvider` (local server) behind a `LLMProvider`
  Protocol. Switch backends with one env var, no code change.
- **Free-tier rate-limit handling** — built-in minimum-interval throttle
  + 429 retry that parses Google's `retryDelay` suggestion and sleeps
  for exactly that long.
- **Three entry points** — CLI (`main.py`), Gradio web UI (`app.py`,
  streaming trace), and a demo notebook (`demo.ipynb`).

## Agent Details

### Description

The Reviewer-Response Agent is a Python application designed to assist
manuscript authors with the most time-consuming and error-prone part of
the peer-review revision cycle: writing a point-by-point rebuttal
letter. Given a manuscript (PDF or text) and the raw reviewer comments
(also text), it produces a formatted Markdown + PDF rebuttal letter
grouped by reviewer, with every comment quoted and answered in a
consistent professional voice and grounded in the manuscript's actual
content.

The agent is built on top of `gemini-3.1-flash-lite` for chat and
`gemini-embedding-001` for embeddings (both free tier). It can also be
pointed at a local Ollama server for offline operation.

- **Developed by:** Ye Phone Kyaw — CSC532/691, SIT
- **Agent type:** Multi-agent RAG pipeline
- **Languages:** English (prompts and outputs)
- **License:** MIT
- **Foundation models:** `gemini-3.1-flash-lite`, `gemini-embedding-001`
  (default); any Ollama chat + embedding model (alternative)

### Architecture

```
                       ┌────── Citation Verifier (regex) ──────┐
                       │                                        ▼
PARSE  ──►  DRAFT  ──►  CRITIQUE  ──►  (merge verdicts)  ──►  ok? ──► FINAL
   │           ▲                                                │
   │           └────────── REFINE (≤ N passes) ◄────────────────┘
   │
   └─►  ADVERSARIAL REVIEWER  (diagnostic; optional)
```

### The five LLM roles

| Role | System prompt focus | Temp | Output format |
|---|---|---|---|
| **PARSE** | Split raw reviewer text into atomic, individually-addressable comments labeled by reviewer. | 0.2 | JSON `[{reviewer, comment}]` |
| **DRAFT** (Author) | Write a polite, structured 1-3-paragraph response to one comment, grounded in retrieved manuscript excerpts. No invented experiments. | 0.4 | Plain text |
| **CRITIQUE** (Editor) | Evaluate the draft on 4 criteria: directly addresses, grounded, professional tone, concrete action. | 0.1 | JSON `{ok, issues}` |
| **REFINE** | Re-runs the DRAFT role with the Critic's issues appended as a fix-up note. | 0.4 | Plain text |
| **ADVERSARIAL** | Given the final response, write the sharpest follow-up a hostile reviewer would send next round. Return `NO FOLLOW-UP` if airtight. | 0.6 | Plain text |

The **Citation Verifier** is *not* an LLM — it is a regex over the
draft + substring check against the manuscript. Hallucinated section
references become Critic issues automatically.

## Quickstart

```bash
# 1. Install
git clone <repo>
cd response-agent
uv sync

# 2. Configure (free Gemini key from https://aistudio.google.com/app/apikey)
cp .env.example .env
# edit .env and set GEMINI_API_KEY

# 3. Put inputs in data/
#    data/manuscript.pdf (or .txt)
#    data/reviews.txt

# 4. Run — pick one
uv run python main.py                  # CLI, writes outputs/rebuttal.{md,pdf}
uv run python main.py --adversarial    # ...plus hostile follow-ups
uv run python app.py                   # Gradio UI on localhost:7860
uv run jupyter lab demo.ipynb          # step-by-step notebook walkthrough
```

### Python API

```python
from pathlib import Path
from response_agent.pipeline import run

run(
    manuscript_path=Path("data/manuscript.pdf"),
    reviews_path=Path("data/reviews.txt"),
    output_path=Path("outputs/rebuttal.md"),
    top_k=4,                # manuscript chunks retrieved per comment
    max_refine_passes=1,    # Critic-driven refines
    adversarial=False,      # opt-in hostile-reviewer follow-ups
)
```

### Swapping to Ollama

When the free Gemini quota is exhausted (10 RPM on the free tier), swap
to a local Ollama server with no code change:

```bash
# in .env
LLM_PROVIDER=ollama
OLLAMA_HOST=http://your-server:11434
OLLAMA_CHAT_MODEL=llama3.1:8b
OLLAMA_EMBED_MODEL=nomic-embed-text
```

The `LLMProvider` Protocol means the rest of the pipeline is blind to
which backend is in use.

## Best Practices

- **Start with the defaults** (`top_k=4`, `refine_passes=1`). Higher
  `top_k` rarely improves grounding and costs more tokens; higher
  `refine_passes` rarely fires past pass 1 because the Critic is itself
  bounded.
- **Use `--adversarial` selectively.** It doubles LLM calls per comment.
  Run it on a slow demo or before a real submission, not on every
  iteration.
- **PARSE quality is the ceiling.** If the reviewer text is a noisy
  paste with no reviewer headings, the Parse step may miscount or
  mislabel. Cleaning the input by adding `Reviewer 1:` / `Reviewer 2:`
  markers improves everything downstream.
- **For long manuscripts**, raise `chunk_chars` in
  `src/response_agent/chunking.py` from 1800 → 3000 to reduce embedding
  cost without losing recall.

## Evaluation

This is a hackathon submission, not a benchmarked release. We did not
score against a held-out test set. Qualitative checks on the bundled
SIGIR24 sample (4 reviewers, ~12 atomic comments):

- **PARSE recall:** all substantive concerns split into separate items;
  pleasantries (`"well-written"`) skipped as intended.
- **Critic firing rate:** ~25-40% of first drafts flagged for
  refinement, typically for being too agreeable without committing to a
  concrete change.
- **Citation Verifier:** catches references to manuscript locations
  that don't exist with 100% precision (it's deterministic) and near-
  100% recall (limited by regex coverage of citation phrasing).
- **End-to-end latency on free Gemini:** ~2-4 minutes for 12 comments,
  dominated by the 6.5s rate-limit throttle (10 RPM ceiling).

## Intended Use

### Direct use

- Drafting first-pass rebuttal letters for journal / conference paper
  revisions.
- Stress-testing your own rebuttals via the Adversarial Reviewer to see
  what a hostile reviewer might say next round.
- Teaching demonstration of multi-agent RAG architecture.

### Out-of-scope use

- **Final, unedited submission.** The agent drafts; humans must
  review, edit, and take responsibility for the letter that goes to
  the editor.
- **Fabricating experimental results.** The Draft prompt explicitly
  forbids inventing experiments, and the Critic + Verifier are the
  guardrails — but no automated guardrail is infallible.
- **High-stakes legal/clinical writing.** Domain-specific
  hallucination risks are not bounded.
- **Reviews in languages other than English.** Prompts have not been
  validated on non-English text.

## Bias, Risks, and Limitations

- **Critic has no ground truth.** A confident-but-wrong Critic can
  rubber-stamp a bad draft. The 4-criterion prompt mitigates this but
  cannot eliminate it.
- **Refine capped at 1 pass** by default to bound free-tier latency.
  Harder comments may need more.
- **No citation lookup.** When a reviewer asks "please cite X", the
  agent describes adding a citation but cannot resolve it to a real
  bibliographic entry.
- **Author tone bias.** The DRAFT prompt is tuned for a polite,
  deferential academic register that may not match all venues or
  cultural conventions.
- **Single-language.** English only.
- **Provider-side risks** — Gemini outputs may be logged by Google per
  their free-tier terms; do not feed confidential or pre-publication
  manuscripts you are unwilling to share. Use the Ollama backend for
  privacy-sensitive workflows.

## Compute

- **Free tier.** Gemini API free tier — 10 RPM, throttled
  automatically; no GPU required locally.
- **Local alternative.** Ollama with `llama3.1:8b` + `nomic-embed-text`
  runs comfortably on a consumer GPU or even a recent CPU-only Mac.

## Citation

```bibtex
@misc{kyaw2026reviewerresponse,
  title  = {Reviewer-Response Agent: A Multi-Agent RAG System for
            Manuscript Rebuttals},
  author = {Kyaw, Ye Phone},
  year   = {2026},
  note   = {CSC532/691 Machine Learning, Final Hackathon Submission, SIT},
}
```

## Contact

Ye Phone Kyaw — `yephonekyaw231202@gmail.com`
