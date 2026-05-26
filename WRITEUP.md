# Reviewer-Response Agent

*CSC532/691 Machine Learning — Final Hackathon*

> **Free-resource declaration.** This project uses only free AI/cloud
> resources available to all SIT students: the free tier of the Google
> Gemini API (`gemini-3.1-flash-lite` for chat, `gemini-embedding-001` for
> embeddings). A local Ollama backend is supported as a drop-in fallback
> when the free quota is exhausted — also free, running on the author's
> home server. No paid services were used.

> **🔗 Live demo:** <PUBLIC_GRADIO_URL_HERE>
> *(Gradio share link; valid for ~72 hours from launch.)*

---

## 1. The problem

When a paper comes back from peer review, the authors must produce a
*point-by-point rebuttal* — a structured letter that quotes every reviewer
comment and answers it. This is slow, repetitive, and high-stakes work:
an unanswered concern is the single most common reason a revision is
rejected.

We built an AI agent that takes a manuscript PDF and the raw reviewer
text, then produces a complete Markdown / PDF rebuttal letter grouped by
reviewer, with every comment answered in a consistent professional voice
and grounded in the manuscript itself.

## 2. Why this is an *agent* and not a single prompt

A naive "here is a comment, write a response" prompt produces drafts that
sound fluent but routinely (a) miss the actual concern, (b) invent
experiments and sections that don't exist, or (c) sound defensive. We
address this with a real **feedback loop** built from five distinct LLM
roles plus one deterministic verifier:

```
                       ┌──────────── Citation Verifier (regex) ────────────┐
                       │                                                   ▼
PARSE  ──►  DRAFT  ──►  CRITIQUE  ──►  (merge verdicts)  ──►  ok? ──► FINAL
   │           ▲                                                │
   │           └──────────── REFINE (≤ N passes) ◄──────────────┘
   │
   └─►  ADVERSARIAL REVIEWER  (diagnostic; what a hostile reviewer would say next)
```

The five LLM roles:

| Role | What it does | Output |
|---|---|---|
| **PARSE** | Splits the raw reviewer text into atomic, individually-addressable points and labels them by reviewer. | JSON `[{reviewer, comment}]` |
| **DRAFT / Author** | Writes a response to one comment, grounded in the top-k retrieved manuscript chunks. | Plain text |
| **CRITIQUE / Editor** | Checks the draft on 4 criteria: directly addresses the concern, grounded in manuscript, professional tone, states a concrete action. | JSON `{ok, issues}` |
| **REFINE** | If the Critic flags issues, the Author rewrites once with that feedback. | Plain text |
| **ADVERSARIAL REVIEWER** *(opt-in)* | Given the final response, writes the sharpest follow-up a hostile reviewer would send in round 2. | Plain text |

The **Citation Verifier** is a deterministic safety net, not an LLM call:
it regex-extracts every `Section X.Y`, `Figure N`, `Table N`, `Appendix
X`, `Equation N` reference from the draft and substring-checks the
manuscript. Missing pointers are merged into the Critic's verdict, so
hallucinated section references trigger the same refine loop without any
new control flow. This is the kind of cheap, high-precision check that
LLM-only pipelines tend to leave on the table.

## 3. Retrieval-augmented generation (RAG)

For a single manuscript the easy thing is to paste the whole paper into
every prompt. We do something a little more honest:

1. **Chunk** the manuscript with a character window (1800 chars / 200
   overlap).
2. **Embed** each chunk with `gemini-embedding-001` (768-dim).
3. **Index** in-memory: an L2-normalized numpy matrix.
4. For each comment, embed the comment and retrieve the top-k chunks by
   cosine similarity. Pass only those chunks into DRAFT / CRITIQUE.

Why this matters: tokens are cheap on Gemini's free tier, but they are
not infinite — and a real workflow with a book-length manuscript or a
multi-paper corpus would break the "paste everything" approach. The RAG
layer is the same ~20 lines either way, so we built it correctly from
the start. It's also what lets the Citation Verifier produce useful
issue messages — the Critic and Author both see the same retrieved
context, so an unverified `Section 99` is detectable as soon as it
appears.

## 4. Provider abstraction — Gemini today, Ollama tomorrow

A common failure mode of student projects is "works on my machine until
the API quota runs out at 11pm." We avoid that with a clean provider
abstraction:

```python
class LLMProvider(Protocol):
    def generate(prompt, *, json_mode, temperature) -> str: ...
    def embed(texts: list[str]) -> np.ndarray: ...
```

Two concrete implementations behind the protocol: **`GeminiProvider`**
(default, free tier) and **`OllamaProvider`** (any local Ollama server,
configurable host + chat/embed model names). Switching is a single env
var, `LLM_PROVIDER=ollama`. No code changes anywhere in the pipeline.

We also built the free-tier rate-limit handling directly into the Gemini
provider:

- A configurable **minimum interval** between calls (default 6.5s — the
  free tier is 10 RPM).
- On 429 errors, the provider parses the server-suggested `retryDelay`
  from the error message (e.g. `"please retry in 40.8s"`) and sleeps for
  exactly that long plus a 1s buffer, then retries automatically.

Result: the pipeline runs to completion on the free tier without manual
intervention even when quota is tight.

## 5. Engineering choices, in plain language

| Choice | Why |
|---|---|
| **`uv` for dependency management** | Reproducible, fast, lockfile-based. `uv sync` and you're done. |
| **`google-genai` + `ollama` Python SDKs** | Both first-party. Provider layer keeps the rest of the code blind to which one is in use. |
| **`fpdf2` for PDF export** | Pure Python, no system deps (no Cairo, no wkhtmltopdf). Markdown → HTML via the `markdown` lib → PDF via `fpdf2.write_html`. |
| **Prompts use `<<PLACEHOLDER>>` markers** | `str.format` chokes on literal `{}` in the prompt body (e.g. JSON schema examples in the PARSE / CRITIQUE prompts). Plain `.replace()` is bulletproof. |
| **JSON response mode** for PARSE + CRITIQUE | `response_mime_type="application/json"` removes all output-parsing fragility — no regex scraping, no fenced-code stripping. |
| **Streaming Gradio UI** | The pipeline is a generator that yields after every sub-step (retrieve / draft / critique / refine). The UI never looks frozen, even during the throttle's 6.5s gaps. |

## 6. Deliverables

The repo ships with two ways to drive the pipeline and two output
formats per run:

- **CLI:** `uv run python main.py [--manuscript ... --reviews ...
  --top-k 4 --refine-passes 1 --adversarial]`
- **Web UI:** `uv run python app.py` — Gradio on `localhost:7860`, with a
  live per-comment trace tab (reviewer text → final response → critic
  verdict line) and a downloads tab for both `rebuttal.md` and
  `rebuttal.pdf`. **Hosted publicly at <PUBLIC_GRADIO_URL_HERE>.**
- **Notebook:** `notebook.ipynb` — the original Kaggle-style single-file
  version, kept as a self-contained reference.

## 7. Limitations and honest future work

- **Critic has no ground truth.** A confident-but-wrong Critic can
  rubber-stamp a bad draft. A second adversarial critic or human-in-the-
  loop step would harden this. (The Adversarial Reviewer hints at this
  direction but doesn't act on it.)
- **Refine is capped at 1 pass.** Harder comments might benefit from 2;
  we kept it at 1 to bound latency on the free tier.
- **No citation lookup.** When a reviewer asks "please cite X", the
  agent can describe adding a citation but cannot resolve it. A small
  bib-database tool would close this gap.
- **Single language.** Prompts are English-only.

## 8. Reproducing this

```fish
git clone <repo>
cd response-agent
uv sync
cp .env.example .env   # set GEMINI_API_KEY
# put your inputs in data/manuscript.pdf and data/reviews.txt
uv run python app.py   # opens the Gradio UI at localhost:7860
# or, for the CLI:
uv run python main.py --adversarial
```

## 9. Free-resource declaration

This project uses only free resources available to all SIT students: the
free Google Gemini API and (optionally) a local Ollama server. No paid
LLM endpoints, no paid cloud GPUs, no private data sources.
