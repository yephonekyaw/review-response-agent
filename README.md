# Reviewer-Response Agent

Drafts a point-by-point manuscript rebuttal from a PDF + raw reviewer comments.
Uses Gemini by default; can swap to a local Ollama model via env var.

## Setup

```fish
# 1. Install deps (uv-managed)
uv sync

# 2. Configure
cp .env.example .env
#   then edit .env and set GEMINI_API_KEY (free: https://aistudio.google.com/app/apikey)

# 3. Put your inputs in data/
#   data/manuscript.pdf
#   data/reviews.txt
```

## Run

```fish
uv run python main.py
# or with explicit paths:
uv run python main.py --manuscript data/manuscript.pdf --reviews data/reviews.txt --out outputs/rebuttal.md
```

Output lands in `outputs/rebuttal.md`.

## Switching to Ollama

When the Gemini free tier runs out, switch backends with a single env change:

```fish
# in .env
LLM_PROVIDER=ollama
OLLAMA_HOST=http://your-home-server:11434
OLLAMA_CHAT_MODEL=llama3.1:8b        # any chat model you've pulled
OLLAMA_EMBED_MODEL=nomic-embed-text  # any embedding model you've pulled
```

No code changes needed — the LLM layer is provider-agnostic.

## Architecture

```
load PDF + reviews
   │
   ▼
chunk manuscript ──► embed chunks ──► in-memory cosine index
   │
   ▼
PARSE  reviewer text → atomic comments (JSON)
   │
   ▼  (per comment)
retrieve top-k chunks ──► DRAFT ──► CRITIQUE ──► REFINE? (≤1 pass)
   │
   ▼
ASSEMBLE  →  outputs/rebuttal.md
```

## File layout

```
src/response_agent/
├── config.py       # env loading, provider selection
├── llm.py          # GeminiProvider / OllamaProvider abstraction
├── ingest.py       # PDF + text loading
├── chunking.py     # character-based chunker w/ overlap
├── retrieval.py    # embed + cosine top-k
├── agents.py       # PARSE / DRAFT / CRITIQUE prompts + functions
├── assemble.py     # render Markdown letter
└── pipeline.py     # orchestration
main.py             # CLI
notebook.ipynb      # kept as the Kaggle submission deliverable
```
