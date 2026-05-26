# Architecture Diagrams

Five Mermaid diagrams covering different views of the system. Paste any of
these into the [Mermaid Live Editor](https://mermaid.live) to render and
export as PNG/SVG for the Kaggle gallery.

---

## 1. High-level pipeline

End-to-end view: inputs in, rebuttal out. This is the "elevator" diagram.

```mermaid
flowchart LR
    A[Manuscript PDF] --> I[Ingest]
    B[Reviews TXT] --> I
    I --> C[Chunk + Embed]
    C --> X[(ChunkIndex<br/>cosine sim)]
    I --> P[PARSE agent]
    P --> Q[Atomic comments]
    Q --> L[Agent Loop<br/>per comment]
    X --> L
    L --> R[Results]
    R --> AS[Assemble]
    AS --> MD[rebuttal.md]
    AS --> PDF[rebuttal.pdf]

    classDef io fill:#e3f2fd,stroke:#1976d2,color:#000
    classDef agent fill:#fff3e0,stroke:#f57c00,color:#000
    classDef store fill:#f3e5f5,stroke:#7b1fa2,color:#000
    class A,B,MD,PDF io
    class P,L,AS agent
    class X store
```

---

## 2. Agent loop (the heart of the system)

What happens for *each* parsed reviewer comment. This is where the "agent"
label is earned — Draft → Critique → Verify → Refine is a real feedback loop,
not a single prompt.

```mermaid
flowchart TD
    C[Reviewer comment] --> RE[Embed comment]
    RE --> RT[Retrieve top-k chunks<br/>from ChunkIndex]
    RT --> CTX[Context]
    CTX --> D[DRAFT agent]
    C --> D
    D --> DR[Draft response]
    DR --> CR[CRITIQUE agent<br/>4 criteria → JSON]
    CTX --> CR
    DR --> V[Citation Verifier<br/>regex: Section/Figure/Table refs]
    DR --> ADV[Adversarial Reviewer<br/>optional]
    V --> MRG{merge<br/>verdicts}
    CR --> MRG
    MRG -->|ok=true| F[Final response]
    MRG -->|ok=false| RF{refine pass<br/>&lt; max?}
    RF -->|yes| D
    RF -->|no| F
    ADV -.->|follow-up<br/>diagnostic only| F

    classDef llm fill:#fff3e0,stroke:#f57c00,color:#000
    classDef tool fill:#e8f5e9,stroke:#388e3c,color:#000
    classDef gate fill:#fce4ec,stroke:#c2185b,color:#000
    class D,CR,ADV llm
    class V,RE,RT tool
    class MRG,RF gate
```

---

## 3. LLM provider abstraction

How the pipeline stays provider-agnostic. Swap `LLM_PROVIDER=gemini` for
`LLM_PROVIDER=ollama` in `.env`; no code changes.

```mermaid
flowchart LR
    subgraph App[Application code]
        AG[agents.py]
        RT[retrieval.py]
    end

    subgraph Layer[LLM layer]
        IF{{LLMProvider<br/>Protocol<br/>.generate / .embed}}
    end

    subgraph Providers[Concrete providers]
        G[GeminiProvider<br/>+ throttle<br/>+ 429 retry]
        O[OllamaProvider<br/>local HTTP]
    end

    AG --> IF
    RT --> IF
    IF -->|LLM_PROVIDER=gemini| G
    IF -->|LLM_PROVIDER=ollama| O
    G --> GA[Gemini API<br/>2.5-flash<br/>text-embedding-004]
    O --> OL[Ollama server<br/>llama3.1:8b<br/>nomic-embed-text]

    classDef api fill:#e1f5fe,stroke:#0277bd,color:#000
    classDef abs fill:#f3e5f5,stroke:#7b1fa2,color:#000
    class GA,OL api
    class IF abs
```

---

## 4. Module layout

What lives where in the codebase. Every module has one job.

```mermaid
flowchart TB
    subgraph Entry[Entry points]
        CLI[main.py<br/>CLI]
        UI[app.py<br/>Gradio UI]
    end

    subgraph Core[src/response_agent/]
        PIP[pipeline.py<br/>orchestration]
        CFG[config.py<br/>env → Config]
        LLM[llm.py<br/>provider abstraction]
        ING[ingest.py<br/>PDF / TXT loaders]
        CHK[chunking.py<br/>character splitter]
        RET[retrieval.py<br/>embed + cosine top-k]
        AGT[agents.py<br/>4 prompts + functions]
        VER[verifier.py<br/>citation regex check]
        ASM[assemble.py<br/>render Markdown letter]
        EXP[export.py<br/>Markdown → PDF]
    end

    CLI --> PIP
    UI --> PIP
    UI --> AGT
    UI --> RET
    PIP --> ING
    PIP --> CHK
    PIP --> RET
    PIP --> AGT
    PIP --> VER
    PIP --> ASM
    PIP --> EXP
    PIP --> LLM
    RET --> LLM
    AGT --> LLM
    LLM --> CFG
```

---

## 5. The 5 LLM roles

The agent is not one prompt — it's five roles, each with a distinct system
prompt and a distinct job.

```mermaid
flowchart LR
    subgraph Roles[5 LLM-driven roles]
        P[PARSE<br/>split raw reviews<br/>→ atomic comments<br/>JSON mode, T=0.2]
        D[DRAFT / Author<br/>write rebuttal<br/>grounded in chunks<br/>T=0.4]
        C[CRITIQUE / Editor<br/>4-criterion check<br/>JSON mode, T=0.1]
        R[REFINE<br/>same as Author<br/>+ critic issues<br/>T=0.4]
        A[ADVERSARIAL<br/>hostile follow-up<br/>diagnostic only<br/>T=0.6]
    end

    P -.->|once per run| D
    D --> C
    C -->|if ok=false| R
    R --> C
    D -.->|optional| A

    classDef role fill:#fff3e0,stroke:#f57c00,color:#000
    class P,D,C,R,A role
```

---

## Rendering tips

- All five render cleanly in [mermaid.live](https://mermaid.live).
- For the gallery, "Diagram 1" (high-level) + "Diagram 2" (agent loop) carry
  the most signal — screenshot those at minimum.
- Mermaid theme: use the default light theme for white-background screenshots,
  or `theme: dark` for dark-mode screenshots that match the Gradio UI.
