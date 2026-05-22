# Math Agentic RAG — v2

> A modular, locally-runnable, human-in-the-loop agentic RAG for solving math questions.
> Rebuild of the original Colab prototype (`main.py.py`, preserved on `main` branch) with persistent storage, real HITL, LLM-judge routing, and a GSM8K evaluation harness.

The original `README.md` and `main.py.py` are untouched. This document describes v2 on the `v2-rebuild` branch.

---

## What changed v1 → v2

| v1 (Colab prototype) | v2 (this branch) |
|---|---|
| `from google.colab import drive` + `!pip install` magic | Plain Python project; `requirements.txt` |
| `/content/...` paths hardcoded | `src/config.py` with sane defaults |
| Qdrant in-memory (re-embeds every run) | **Qdrant local persistent** + **content-hash dedup** |
| Row-by-row embedding | **Batched embed** (`batch=64`) |
| `os.getenv("63ee70aa-...")` (Exa key passed as env var NAME) | Real `.env` with `EXA_API_KEY` |
| `from dspy import Chain` (removed in modern DSPy) | Plain `MathAgent` class, modern LLM client |
| `HuggingFaceHub` Flan-T5 (weak at math) | **Groq Llama-3.3** (fast + strong reasoning) |
| `len(answer) < 10` quality heuristic | **LLM-judge** routing |
| HITL appends to DataFrame, **never updates the index** | HITL **upserts into Qdrant live** — fix of the silent v1 bug |
| No evaluation | **GSM8K exact-match eval** with route breakdown |

---

## Architecture

```
                       ┌──────────────────┐
User math question ──▶ │   MathAgent      │
                       └────────┬─────────┘
                                │
              ┌─────────────────┼──────────────────┐
              ▼                 ▼                  ▼
       ┌──────────┐      ┌─────────────┐    ┌──────────────┐
       │ Embed    │      │ Groq        │    │ LLM-judge    │
       │ MiniLM   │      │ Llama-3.3   │    │ "grounded?"  │
       │ (batched)│      │ generation  │    │ YES / NO     │
       └────┬─────┘      └─────────────┘    └──────┬───────┘
            │                                       │
            ▼                                NO ──▶ │
   ┌─────────────────┐                              ▼
   │ Qdrant local    │                       ┌────────────┐
   │ persistent      │                       │ Exa web    │
   │ + content-hash  │                       │ fallback   │
   │ dedup           │                       └─────┬──────┘
   └─────────────────┘                             ▼
                                          Re-generate w/ web ctx

After every answer:
   user types correction → embed → live upsert to Qdrant (v1 bug fix)
```

---

## Setup

### Prerequisites
- **Python 3.10+** (tested on 3.12.4 Windows)
- **Groq API key** (free) — https://console.groq.com
- **Exa API key** (free tier) — https://exa.ai
- Optional: **ORCA-Math 200k parquet** — https://www.kaggle.com/datasets/johnsonhk88/microsoftorca-math-word-problems-200k
  → place at `Datasets/raw/orca/200k.parquet`

### Install
```powershell
git checkout v2-rebuild
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> **Windows note:** if `import torch` crashes with `WinError 1114`, force-reinstall a CPU wheel:
> `pip install --force-reinstall "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu`

### Configure
```powershell
copy .env.example .env
# edit .env: GROQ_API_KEY=..., EXA_API_KEY=...
```

### Prepare data
```powershell
python scripts/prepare_data.py
```
Unzips the three repo datasets, verifies file layout. Prints a `MISSING` line for ORCA-Math if you didn't download it (the build still works on the other 3 sources).

### Build the index
```powershell
# quick demo (~3 min on CPU)
python scripts/build_index.py --subset 5000

# full corpus (~30+ min on CPU)
python scripts/build_index.py
```
Idempotent — content-hash dedup means re-running on the same data is a no-op.

### Tests
```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

### Use it
```powershell
# Interactive chat with HITL
python scripts/chat.py

# Or eval on GSM8K test split
python scripts/eval.py --n 50
```

---

## Project layout

```
AI-MATH-AGENT/
├── main.py.py            # ORIGINAL Colab export, untouched
├── README.md             # ORIGINAL readme, untouched
├── README.v2.md          # this file
├── ASSESSMENT.md         # writeup for the prework submission
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   ├── config.py         # paths, env vars, model IDs, top-k
│   ├── data.py           # load + normalize 4 datasets
│   ├── index.py          # persistent Qdrant, batched embed, hash dedup, retrieve
│   ├── agent.py          # MathAgent: retrieve -> gen -> judge -> fallback
│   ├── fallback.py       # Exa web search wrapper
│   ├── feedback.py       # HITL loop with REAL upsert (v1 bug fix)
│   └── evaluate.py       # GSM8K exact-match accuracy
├── scripts/
│   ├── prepare_data.py
│   ├── build_index.py
│   ├── chat.py
│   └── eval.py
├── tests/
│   ├── test_data.py
│   ├── test_index.py     # round-trip embed + retrieve + dedup
│   └── test_agent.py     # routing + answer extraction
└── Datasets/             # zips kept; raw/ is gitignored
```

---

## How the HITL bug fix works

In v1 the feedback loop appended `(question, correction)` to an in-memory pandas DataFrame and printed *"Will incorporate into next training."* But the Qdrant index was built once at startup and never rebuilt, so corrections never affected retrieval.

In v2 (`src/feedback.py`), an `n` answer triggers `add_correction(question, correction)` in `src/index.py`, which:
1. Hashes `(question, correction)` and converts to a UUID point id
2. Embeds with the same MiniLM model used for the corpus
3. Calls `client.upsert()` on the live Qdrant collection
4. Logs the new point id so the user can see the change happened

The next retrieval — even within the same chat session — will surface that correction.

---

## What's intentionally NOT in v2 (next steps)

- No fine-tuning (no compute, out of scope)
- No multi-turn conversation memory (single-turn agent)
- No web UI (CLI only)
- No structured logging / telemetry
- No Docker / CI

These are the natural follow-ups; see `ASSESSMENT.md` Section 4.
