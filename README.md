# ChatLibrary

> **Archived.** ChatLibrary was built in September 2024 as an early experiment in
> agentic, self-correcting RAG (retrieval-augmented generation). It is kept here
> as a portfolio artifact, not an actively maintained product. The reusable
> grounding machinery lives in [`grounding/`](./grounding/).

## What it is

A chat application that answers questions from a private knowledge base with
cited references. Instead of a single retrieve-then-generate pass, it runs a
LangGraph-based **corrective RAG loop**:

```
question
  ├─ route ──► vectorstore ──► retrieve docs ──► grade each doc's relevance
  │                                          ├─ all relevant ──► generate answer
  │                                          └─ none relevant ─► rewrite query ──► re-retrieve
  │              │
  │              └─ generated answer ──► hallucination check ──► supported? ──► answer-quality check
  │                                          ├─ hallucinated ─► regenerate
  │                                          └─ unhelpful ──► rewrite query, retry
  └─ route ──► default answer (not backed by the library)
```

Every grader (router, relevance, hallucination, answer quality) is an LLM with
structured output, and every answer ships with its source documents as
references. The Streamlit UI adds star-rating feedback, per-user cost/token
tracking, and multi-library source selection, with feedback persisted to a
parquet log.

At the time (fall 2024) this looped, self-checking style of RAG was the
cutting edge of "agentic" retrieval; the pattern is now standard practice.

## Project structure

- `app.py`, `pages/home.py` — Streamlit UI (chat, settings, feedback, cost tracking)
- `src/rag/graph.py` — the corrective-RAG LangGraph pipeline
- `src/rag/chatbot.py` — plain conversational fallback chain (no sources selected)
- `src/rag/tools.py` — structured-output graders (router, relevance, hallucination, answer, query rewriter)
- `src/vectordb/` — ChromaDB ingestion (`db.py`) and retrieval (`retriever.py`)
- `src/utils/` — config, Azure OpenAI helpers, feedback logging
- `grounding/` — framework-agnostic extraction of the grader loop, usable without the app

## Installation

```bash
git clone https://github.com/rnepal2/ChatLibrary.git
cd ChatLibrary
pip install -r requirements.txt
```

Set the Azure OpenAI credentials:

```bash
export AZURE_OPENAI_API_KEY='<your-key>'
export AZURE_OPENAI_ENDPOINT='https://<your-resource>.openai.azure.com/'
```

## Usage

```bash
streamlit run app.py --server.port 3000
```

Collections (`Library_1`, `Library_2`) are expected to exist in the ChromaDB
persistent store; see `src/vectordb/db.py` for the ingestion script.

## Tech stack

Streamlit · LangChain 0.3 · LangGraph 0.2 · Azure OpenAI (GPT-4o / 4o-mini) ·
ChromaDB · structured-output graders · parquet feedback log

## Maintenance note (Oct 2026)

This repo is archived: no new features, no dependency upgrades (the pinned
2024-era stack is kept so the experiment remains reproducible as built).

Known limitations, kept as-is for historical accuracy:

- The `web_search` graph node is a placeholder — there is no real web search.
- `src/vectordb/retriever.py` hardcodes the vector store path (`/chromadb`).
- `pages/home.py` had a dead feedback-logging gate (`len(sources) > 10`, always
  false with two source checkboxes); fixed to `> 0` during archival so the
  feedback feature works as documented.
- `requirements.txt` was missing `pandas`, `docx2txt`, `pyarrow` and `numpy`
  despite being imported; added during archival.
- There is no authentication — every session runs as a guest user.
- Collection selection is hardcoded to `Library_1` / `Library_2`.

## License

MIT
