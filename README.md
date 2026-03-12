# RAG Research Assistant
### Portfolio Project — OpenAI Data Scientist Application

A production-style Retrieval Augmented Generation (RAG) system built on arXiv AI papers, with a built-in failure analysis framework.

---

## What It Does

- **Fetches** recent AI papers from arXiv via their public API
- **Chunks & embeds** abstracts using OpenAI `text-embedding-3-small`
- **Retrieves** relevant chunks via cosine similarity
- **Generates** answers using GPT-4o-mini with grounded context
- **Analyzes failures** automatically on every query

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Then enter your OpenAI API key in the sidebar, pick a topic, and build the knowledge base.

---

## Failure Analysis Framework

This is the core data science contribution. Every RAG query is evaluated for 6 failure modes:

| Flag | Description | Why It Matters |
|------|-------------|----------------|
| `LOW_RETRIEVAL_CONFIDENCE` | Top similarity < 0.70 | Query may be out of distribution |
| `NO_CHUNKS_RETRIEVED` | Vector store returned nothing | Ingestion or embedding issue |
| `POSSIBLE_HALLUCINATION` | Model explicitly signals uncertainty | Answer may be fabricated |
| `HIGH_UNCERTAINTY` | Hedging language detected | Low confidence generation |
| `LOW_SOURCE_DIVERSITY` | All chunks from same paper | Retrieval is too narrow |
| `POSSIBLE_TOPIC_DRIFT` | Answer doesn't overlap with query | Generation went off-topic |

Each query gets a **Health Score** (0–100) based on flags triggered.

---

## Architecture

```
arXiv API
    │
    ▼
ArxivFetcher → Paper objects
    │
    ▼
TextChunker (400 words, 80 overlap)
    │
    ▼
OpenAI Embeddings (text-embedding-3-small)
    │
    ▼
VectorStore (cosine similarity, pure numpy)
    │
    ▼
Query → Embed → Retrieve top-K → GPT-4o-mini → Answer
                                                    │
                                                    ▼
                                          FailureAnalyzer
                                          → flags + health score
```

---

## Key Design Decisions

**Why no FAISS?** Pure numpy keeps dependencies minimal and makes the cosine similarity logic transparent — important for a data science portfolio where you want to show understanding, not just library usage.

**Why arXiv?** Domain-specific corpora expose RAG failure modes more clearly than general Wikipedia. Questions about cutting-edge papers are more likely to hit retrieval boundaries.

**Why GPT-4o-mini?** Cost-efficient for experimentation. The pipeline is model-agnostic; swap to GPT-4o for production.

---

## Extending This Project

Ideas to push further:
- Add **re-ranking** (cross-encoder) after initial retrieval
- Implement **hypothetical document embeddings (HyDE)**
- Build a **golden dataset** of Q&A pairs and compute recall@k
- Add **citation grounding** — verify each sentence against source chunks
- Experiment with **different chunking strategies** and compare failure rates

---

## Skills Demonstrated

- RAG pipeline architecture end-to-end
- Embedding and vector similarity from scratch
- Systematic failure mode identification and measurement
- Clean Python (dataclasses, type hints, separation of concerns)
- Streamlit for interactive ML demos
