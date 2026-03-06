# Advanced RAG Chatbot

## Project Structure

```
chatbot/
│
├── app.py                        ← Entry point (Streamlit UI)
│
├── config/
│   └── settings.py               ← All configuration in one place
│
├── storage/
│   └── store.py                  ← Persistent ChromaDB, DuckDB, BM25
│
├── indexing/
│   └── indexer.py                ← JSON loading, chunking, embedding, indexing
│
├── retrieval/
│   └── retriever.py              ← Hybrid search, RRF, MMR, Cohere rerank
│
├── generation/
│   └── generator.py              ← LLM answer, SQL agent, faithfulness check
│
├── pipeline/
│   └── orchestrator.py           ← Master pipeline: routes + wires everything
│
├── data/                         ← DROP YOUR JSON FILES HERE
├── storage/                      ← Auto-created: persistent indexes saved here
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Add Your Data

Put all `.json` files in the `data/` folder.

## Run

```bash
streamlit run app.py
```

## What Happens

### First Run
- Loads all JSON files from `data/`
- Chunks, embeds, and saves to `storage/` (persistent)
- Builds DuckDB tables from tabular JSON
- Saves BM25 index to disk

### Every Restart After
- Loads indexes from `storage/` instantly — **no re-embedding**
- Only re-indexes if JSON files have changed

## Query Flow

```
User Query
    │
    ▼
[Cache Check] ──── HIT ────▶ Return instantly
    │ MISS
    ▼
[Route + HyDE] ← ONE merged LLM call (GPT-4o-mini)
    │
    ├── GENERAL    ──▶ GPT-4o direct answer
    │
    ├── ANALYTICAL ──▶ DuckDB SQL Agent ──▶ Narrate result
    │
    └── TEXT ──▶ Hybrid Retrieval
                    │
                    ├── Vector Search (ChromaDB)
                    ├── BM25 Search
                    └── RRF Fusion
                         │
                         ▼
                    MMR Diversity Filter
                         │
                         ▼
                    Cohere Reranker
                         │
                         ▼
                    GPT-4o Answer (streamed)
                         │
                         ▼
                    Faithfulness Check (background thread)
```

## LLM Calls Per Query

| Route   | Calls | Perceived Latency |
|---------|-------|-------------------|
| GENERAL | 2     | ~1.5s             |
| ANALYTICAL (no error) | 3 | ~3s |
| TEXT (short query, no HyDE) | 2 | ~2.5s |
| TEXT (long query, HyDE) | 2* | ~3s |

*Router + HyDE merged into 1 call. Faithfulness runs async — user sees no wait.

## Configuration

Edit `config/settings.py` to tune:
- `CHUNK_SIZE` — chunk size in words
- `INITIAL_TOP_K` — candidates before reranking
- `FINAL_TOP_K` — chunks after reranking
- `HYDE_MIN_WORDS` — minimum words to trigger HyDE
- `MMR_LAMBDA` — relevance vs diversity balance
