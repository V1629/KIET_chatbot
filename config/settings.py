# ══════════════════════════════════════════════════════
# config/settings.py — Central configuration
# ══════════════════════════════════════════════════════

# ── Folders ───────────────────────────────────────────
DATA_FOLDER       = "data"           # Drop your .json files here
PERSIST_DIR       = "./storage"      # All persistent indexes saved here
CHROMA_DIR        = "./storage/chroma"
DUCKDB_PATH       = "./storage/duck.db"
BM25_PATH         = "./storage/bm25.pkl"
HASH_PATH         = "./storage/data_hash.txt"

# ── Chunking ──────────────────────────────────────────
CHUNK_SIZE        = 600              # words per chunk
OVERLAP           = 80               # word overlap between chunks

# ── Retrieval ─────────────────────────────────────────
INITIAL_TOP_K     = 20              # candidates before reranking
FINAL_TOP_K       = 5               # chunks after reranking
MMR_LAMBDA        = 0.6             # 0=diversity, 1=relevance
RERANK_MIN_SCORE  = 0.15            # below this → context too weak, fall back to GPT

# ── Models ────────────────────────────────────────────
EMBED_MODEL       = "text-embedding-3-small"
LLM_MODEL         = "gpt-4o"
ROUTER_MODEL      = "gpt-4o-mini"   # Cheaper + faster for routing/HyDE
COHERE_RERANK     = "rerank-english-v3.0"

# ── HyDE ─────────────────────────────────────────────
# Only use HyDE if query is vague (fewer than this many words = likely specific)
HYDE_MIN_WORDS    = 6               # queries with < 6 words skip HyDE

# ── Cache ─────────────────────────────────────────────
CACHE_MAX_SIZE    = 200

# ── Embedding batch size ──────────────────────────────
EMBED_BATCH_SIZE  = 100
