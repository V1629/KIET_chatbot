# ══════════════════════════════════════════════════════
# pipeline/orchestrator.py — Master query pipeline
#
# This is the single entry point for every user query.
# It wires together: cache → router → retrieval → generation
#
# Optimizations applied:
#   ✅ Merged router + HyDE into ONE LLM call
#   ✅ HyDE skipped for short/specific queries
#   ✅ Faithfulness check is ASYNC (runs after streaming)
#   ✅ In-memory cache with MD5 key
# ══════════════════════════════════════════════════════

import hashlib
import json
from typing import Optional

import cohere
from openai import OpenAI

from config.settings import ROUTER_MODEL, CACHE_MAX_SIZE, RERANK_MIN_SCORE
from retrieval.retriever import retrieve
from generation.generator import (
    answer_from_chunks, run_sql_agent,
    answer_general, answer_general_fallback, check_faithfulness
)


# ══════════════════════════════════════════════════════
# CACHE — Simple in-memory dict
# ══════════════════════════════════════════════════════

_cache: dict = {}


def cache_get(query: str) -> Optional[dict]:
    key = hashlib.md5(query.lower().strip().encode()).hexdigest()
    return _cache.get(key)


def cache_set(query: str, value: dict):
    global _cache
    if len(_cache) >= CACHE_MAX_SIZE:
        oldest = next(iter(_cache))
        del _cache[oldest]
    key = hashlib.md5(query.lower().strip().encode()).hexdigest()
    _cache[key] = value


def cache_clear():
    global _cache
    _cache = {}


def cache_size() -> int:
    return len(_cache)


# ══════════════════════════════════════════════════════
# OPTIMIZED ROUTER — Merges routing + HyDE in ONE call
# ══════════════════════════════════════════════════════

ROUTER_SYSTEM = """You are a query classifier for a data chatbot.

Classify the query into one of:
  ANALYTICAL — numbers, calculations, aggregations, averages, counts, sums, statistics
  TEXT       — searching for information, descriptions, finding records by content
  GENERAL    — general knowledge unrelated to any specific dataset

Also, if the query is TEXT type AND has 6 or more words (vague/broad),
generate a short hypothetical answer passage (3-5 sentences) that would answer it.
Otherwise set hyde to null.

Respond ONLY with valid JSON (no markdown):
{
  "route": "TEXT" | "ANALYTICAL" | "GENERAL",
  "hyde": "hypothetical passage..." | null
}"""


def route_and_hyde(client: OpenAI, query: str) -> tuple[str, Optional[str]]:
    """
    OPTIMIZED: Single LLM call that both:
      1. Classifies the query (TEXT / ANALYTICAL / GENERAL)
      2. Generates HyDE if needed

    Previously this was 2 separate LLM calls.
    Now it's 1 — saves ~0.4s latency per query.
    """
    r = client.chat.completions.create(
        model    = ROUTER_MODEL,
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user",   "content": query},
        ],
        temperature = 0,
        max_tokens  = 300,
    )
    raw = r.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
        route  = parsed.get("route", "TEXT").upper()
        hyde   = parsed.get("hyde", None)
        if route not in ("TEXT", "ANALYTICAL", "GENERAL"):
            route = "TEXT"
        return route, hyde
    except json.JSONDecodeError:
        # Fallback if LLM doesn't return valid JSON
        if "ANALYTICAL" in raw.upper(): return "ANALYTICAL", None
        if "GENERAL"    in raw.upper(): return "GENERAL",    None
        return "TEXT", None


# ══════════════════════════════════════════════════════
# PIPELINE RESULT — Structured return type
# ══════════════════════════════════════════════════════

class PipelineResult:
    """Everything the UI needs to render a response."""
    def __init__(self):
        self.stream        = None    # OpenAI streaming response (or cached string)
        self.route         = ""      # TEXT / ANALYTICAL / GENERAL
        self.chunks        = []      # Retrieved chunks (for sources)
        self.result_df     = None    # DataFrame for ANALYTICAL queries
        self.from_cache    = False   # Was this a cache hit?
        self.hyde_used     = False   # Was HyDE applied?
        self.steps         = []      # Pipeline steps for UI display
        self.answer_text   = ""      # Filled in AFTER streaming completes
        self.faithful      = True    # Faithfulness check result


# ══════════════════════════════════════════════════════
# MASTER PIPELINE
# ══════════════════════════════════════════════════════

def run_pipeline(idx: dict,
                 client: OpenAI,
                 co_client: cohere.Client,
                 query: str) -> PipelineResult:
    """
    Master pipeline — called for every user query.

    Flow:
        Cache check
            ↓ miss
        Route + HyDE  (1 LLM call, merged)
            ↓
        ┌── TEXT ──────────────────────────────────────────┐
        │  Retrieve: Vector + BM25 → RRF → MMR → Rerank   │
        │  Generate: GPT-4o grounded answer (streamed)     │
        │  [Async after stream]: Faithfulness check        │
        └──────────────────────────────────────────────────┘
        ┌── ANALYTICAL ────────────────────────────────────┐
        │  SQL Agent: generate → execute → self-correct    │
        │  Narrate: GPT-4o explains result (streamed)      │
        └──────────────────────────────────────────────────┘
        ┌── GENERAL ───────────────────────────────────────┐
        │  GPT-4o direct answer (streamed)                 │
        └──────────────────────────────────────────────────┘
            ↓
        Cache result
    """
    result = PipelineResult()

    # ── 0. Cache check ─────────────────────────────────────────────────────
    cached = cache_get(query)
    if cached:
        result.from_cache  = True
        result.stream      = cached["answer"]
        result.route       = cached["route"]
        result.result_df   = cached.get("df")
        result.steps       = ["⚡ Cache hit — instant answer"]
        return result

    # ── 1. Route + HyDE (single merged LLM call) ───────────────────────────
    route, hyde_text = route_and_hyde(client, query)
    result.route = route
    result.steps.append(f"🔀 Routed as: **{route}**")

    # ── 2. GENERAL path ────────────────────────────────────────────────────
    if route == "GENERAL":
        result.stream = answer_general(client, query)
        result.steps.append("🌐 Answering from general knowledge — no retrieval")
        return result

    # ── 3. ANALYTICAL path ─────────────────────────────────────────────────
    if route == "ANALYTICAL" and idx.get("db_schemas"):
        result.steps.append("🗃️ SQL Agent: generating query...")
        result_df, sql_stream = run_sql_agent(client, query, idx["db_schemas"])
        result.result_df = result_df

        if isinstance(sql_stream, str):
            # SQL completely failed — fall back to TEXT path
            result.steps.append(f"⚠️ SQL failed ({sql_stream}) — falling back to text search")
            route        = "TEXT"
            result.route = "TEXT"
        else:
            result.stream = sql_stream
            result.steps.append("✅ SQL executed — streaming result")
            return result

    # ── 4. TEXT path ───────────────────────────────────────────────────────

    # Pass HyDE text into retriever if it was generated in step 1
    # (avoids a second LLM call — HyDE was already generated by router)
    result.steps.append("🔍 Hybrid retrieval: Vector + BM25 + RRF...")

    if hyde_text:
        # Inject pre-generated HyDE into retrieval
        result.chunks, result.hyde_used = retrieve(
            idx, client, co_client, query, hyde_override=hyde_text
        )
        result.steps.append("💡 HyDE applied (pre-generated in routing step)")
    else:
        result.chunks, result.hyde_used = retrieve(
            idx, client, co_client, query
        )

    result.steps.append(f"📊 Cohere reranked → top {len(result.chunks)} chunks")

    # ── 4b. Fallback: if no chunks or all rerank scores are too low,
    #         use GPT-4o general knowledge instead of grounded answer ────
    if not result.chunks or all(
        c.get("rerank_score", 0) < RERANK_MIN_SCORE for c in result.chunks
    ):
        result.route = "GENERAL"
        result.chunks = []
        result.stream = answer_general_fallback(client, query)
        result.steps.append(
            "🌐 Context not strong enough — answering with GPT-4o general knowledge"
        )
        return result

    result.stream = answer_from_chunks(client, query, result.chunks)

    return result


# ══════════════════════════════════════════════════════
# POST-STREAM: Faithfulness check (called after streaming)
# ══════════════════════════════════════════════════════

def run_faithfulness(client: OpenAI,
                     result: PipelineResult) -> bool:
    """
    Run faithfulness check AFTER the answer has been streamed to the user.
    User already sees the full answer — this runs in the background.
    Returns True if answer is faithful, False if hallucination detected.
    """
    if result.route != "TEXT" or not result.chunks or not result.answer_text:
        return True
    return check_faithfulness(client, result.chunks, result.answer_text)
