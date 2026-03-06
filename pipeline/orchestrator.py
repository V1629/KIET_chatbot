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
import random
from typing import Optional

import cohere
from openai import OpenAI

from config.settings import ROUTER_MODEL, CACHE_MAX_SIZE, RERANK_MIN_SCORE
from retrieval.retriever import retrieve
from retrieval.web_search import web_search, format_web_context
from generation.generator import (
    answer_from_chunks, check_context_sufficiency, run_sql_agent,
    answer_general, answer_general_fallback, answer_from_web,
    check_faithfulness
)


# ══════════════════════════════════════════════════════
# BLOCKED QUERY RESPONSES — hardcoded rejection messages
# ══════════════════════════════════════════════════════

BLOCKED_RESPONSES = [
    "🎓 I'm the KIET University assistant — I can only help with questions about KIET, its courses, campus life, admissions, placements, and student policies. Please ask me something related to KIET!",
    "🎓 Sorry, I'm designed to answer questions about KIET University only — academics, admissions, placements, hostels, fees, faculty, and campus life. How can I help you with KIET?",
    "🎓 That's outside my area! I'm here to help with everything about KIET University — from admissions and courses to placements and campus facilities. Ask me anything about KIET!",
]


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

ROUTER_SYSTEM = """You are a query classifier and rewriter for a KIET University data chatbot.

Do THREE things:

1. **Classify** the query into one of:
   ANALYTICAL — numbers, calculations, aggregations, averages, counts, sums, statistics about KIET
   TEXT       — searching for information, descriptions, finding records about KIET or university/education topics
   GENERAL    — general knowledge completely unrelated to KIET or any university/education topic
   BLOCKED    — queries that are completely unrelated to KIET University, education, students,
                or campus life. Examples: cooking recipes, movie reviews, celebrity gossip,
                coding help, stock market, weather, sports scores, personal advice, etc.

   IMPORTANT CLASSIFICATION RULES:
   - If the query mentions anything about a university, college, campus, exams,
     attendance, fees, placements, faculty, courses, admissions, hostels, or student life,
     classify it as TEXT (not GENERAL or BLOCKED), even if the user doesn't mention "KIET" explicitly.
   - If the query is a greeting (hi, hello, how are you) → TEXT
   - If the query is about education in general → TEXT
   - ONLY classify as BLOCKED if the query has absolutely NOTHING to do with education,
     universities, students, campus, or KIET.

2. **Rewrite** the user's casual/colloquial query into a clear, formal search query
   that would match university policy documents. Fix slang, abbreviations, and
   informal phrasing. Always output this as "search_query". For BLOCKED queries, set to null.

3. If the query is TEXT type AND has 6 or more words, generate a short hypothetical
   answer passage (3-5 sentences). Otherwise set hyde to null.

Respond ONLY with valid JSON (no markdown):
{
  "route": "TEXT" | "ANALYTICAL" | "GENERAL" | "BLOCKED",
  "search_query": "formal rewritten search query..." | null,
  "hyde": "hypothetical passage..." | null
}"""


def route_and_hyde(client: OpenAI, query: str) -> tuple[str, Optional[str], str]:
    """
    OPTIMIZED: Single LLM call that:
      1. Classifies the query (TEXT / ANALYTICAL / GENERAL)
      2. Rewrites casual query → formal search query
      3. Generates HyDE if needed

    Returns (route, hyde_text_or_none, rewritten_search_query)
    """
    r = client.chat.completions.create(
        model    = ROUTER_MODEL,
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user",   "content": query},
        ],
        temperature = 0,
        max_tokens  = 200,
    )
    raw = r.choices[0].message.content.strip()

    try:
        parsed       = json.loads(raw)
        route        = parsed.get("route", "TEXT").upper()
        hyde         = parsed.get("hyde", None)
        search_query = parsed.get("search_query", query)
        if route not in ("TEXT", "ANALYTICAL", "GENERAL", "BLOCKED"):
            route = "TEXT"
        return route, hyde, search_query
    except json.JSONDecodeError:
        # Fallback if LLM doesn't return valid JSON
        if "ANALYTICAL" in raw.upper(): return "ANALYTICAL", None, query
        if "GENERAL"    in raw.upper(): return "GENERAL",    None, query
        if "BLOCKED"    in raw.upper(): return "BLOCKED",    None, query
        return "TEXT", None, query


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

    # ── 1. Route + HyDE + Query Rewrite (single merged LLM call) ──────────
    route, hyde_text, search_query = route_and_hyde(client, query)
    result.route = route
    result.steps.append(f"🔀 Routed as: **{route}**")
    if search_query != query:
        result.steps.append(f"✏️ Rewritten query: _{search_query}_")

    # ── 1b. BLOCKED path — reject non-KIET queries with hardcoded response ─
    if route == "BLOCKED":
        result.route = "BLOCKED"
        result.stream = random.choice(BLOCKED_RESPONSES)
        result.steps.append("🚫 Query unrelated to KIET — blocked")
        return result

    # ── 2. GENERAL path — try web search, then GPT ──────────────────────
    if route == "GENERAL":
        web_results = web_search(query)
        if web_results:
            web_context = format_web_context(web_results)
            result.route  = "WEB"
            result.stream = answer_from_web(client, query, web_context)
            result.steps.append(
                f"🌐 Web search: {len(web_results)} results — answering from web"
            )
        else:
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

    # Use the rewritten search_query for retrieval (better recall)
    # but keep the original query for LLM answer generation (preserves user intent)
    result.steps.append("🔍 Hybrid retrieval: Vector + BM25 + RRF...")

    if hyde_text:
        # Inject pre-generated HyDE into retrieval
        result.chunks, result.hyde_used = retrieve(
            idx, client, co_client, search_query, hyde_override=hyde_text
        )
        result.steps.append("💡 HyDE applied (pre-generated in routing step)")
    else:
        result.chunks, result.hyde_used = retrieve(
            idx, client, co_client, search_query
        )

    result.steps.append(f"📊 Cohere reranked → top {len(result.chunks)} chunks")

    # ── 4b. Fallback: if no chunks or all rerank scores are too low,
    #         try web search first, then GPT-4o general knowledge ────
    if not result.chunks or all(
        c.get("rerank_score", 0) < RERANK_MIN_SCORE for c in result.chunks
    ):
        # --- Web search fallback ---
        result.steps.append("🔍 Context too weak — trying web search...")
        web_results = web_search(query)

        if web_results:
            web_context = format_web_context(web_results)
            result.route  = "WEB"
            result.chunks = []  # no local chunks used
            result.stream = answer_from_web(client, query, web_context)
            result.steps.append(
                f"🌐 Found {len(web_results)} web results — answering from web"
            )
            return result

        # --- Final fallback: GPT-4o with baked-in KIET facts ---
        result.route  = "GENERAL"
        result.chunks = []
        result.stream = answer_general_fallback(client, query)
        result.steps.append(
            "🌐 Web search unavailable — answering with GPT-4o general knowledge"
        )
        return result

    # ── 4c. Context sufficiency check: chunks exist but do they actually
    #         answer the question? (fast check with gpt-4o-mini) ────────
    result.steps.append("🧪 Checking if context answers the question...")
    context_ok = check_context_sufficiency(client, query, result.chunks)

    if not context_ok:
        # Context has related text but NOT the specific answer → web search
        result.steps.append("❌ Context doesn't contain a direct answer — trying web search...")
        web_results = web_search(query)

        if web_results:
            web_context = format_web_context(web_results)
            result.route  = "WEB"
            result.chunks = []
            result.stream = answer_from_web(client, query, web_context)
            result.steps.append(
                f"🌐 Found {len(web_results)} web results — answering from web"
            )
            return result

        # Web also failed → GPT fallback
        result.route  = "GENERAL"
        result.chunks = []
        result.stream = answer_general_fallback(client, query)
        result.steps.append(
            "🌐 Web search unavailable — answering with GPT-4o general knowledge"
        )
        return result

    result.steps.append("✅ Context is sufficient — generating answer...")
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
    if result.route not in ("TEXT",) or not result.chunks or not result.answer_text:
        return True
    return check_faithfulness(client, result.chunks, result.answer_text)
