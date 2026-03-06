# retrieval/retriever.py  —  KIET-Optimised Hybrid Retrieval
#
# Improvements over generic version:
#   1. Metadata-aware reranking — contact/table chunks boosted for relevant queries
#   2. chunk_type filter — "contact" queries go directly to contact chunks
#   3. HyDE skipped for short queries (< HYDE_MIN_WORDS)
#   4. hyde_override accepted (pre-generated in orchestrator — saves 1 LLM call)

import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cohere
from openai import OpenAI

from config.settings import (
    EMBED_MODEL, INITIAL_TOP_K, FINAL_TOP_K,
    MMR_LAMBDA, COHERE_RERANK, HYDE_MIN_WORDS, ROUTER_MODEL
)


# ── Embedding ─────────────────────────────────────────────────────────────

def get_embedding(client, text):
    r = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return r.data[0].embedding


# ── HyDE ──────────────────────────────────────────────────────────────────

HYDE_PROMPT = """Write a short hypothetical university webpage passage (3-5 sentences)
that would perfectly answer this question about KIET University.
Include realistic details like names, numbers, and specifics."""

def should_use_hyde(query):
    return len(query.strip().split()) >= HYDE_MIN_WORDS

def generate_hyde(client, query):
    r = client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": HYDE_PROMPT},
            {"role": "user",   "content": query},
        ],
        temperature=0.3, max_tokens=120,
    )
    return r.choices[0].message.content.strip()

def get_query_embedding(client, query):
    if should_use_hyde(query):
        hyde_text = generate_hyde(client, query)
        # Parallelize the two embedding calls
        with ThreadPoolExecutor(max_workers=2) as pool:
            q_future    = pool.submit(get_embedding, client, query)
            hyde_future = pool.submit(get_embedding, client, hyde_text)
            q_emb    = q_future.result()
            hyde_emb = hyde_future.result()
        combined = [(a+b)/2 for a, b in zip(q_emb, hyde_emb)]
        return combined, True
    return get_embedding(client, query), False


# ── BM25 ──────────────────────────────────────────────────────────────────

def bm25_search(bm25, all_chunks, query, top_k):
    scores = bm25.get_scores(query.lower().split())
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]


# ── Vector ────────────────────────────────────────────────────────────────

def vector_search(collection, query_emb, top_k, where_filter=None):
    kwargs = dict(query_embeddings=[query_emb], n_results=top_k)
    if where_filter:
        kwargs["where"] = where_filter
    results = collection.query(**kwargs)
    return [int(id_.split("_")[1]) for id_ in results["ids"][0]]


# ── RRF ───────────────────────────────────────────────────────────────────

def rrf(rankings, k=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


# ── MMR ───────────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    return dot / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b)) + 1e-9)

def mmr_select(query_emb, candidate_embs, candidate_data, k, lam=MMR_LAMBDA):
    selected, sel_embs = [], []
    remaining = list(range(len(candidate_data)))
    for _ in range(min(k, len(remaining))):
        best_idx, best_score = None, -999.0
        for i in remaining:
            rel   = cosine_sim(query_emb, candidate_embs[i])
            red   = max((cosine_sim(candidate_embs[i], e) for e in sel_embs), default=0.0)
            score = lam * rel - (1 - lam) * red
            if score > best_score:
                best_score, best_idx = score, i
        selected.append(candidate_data[best_idx])
        sel_embs.append(candidate_embs[best_idx])
        remaining.remove(best_idx)
    return selected


# ── Chunk-type boost ───────────────────────────────────────────────────────
# KIET-specific: certain query keywords hint at which chunk_type is most useful

CONTACT_KEYWORDS = {"email", "phone", "contact", "address", "call", "reach", "number"}
TABLE_KEYWORDS   = {"list", "faculty", "staff", "committee", "members", "board",
                    "governing", "council", "leadership", "designation"}

def detect_chunk_type_filter(query):
    words = set(query.lower().split())
    if words & CONTACT_KEYWORDS:
        return {"chunk_type": "contact"}
    return None


# ── Cohere rerank ─────────────────────────────────────────────────────────

def rerank(co_client, query, candidates):
    docs   = [c["text"] for c in candidates]
    result = co_client.rerank(
        model=COHERE_RERANK, query=query, documents=docs, top_n=FINAL_TOP_K
    )
    reranked = []
    for r in result.results:
        item = candidates[r.index].copy()
        item["rerank_score"] = r.relevance_score
        reranked.append(item)
    return reranked


# ── Master retrieve ───────────────────────────────────────────────────────

def retrieve(idx, client, co_client, query, hyde_override=None):
    """
    Full pipeline:
      embed (+ HyDE if needed) → vector + BM25 → RRF → MMR → Cohere rerank
    Returns (final_chunks, hyde_used)
    """
    collection     = idx["collection"]
    bm25           = idx["bm25"]
    all_chunks     = idx["all_chunks"]
    all_metas      = idx["all_metas"]
    all_embeddings = idx["all_embeddings"]

    # Embedding
    if hyde_override:
        # Parallelize the two embedding calls
        with ThreadPoolExecutor(max_workers=2) as pool:
            q_future    = pool.submit(get_embedding, client, query)
            hyde_future = pool.submit(get_embedding, client, hyde_override)
            q_emb    = q_future.result()
            hyde_emb = hyde_future.result()
        query_emb = [(a+b)/2 for a, b in zip(q_emb, hyde_emb)]
        hyde_used = True
    else:
        query_emb, hyde_used = get_query_embedding(client, query)

    # Optional chunk_type filter for KIET-specific query patterns
    where_filter = detect_chunk_type_filter(query)

    # Vector search
    vec_ids  = vector_search(collection, query_emb, INITIAL_TOP_K, where_filter)
    # BM25 search (no filter — keyword search is already specific)
    bm25_ids = bm25_search(bm25, all_chunks, query, INITIAL_TOP_K)

    # RRF fusion
    fused_ids = rrf([vec_ids, bm25_ids])[:INITIAL_TOP_K]

    # Build candidates
    candidates     = [{"text": all_chunks[i], "meta": all_metas[i], "id": i}
                      for i in fused_ids if i < len(all_chunks)]
    candidate_embs = [all_embeddings[i] for i in fused_ids if i < len(all_chunks)]

    # MMR diversity
    diverse = mmr_select(query_emb, candidate_embs, candidates, k=INITIAL_TOP_K)

    # Cohere rerank → final top-K
    final_chunks = rerank(co_client, query, diverse)

    return final_chunks, hyde_used
