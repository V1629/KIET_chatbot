# ══════════════════════════════════════════════════════
# retrieval/web_search.py — Tavily web search fallback
#
# When local retrieval context is too weak, this module
# fetches live web results via Tavily and returns them
# in a format the generator can use.
# ══════════════════════════════════════════════════════

import os
import logging
from typing import Optional

from config.settings import WEB_TOP_K

logger = logging.getLogger(__name__)


def _get_tavily_client():
    """Lazy-init Tavily client. Returns None if key is missing."""
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        logger.warning("TAVILY_API_KEY not set — web search disabled")
        return None
    try:
        from tavily import TavilyClient
        return TavilyClient(api_key=api_key)
    except ImportError:
        logger.warning("tavily-python not installed — web search disabled")
        return None


def web_search(query: str, max_results: int = WEB_TOP_K) -> list[dict]:
    """
    Search the web via Tavily and return a list of result dicts.

    Each result dict has:
        - title: str
        - url: str
        - content: str  (snippet / summary)

    Returns an empty list if Tavily is unavailable or the search fails.
    """
    client = _get_tavily_client()
    if client is None:
        return []

    # Always scope the search to KIET University so generic phrases
    # like "this college" don't return results about random institutions.
    kiet_keywords = ["kiet", "kiet university", "kiet ghaziabad",
                     "kiet deemed", "kiet group"]
    query_lower = query.lower()
    if not any(kw in query_lower for kw in kiet_keywords):
        search_query = f"KIET University Ghaziabad {query}"
    else:
        search_query = query

    try:
        response = client.search(
            query=search_query,
            max_results=max_results,
            search_depth="basic",          # "basic" is fast; "advanced" is slower but richer
            include_answer=False,
            include_raw_content=False,
        )
        results = response.get("results", [])

        # Normalise into a clean list
        clean = []
        for r in results:
            clean.append({
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "content": r.get("content", ""),
            })
        return clean

    except Exception as e:
        logger.error(f"Tavily web search failed: {e}")
        return []


def format_web_context(results: list[dict]) -> str:
    """Format web search results into a context string for the LLM."""
    if not results:
        return ""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[Web Source {i}: {r['title']}]\n"
            f"URL: {r['url']}\n"
            f"{r['content']}\n"
        )
    return "\n".join(parts)
