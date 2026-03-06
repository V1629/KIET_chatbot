# generation/generator.py  —  KIET-Optimised Answer Generation
#
# Changes from generic version:
#   • System prompt references KIET University explicitly
#   • SQL narration knows it's university/tabular data
#   • Faithfulness check uses ROUTER_MODEL (fast, cheap)

import pandas as pd
from openai import OpenAI

from config.settings import LLM_MODEL, ROUTER_MODEL
from storage.store import get_duckdb_connection


# ── TEXT: grounded answer ─────────────────────────────────────────────────

TEXT_SYSTEM = """You are the official KIET University information assistant.

CRITICAL RULE — CONTEXT SUFFICIENCY CHECK:
If the provided context does NOT contain a direct, specific answer to the user's question
(e.g. the user asks for a number/count/name/date but the context only has vague mentions),
you MUST respond with EXACTLY this on the first line:
[CONTEXT_INSUFFICIENT]
Then stop. Do NOT attempt to answer. Do NOT suggest contacting anyone. Do NOT say "the context does not specify".

If the context DOES contain the answer, respond normally following these rules:
- For YES/NO questions: answer with a clear "Yes" or "No" FIRST, then give a one-line explanation.
- Every factual claim backed by context must end with [Source: <page title or URL>].
- If asked for contact details (email/phone/address), provide them exactly as given.
- Never invent KIET-specific names, numbers, dates, or rankings that are not in context.
- Be concise and direct: bullet points for lists, single sentence for simple facts.
- Match the user's tone — if they ask casually, respond conversationally.
- NEVER say "the context does not specify", "not mentioned", "you may contact" or anything similar. Either answer fully or output [CONTEXT_INSUFFICIENT]."""

def _build_text_context(chunks):
    """Build context string from chunks."""
    return "\n".join(
        f"\n[Source: {c['meta'].get('page_title', c['meta'].get('url',''))}]\n{c['text']}"
        for c in chunks
    )


def check_context_sufficiency(client, query, chunks):
    """
    Quick non-streaming check: does the context actually answer the query?
    Returns True if context is sufficient, False if [CONTEXT_INSUFFICIENT].
    Uses a FAST peek — only reads the first few tokens of a streaming response.
    """
    context = _build_text_context(chunks)
    stream = client.chat.completions.create(
        model=ROUTER_MODEL,   # fast + cheap for this check
        messages=[
            {"role": "system", "content": TEXT_SYSTEM},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.0, max_tokens=30, stream=True,
    )
    # Peek at just the first ~30 tokens then stop
    prefix = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        prefix += delta
        if len(prefix) >= 25:
            break
    # Close the stream early
    stream.close()
    return "[CONTEXT_INSUFFICIENT]" not in prefix


def answer_from_chunks(client, query, chunks):
    context = _build_text_context(chunks)
    return client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": TEXT_SYSTEM},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.0, stream=True,
    )


# ── ANALYTICAL: SQL agent ─────────────────────────────────────────────────

SQL_SYSTEM = """You are a DuckDB SQL expert working with KIET University data tables.
Return ONLY raw SQL. No explanation. No markdown backticks.
Use exact table and column names from the schema.
For text matching use ILIKE '%value%'. Limit to 100 rows unless aggregating."""

NARRATE_SYSTEM = """You are a KIET University data analyst.
Given a SQL result, write a clear, concise natural language answer.
Include exact values. Keep it brief."""

def _gen_sql(client, query, schema_str, error=""):
    prompt = f"KIET University database schema:\n{schema_str}\n\n"
    if error:
        prompt += f"Previous attempt failed: {error}\nFix the SQL.\n\n"
    prompt += f"Question: {query}\nSQL:"
    r = client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=[{"role": "system", "content": SQL_SYSTEM},
                  {"role": "user",   "content": prompt}],
        temperature=0, max_tokens=300,
    )
    return r.choices[0].message.content.strip()

def run_sql_agent(client, query, db_schemas):
    if not db_schemas:
        return None, "No tabular data available."
    schema_str = "\n\n".join(f"Table: {t}\n{s}" for t, s in db_schemas.items())
    conn = get_duckdb_connection()
    sql  = _gen_sql(client, query, schema_str)
    try:
        result_df = conn.execute(sql).df()
    except Exception as e1:
        sql = _gen_sql(client, query, schema_str, error=str(e1))
        try:
            result_df = conn.execute(sql).df()
        except Exception as e2:
            conn.close()
            return None, f"SQL failed: {e2}"
    conn.close()
    if result_df.empty:
        return result_df, "The database query returned no matching records. Let me try to answer from other sources."
    preview = result_df.head(20).to_string(index=False)
    stream  = client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=[{"role": "system", "content": NARRATE_SYSTEM},
                  {"role": "user",   "content": f"Question: {query}\n\nResult:\n{preview}\n\nAnswer:"}],
        temperature=0.0, stream=True,
    )
    return result_df, stream


# ── GENERAL ───────────────────────────────────────────────────────────────

GENERAL_SYSTEM = "You are a helpful assistant. Answer concisely and accurately."

def answer_general(client, query):
    return client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=[{"role": "system", "content": GENERAL_SYSTEM},
                  {"role": "user",   "content": query}],
        temperature=0.2, stream=True,
    )


# ── WEB SEARCH ANSWER ────────────────────────────────────────────────────

WEB_SYSTEM = """You are the official KIET University assistant (KIET Deemed to be University, Delhi-NCR, Ghaziabad), augmented with live web search.

You have been given web search results for the user's query. Use them to give
a comprehensive, accurate answer.

CRITICAL RULES:
- You MUST only use information that is about KIET University (KIET Group of Institutions,
  KIET Ghaziabad, Delhi-NCR). Ignore ANY web result about other colleges or universities.
- When the user says "this college", "the college", "here", "our university" etc., they
  ALWAYS mean KIET University — never any other institution.
- If none of the web results are about KIET, say: "I couldn't find specific information
  about this for KIET University. Please contact admissions@kiet.edu or +91-8445557599."
- Cite sources inline as [Source: <title or URL>].
- For YES/NO questions, answer yes or no first, then explain.
- Do NOT say "according to web search" or reveal the search process.
- Be conversational and helpful."""


def answer_from_web(client, query, web_results_context: str):
    """Generate an answer grounded in Tavily web search results."""
    return client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": WEB_SYSTEM},
            {"role": "user",   "content": f"Web Results:\n{web_results_context}\n\nQuestion: {query}"},
        ],
        temperature=0.2, stream=True,
    )


GENERAL_FALLBACK_SYSTEM = """You are KIET University's official assistant (KIET Deemed to be University, Delhi-NCR, Ghaziabad).

The internal knowledge base did not have a strong match for this query, so answer using
your general knowledge — but ALWAYS answer as KIET's assistant first.

CRITICAL KIET FACTS (use these when relevant):
• Minimum attendance for exam eligibility: 75%
• NAAC A+ accredited, NBA-accredited programmes
• QS I-GAUGE Diamond Rating (2025-2030)
• Deemed University status under UGC Act since November 2025
• 10,000+ students across Engineering, Computer Applications, Management, Pharmacy
• 26,000+ alumni network
• 2,100+ recruiters visited since inception
• Highest international placement: ₹1.78 crore
• TBI-KIET has incubated 240+ startups
• Contact: admissions@kiet.edu | +91-8445557599
• Address: KIET, Delhi-NCR, Ghaziabad-Meerut Road, Ghaziabad (201206)

RULES:
- If the question is about KIET policies, use the facts above.
- For general education/university questions, answer directly and confidently.
- Do NOT mention the internal database, retrieval process, or that context was missing.
- NEVER say "I don't have that information" or "this is not available". Always provide the best possible answer.
- Be concise and direct. For yes/no questions, answer yes or no first."""

def answer_general_fallback(client, query):
    """GPT-4o-mini fallback when retrieved context is too weak to answer."""
    return client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": GENERAL_FALLBACK_SYSTEM},
            {"role": "user",   "content": query},
        ],
        temperature=0.2, stream=True,
    )


# ── Faithfulness check ────────────────────────────────────────────────────

FAITH_SYSTEM = """Given context and an answer about KIET University, 
check if every factual claim is supported by the context.
Reply with only: FAITHFUL or NOT_FAITHFUL"""

def check_faithfulness(client, chunks, answer):
    context = "\n".join(c["text"] for c in chunks[:3])
    r = client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=[{"role": "system", "content": FAITH_SYSTEM},
                  {"role": "user",   "content": f"Context:\n{context}\n\nAnswer:\n{answer}"}],
        temperature=0, max_tokens=10,
    )
    return "NOT_FAITHFUL" not in r.choices[0].message.content.upper()
