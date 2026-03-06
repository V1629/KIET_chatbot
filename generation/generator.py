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

Rules:
- Prioritise answering from the provided context about KIET University.
- Every factual claim backed by context must end with [Source: <page title or URL>].
- If asked for contact details (email/phone/address), provide them exactly as given.
- If the context does not contain the answer, use your general knowledge to give a helpful, accurate response. Do NOT say the information is unavailable.
- Never invent KIET-specific names, numbers, dates, or rankings that are not in context.
- Be concise: bullet points for lists, single sentence for simple facts."""

def answer_from_chunks(client, query, chunks):
    context = "\n".join(
        f"\n[Source: {c['meta'].get('page_title', c['meta'].get('url',''))}]\n{c['text']}"
        for c in chunks
    )
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
        model=LLM_MODEL,
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
        return result_df, "No records found for that query."
    preview = result_df.head(20).to_string(index=False)
    stream  = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": NARRATE_SYSTEM},
                  {"role": "user",   "content": f"Question: {query}\n\nResult:\n{preview}\n\nAnswer:"}],
        temperature=0.0, stream=True,
    )
    return result_df, stream


# ── GENERAL ───────────────────────────────────────────────────────────────

GENERAL_SYSTEM = "You are a helpful assistant. Answer concisely and accurately."

def answer_general(client, query):
    return client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": GENERAL_SYSTEM},
                  {"role": "user",   "content": query}],
        temperature=0.2, stream=True,
    )


GENERAL_FALLBACK_SYSTEM = """You are a knowledgeable and helpful assistant.

A user has asked a question related to KIET University. The available internal knowledge base did not provide sufficient information to confidently answer the query.

Your task is to provide a **clear, accurate, and helpful answer using your general knowledge**.

Guidelines:

* Do **not** mention the internal database, retrieval process, or missing context.
* Respond naturally as if answering the question directly.
* If the question relates to universities, academic processes, or events, provide a **reasonable and realistic explanation based on typical university practices**.
* Avoid speculation. If exact details are uncertain, provide a **general but helpful answer** that would typically apply in similar situations.
* Keep the response **concise, clear, and informative**.

Your goal is to ensure the user receives a **useful answer even when specific internal information is unavailable**.
"""

def answer_general_fallback(client, query):
    """GPT-4o fallback when retrieved context is too weak to answer."""
    return client.chat.completions.create(
        model=LLM_MODEL,
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
