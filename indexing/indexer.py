# indexing/indexer.py  —  KIET-Optimised Indexing Pipeline
#
# Data structure (213 pages):
#   url, page_title, heading   -> metadata / context header
#   main_content  (list[str])  -> primary text  -> sentence-aware chunks
#   sections      (dict)       -> labelled blocks -> chunk per section
#   tables        (list[dict]) -> {headers, rows} -> NL sentences + DuckDB
#   contact_info  (dict)       -> dedicated contact chunk
#
# Every chunk gets a context header:
#   [Page: <title> | Section: <heading> | URL: <url>]
# so the LLM always knows the source even mid-context.

import os, re, json, glob
from typing import Optional

import pandas as pd
from openai import OpenAI
from rank_bm25 import BM25Okapi

from config.settings import (
    DATA_FOLDER, EMBED_MODEL, EMBED_BATCH_SIZE,
)
from storage.store import (
    clear_chroma_collection, get_duckdb_connection,
    save_bm25, save_hash, compute_data_hash
)


# ── Sentence-aware splitter ────────────────────────────────────────────────

MAX_WORDS_PER_EMBED_INPUT = 500


def _normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def hard_word_chunks(text, max_words=180, overlap_words=20):
    """Fallback splitter for very long or punctuation-poor text."""
    words = _normalize_whitespace(text).split()
    if not words:
        return []

    chunks = []
    step = max(1, max_words - overlap_words)
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks

def split_sentences(text):
    text = text.strip()
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def sentence_chunks(text, max_words=180, overlap_sents=2):
    """Accumulate sentences up to max_words; overlap last N sentences."""
    normalized = _normalize_whitespace(text)
    if not normalized:
        return []

    sentences = split_sentences(normalized)
    if len(sentences) <= 1 and len(normalized.split()) > max_words:
        return hard_word_chunks(normalized, max_words=max_words, overlap_words=20)

    if not sentences:
        return []
    chunks, current, current_words = [], [], 0
    for sent in sentences:
        sent_words = sent.split()
        wc = len(sent_words)
        if wc > max_words:
            if current:
                chunks.append(" ".join(current))
                current, current_words = [], 0
            chunks.extend(hard_word_chunks(sent, max_words=max_words, overlap_words=20))
            continue
        if current_words + wc > max_words and current:
            chunks.append(" ".join(current))
            current = current[-overlap_sents:]
            current_words = sum(len(s.split()) for s in current)
        current.append(sent)
        current_words += wc
    if current:
        chunks.append(" ".join(current))
    return chunks


def prepare_text_for_embedding(text, max_words=MAX_WORDS_PER_EMBED_INPUT):
    """Final guardrail before embedding requests."""
    normalized = _normalize_whitespace(text)
    if not normalized:
        return []
    if len(normalized.split()) <= max_words:
        return [normalized]
    return hard_word_chunks(normalized, max_words=max_words, overlap_words=40)


# ── Context header ─────────────────────────────────────────────────────────

def make_header(record):
    title   = record.get("page_title", "").strip()
    heading = record.get("heading", "").strip()
    url     = record.get("url", "").strip()
    parts = []
    if title:   parts.append(f"Page: {title}")
    if heading: parts.append(f"Section: {heading}")
    if url:     parts.append(f"URL: {url}")
    return "[" + " | ".join(parts) + "]\n" if parts else ""


# ── Table helpers ──────────────────────────────────────────────────────────

def table_to_nl(table, header):
    """Each row becomes a natural-language sentence chunk."""
    headers = table.get("headers", [])
    rows    = table.get("rows",    [])
    chunks  = []
    for row in rows:
        if len(row) != len(headers):
            continue
        pairs = " | ".join(f"{h}: {v}" for h, v in zip(headers, row) if str(v).strip())
        if pairs:
            chunks.append(header + pairs)
    return chunks


def table_to_dataframe(table):
    headers = table.get("headers", [])
    rows    = table.get("rows",    [])
    if not headers or not rows:
        return None
    try:
        return pd.DataFrame(rows, columns=headers)
    except Exception:
        return None


# ── Contact chunk ──────────────────────────────────────────────────────────

def contact_chunk(record, header):
    ci = record.get("contact_info", {})
    if not ci:
        return None
    parts = []
    if ci.get("emails"): parts.append("Emails: " + ", ".join(ci["emails"]))
    if ci.get("phones"): parts.append("Phones: " + ", ".join(ci["phones"]))
    if ci.get("address"): parts.append("Address: " + ci["address"])
    return header + "\n".join(parts) if parts else None


# ── Per-record chunk builder ───────────────────────────────────────────────

def chunks_from_record(record):
    """Return list of (chunk_text, metadata_dict) for one page record."""
    results  = []
    header   = make_header(record)
    source   = record.get("url", record.get("page_title", "unknown"))
    base_meta = {
        "source":     source,
        "url":        record.get("url", ""),
        "page_title": record.get("page_title", ""),
        "heading":    record.get("heading", ""),
    }

    # 1. main_content
    for item in record.get("main_content", []):
        if not isinstance(item, str) or len(item.strip()) < 20:
            continue
        for chunk in sentence_chunks(item, max_words=180, overlap_sents=2):
            results.append((header + chunk, {**base_meta, "chunk_type": "main"}))

    # 2. sections
    for sec_name, sec_items in record.get("sections", {}).items():
        sec_header = header + f"[Section: {sec_name}]\n"
        for item in sec_items:
            if not isinstance(item, str) or len(item.strip()) < 20:
                continue
            for chunk in sentence_chunks(item, max_words=180, overlap_sents=2):
                results.append((sec_header + chunk,
                                {**base_meta, "chunk_type": "section",
                                 "section_name": sec_name}))

    # 3. tables — row-level NL chunks
    for table in record.get("tables", []):
        for nl_chunk in table_to_nl(table, header):
            results.append((nl_chunk, {**base_meta, "chunk_type": "table_row"}))

    # 4. contact info
    cc = contact_chunk(record, header)
    if cc:
        results.append((cc, {**base_meta, "chunk_type": "contact"}))

    return results


def expand_embedding_chunks(chunk_records):
    """Ensure every stored chunk is safe to send to the embedding model."""
    expanded = []
    for chunk_text, meta in chunk_records:
        safe_chunks = prepare_text_for_embedding(chunk_text)
        if not safe_chunks:
            continue
        for idx, safe_chunk in enumerate(safe_chunks):
            safe_meta = dict(meta)
            if len(safe_chunks) > 1:
                safe_meta["subchunk_index"] = idx
                safe_meta["subchunk_total"] = len(safe_chunks)
            expanded.append((safe_chunk, safe_meta))
    return expanded


# ── DuckDB ingestion ───────────────────────────────────────────────────────

def ingest_tables_to_duckdb(all_data):
    conn    = get_duckdb_connection()
    schemas = {}
    for record in all_data:
        tables = record.get("tables", [])
        if not tables:
            continue
        url_slug = re.sub(r"[^a-z0-9]", "_",
                          record.get("url","").rstrip("/").split("/")[-1].lower())
        page_tag = url_slug or "page"
        for t_idx, table in enumerate(tables):
            df = table_to_dataframe(table)
            if df is None or df.empty:
                continue
            tname = (f"{page_tag}_{t_idx}" if len(tables) > 1 else page_tag)[:60]
            conn.execute(f'DROP TABLE IF EXISTS "{tname}"')
            conn.execute(f'CREATE TABLE "{tname}" AS SELECT * FROM df')
            col_info   = ", ".join(f"{c} (text)" for c in df.columns)
            sample_str = df.head(3).to_string(index=False)
            schemas[tname] = (
                f"From page: {record.get('page_title','')}\n"
                f"URL: {record.get('url','')}\n"
                f"Columns: {col_info}\nSample:\n{sample_str}"
            )
    conn.close()
    return schemas


# ── Embeddings ─────────────────────────────────────────────────────────────

def get_embeddings(client, texts):
    r = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [x.embedding for x in r.data]


def embed_in_batches(client, texts, progress_callback=None):
    all_embs, total = [], len(texts)
    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[i:i+EMBED_BATCH_SIZE]
        all_embs.extend(get_embeddings(client, batch))
        if progress_callback:
            progress_callback(
                f"Embedding {min(i+EMBED_BATCH_SIZE, total)}/{total} chunks...",
                min(i+EMBED_BATCH_SIZE, total) / total
            )
    return all_embs


# ── Master build ───────────────────────────────────────────────────────────

def build_indexes(api_key, progress_callback=None):
    client     = OpenAI(api_key=api_key)
    json_files = sorted(glob.glob(os.path.join(DATA_FOLDER, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No .json files in '{DATA_FOLDER}/'")

    # Load JSON
    all_data = []
    for fp in json_files:
        with open(fp, "r", encoding="utf-8") as f:
            raw = json.load(f)
        all_data.extend(raw if isinstance(raw, list) else [raw])

    if progress_callback:
        progress_callback(f"Loaded {len(all_data)} pages", 0.05)

    # Build chunks
    all_chunks, all_metas = [], []
    for record in all_data:
        for chunk_text, meta in expand_embedding_chunks(chunks_from_record(record)):
            all_chunks.append(chunk_text)
            all_metas.append(meta)

    if progress_callback:
        progress_callback(f"Created {len(all_chunks)} chunks", 0.10)

    # BM25
    bm25 = BM25Okapi([c.lower().split() for c in all_chunks])
    save_bm25(bm25, all_chunks, all_metas)

    # ChromaDB
    collection     = clear_chroma_collection()
    all_embeddings = embed_in_batches(client, all_chunks, progress_callback)

    ids   = [f"chunk_{i}" for i in range(len(all_chunks))]
    BATCH = 500
    for i in range(0, len(all_chunks), BATCH):
        safe_metas = [
            {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
             for k, v in m.items()}
            for m in all_metas[i:i+BATCH]
        ]
        collection.add(
            documents  = all_chunks[i:i+BATCH],
            embeddings = all_embeddings[i:i+BATCH],
            ids        = ids[i:i+BATCH],
            metadatas  = safe_metas,
        )

    if progress_callback:
        progress_callback("ChromaDB saved ✅", 0.90)

    # DuckDB
    db_schemas = ingest_tables_to_duckdb(all_data)
    save_hash(compute_data_hash())

    if progress_callback:
        progress_callback(f"Done — {len(db_schemas)} SQL table(s) ✅", 1.0)

    return dict(collection=collection, bm25=bm25, all_chunks=all_chunks,
                all_metas=all_metas, all_embeddings=all_embeddings,
                db_schemas=db_schemas, file_list=json_files)


def load_indexes_from_disk():
    from storage.store import get_chroma_collection, load_bm25, get_duckdb_schemas
    collection                  = get_chroma_collection()
    bm25, all_chunks, all_metas = load_bm25()
    db_conn                     = get_duckdb_connection()
    db_schemas                  = get_duckdb_schemas(db_conn)
    db_conn.close()
    result = collection.get(include=["embeddings"])
    embeddings = result.get("embeddings")
    if embeddings is None:
        all_embeddings = []
    elif hasattr(embeddings, "tolist"):
        all_embeddings = embeddings.tolist()
    else:
        all_embeddings = list(embeddings)
    return dict(collection=collection, bm25=bm25, all_chunks=all_chunks,
                all_metas=all_metas, all_embeddings=all_embeddings,
                db_schemas=db_schemas)
