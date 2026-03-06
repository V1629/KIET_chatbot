import glob
import hashlib
import os
import pickle
from typing import Tuple

import chromadb
import duckdb

from config.settings import BM25_PATH, CHROMA_DIR, DATA_FOLDER, DUCKDB_PATH, HASH_PATH


def _ensure_storage_dirs() -> None:
    os.makedirs(os.path.dirname(BM25_PATH) or ".", exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)


def compute_data_hash() -> str:
    """Return a stable hash of all JSON files under the data directory."""
    digest = hashlib.sha256()
    json_files = sorted(glob.glob(os.path.join(DATA_FOLDER, "*.json")))
    for path in json_files:
        digest.update(os.path.basename(path).encode("utf-8"))
        with open(path, "rb") as file_obj:
            while True:
                chunk = file_obj.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def save_hash(data_hash: str) -> None:
    _ensure_storage_dirs()
    with open(HASH_PATH, "w", encoding="utf-8") as file_obj:
        file_obj.write(data_hash)


def data_changed() -> bool:
    current_hash = compute_data_hash()
    if not os.path.exists(HASH_PATH):
        return True
    try:
        with open(HASH_PATH, "r", encoding="utf-8") as file_obj:
            stored_hash = file_obj.read().strip()
    except OSError:
        return True
    return stored_hash != current_hash


def indexes_exist() -> bool:
    return (
        os.path.exists(BM25_PATH)
        and os.path.exists(DUCKDB_PATH)
        and os.path.isdir(CHROMA_DIR)
        and any(os.scandir(CHROMA_DIR))
        and os.path.exists(HASH_PATH)
    )


def save_bm25(bm25, all_chunks, all_metas) -> None:
    _ensure_storage_dirs()
    payload: Tuple[object, list, list] = (bm25, all_chunks, all_metas)
    with open(BM25_PATH, "wb") as file_obj:
        pickle.dump(payload, file_obj)


def clear_chroma_collection(name: str = "rag_chunks"):
    _ensure_storage_dirs()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.get_or_create_collection(name)


def get_chroma_collection(name: str = "rag_chunks"):
    _ensure_storage_dirs()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name)


def get_duckdb_connection():
    _ensure_storage_dirs()
    return duckdb.connect(DUCKDB_PATH)


def load_bm25():
    with open(BM25_PATH, "rb") as file_obj:
        bm25, all_chunks, all_metas = pickle.load(file_obj)
    return bm25, all_chunks, all_metas


def get_duckdb_schemas(conn) -> dict:
    rows = conn.execute("SHOW TABLES").fetchall()
    schemas = {}
    for row in rows:
        table_name = row[0]
        columns = conn.execute(f'DESCRIBE "{table_name}"').fetchall()
        formatted_columns = ", ".join(f"{col[0]} ({col[1]})" for col in columns)
        sample_df = conn.execute(f'SELECT * FROM "{table_name}" LIMIT 3').df()
        sample_str = sample_df.to_string(index=False) if not sample_df.empty else "<empty>"
        schemas[table_name] = f"Columns: {formatted_columns}\nSample:\n{sample_str}"
    return schemas
