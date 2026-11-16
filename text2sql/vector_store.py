"""Chroma store: one chunk (document) per table, containing the table's full JSON metadata.

NOTE: Not used by the application (kept for reference only). Table metadata is now read from the
in-memory dict returned by metadata.load_table_details(). Requires `chromadb` to be installed.
"""
import json
from functools import lru_cache

import chromadb
from langchain_ollama import OllamaEmbeddings

from . import config
from .metadata import load_table_details


@lru_cache
def _embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL)


@lru_cache
def _collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(config.CHROMA_COLLECTION,
                                           metadata={"hnsw:space": "cosine"})


def _embedding_text(table: dict) -> str:
    # nomic-embed-text expects task prefixes
    cols = "; ".join(f"{c['name']} ({c['type']}): {c.get('description', '')}" for c in table["columns"])
    return (f"search_document: Table {table['table_name']} in module {table.get('module', '')}. "
            f"{table.get('description', '')} Columns: {cols}")


def ingest(reset: bool = False) -> int:
    """Embeds every table metadata JSON file into Chroma (upsert by table name)."""
    col = _collection()
    tables = load_table_details()
    if reset:
        existing = col.get(include=[])["ids"]
        if existing:
            col.delete(ids=existing)
    if not tables:
        return 0
    names = list(tables)
    col.upsert(
        ids=names,
        documents=[json.dumps(tables[n], ensure_ascii=False) for n in names],
        embeddings=_embeddings().embed_documents([_embedding_text(tables[n]) for n in names]),
        metadatas=[{"table_name": n, "module": tables[n].get("module", "").lower()} for n in names],
    )
    return len(names)


def ensure_ingested() -> None:
    if _collection().count() == 0:
        print(f"[vector_store] ingested {ingest()} tables into Chroma")


def get_tables(names: list[str]) -> dict[str, dict]:
    """Exact lookup of full table metadata by table name."""
    if not names:
        return {}
    res = _collection().get(ids=names, include=["documents"])
    return {i: json.loads(d) for i, d in zip(res["ids"], res["documents"])}


def search_tables(question: str, modules: list[str] | None = None, k: int = 4) -> dict[str, dict]:
    """Semantic search fallback, optionally restricted to the given modules."""
    col = _collection()
    if col.count() == 0:
        return {}
    where = None
    if modules:
        mods = [m.lower() for m in modules]
        where = {"module": mods[0]} if len(mods) == 1 else {"module": {"$in": mods}}
    res = col.query(
        query_embeddings=[_embeddings().embed_query(f"search_query: {question}")],
        n_results=min(k, col.count()),
        where=where,
        include=["documents"],
    )
    return {i: json.loads(d) for i, d in zip(res["ids"][0], res["documents"][0])}
