"""
src/vectorstore.py — Phase 3b
Builds and loads the ChromaDB vector store.

Run once to build:
    python src/vectorstore.py

All subsequent phases call load_vectorstore() — never build again.
"""
import os
import pickle
from langchain_chroma import Chroma
from embeddings import load_embedding_model, EmbeddingAdapter

CHROMA_DIR = "chroma_db"


# ─────────────────────────────────────────────
# 1. BUILD (run once)
# ─────────────────────────────────────────────

def build_vectorstore(chunks: list, model: EmbeddingAdapter) -> Chroma:
    """
    Creates ChromaDB vector store from chunks.
    Automatically embeds, stores vectors + text + metadata, persists to disk.
    Run this ONCE — use load_vectorstore() every time after.
    """
    print(f"\n Building ChromaDB from {len(chunks)} chunks ...")
    print(f" Persisting to: {CHROMA_DIR}/")

    if os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print(" Removed existing chroma_db/ — rebuilding from scratch")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=model,
        persist_directory=CHROMA_DIR,
        collection_name="finsight",
    )

    count = vectorstore._collection.count()
    print(f" Vector store built. Total vectors: {count}")
    _print_stats(vectorstore)
    return vectorstore


# ─────────────────────────────────────────────
# 2. LOAD (use every time after first build)
# ─────────────────────────────────────────────

def load_vectorstore(model: EmbeddingAdapter) -> Chroma:
    """Loads existing ChromaDB from disk. Use this in all phases after Phase 3."""
    print(f"\n Loading ChromaDB from {CHROMA_DIR}/ ...")

    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            "chroma_db/ not found. Run vectorstore.py first to build it."
        )

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=model,
        collection_name="finsight",
    )
    count = vectorstore._collection.count()
    print(f" Loaded. Total vectors: {count}")
    return vectorstore


# ─────────────────────────────────────────────
# 3. FILTERED SEARCH (bonus — filter by ticker or year)
# ─────────────────────────────────────────────

def filtered_search(vectorstore: Chroma, query: str,
                    ticker: str = None, year: str = None, k: int = 5) -> list:
    """
    Search only within a specific ticker or year.
    Example: filtered_search(vs, "net margin", ticker="AAPL", year="2023")
    Shows production awareness — most portfolio RAGs can't filter by metadata.
    """
    where = {}
    if ticker:
        where["ticker"] = ticker.upper()
    if year:
        where["year"] = str(year)

    if where:
        return vectorstore.similarity_search_with_score(query, k=k, filter=where)
    return vectorstore.similarity_search_with_score(query, k=k)


# ─────────────────────────────────────────────
# 4. SANITY CHECK
# ─────────────────────────────────────────────

def sanity_check(vectorstore: Chroma) -> None:
    """
    Runs 3 test queries and prints results.
    Run after building — confirm retrieved chunks look relevant.
    """
    test_queries = [
        "What was GOOG's net margin in 2023?",
        "Which company had a going concern warning?",
        "What was the executive sentiment for CPB?",
    ]

    print(f"\n {'─'*60}")
    print(f"  SANITY CHECK — 3 test queries")
    print(f"  {'─'*60}")

    for query in test_queries:
        print(f"\n  Query: '{query}'")
        results = vectorstore.similarity_search_with_score(query, k=3)
        for j, (doc, score) in enumerate(results):
            print(f"  [{j+1}] score={score:.3f} | "
                  f"ticker={doc.metadata.get('ticker','?')} | "
                  f"year={doc.metadata.get('year','?')} | "
                  f"chunk_id={doc.metadata.get('chunk_id','?')}")
            print(f"       {doc.page_content[:120].strip()} ...")
    print(f"\n  {'─'*60}")


# ─────────────────────────────────────────────
# 5. STATS
# ─────────────────────────────────────────────

def _print_stats(vectorstore: Chroma) -> None:
    """Prints vector store stats — copy into your README."""
    count = vectorstore._collection.count()
    db_size_mb = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(CHROMA_DIR)
        for f in files
    ) / (1024 * 1024)

    print(f"\n  ── Vector Store Stats ──")
    print(f"  Total vectors   : {count}")
    print(f"  Embedding dim   : 384  (BAAI/bge-small-en-v1.5)")
    print(f"  DB size on disk : {db_size_mb:.1f} MB")
    print(f"  Location        : {CHROMA_DIR}/")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    CHUNKS_PATH = "data/chunks.pkl"

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    print(f" Loaded {len(chunks)} chunks")

    model       = load_embedding_model()
    vectorstore = build_vectorstore(chunks, model)
    sanity_check(vectorstore)

    print("\n Phase 3b complete. Run retrieval.py next.")