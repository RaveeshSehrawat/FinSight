"""
pipeline.py — The heart of FinSight.

This is the single file that ties everything together.
Every other module (ingest, embeddings, vectorstore, retrieval, generator)
is loaded here and wired into one clean rag_query() function.

Usage:
    from pipeline import FinSightPipeline
    pipeline = FinSightPipeline()
    result = pipeline.query("What was Apple's revenue in 2023?")
    print(result["answer"])
"""

import os
import json
import pickle
import datetime
from pathlib import Path

from embeddings import load_embedding_model
from vectorstore import load_vectorstore
from retrieval  import build_bm25_index, retrieve, extract_company_ticker
from generator  import load_llm, generate_answer


# ─────────────────────────────────────────────
# FINSIGHT PIPELINE CLASS
# ─────────────────────────────────────────────

class FinSightPipeline:
    """
    Wraps all RAG components into a single object.

    On init: loads embedding model, vector store, BM25 index, and LLM.
    On query: runs retrieve → rerank → generate and returns result.

    Usage:
        pipeline = FinSightPipeline()
        result   = pipeline.query("What was Apple's revenue?")
    """

    def __init__(
        self,
        chunks_path  : str = "data/chunks.pkl",
        log_path     : str = "eval/query_log.jsonl",
        llm_model    : str = "google/flan-t5-base",
        strategy     : str = "hybrid",
    ):
        print("\n Initialising FinSight pipeline ...")

        # Load chunks (needed for BM25)
        print(" Loading chunks ...")
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        # Load models
        self.embedding_model = load_embedding_model()
        self.vectorstore     = load_vectorstore(self.embedding_model)
        self.bm25_index      = build_bm25_index(self.chunks)
        self.llm             = load_llm(llm_model)

        self.default_strategy = strategy
        self.log_path         = log_path

        # Ensure eval directory exists for logging
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        print(f"\n FinSight ready.")
        print(f"  Chunks loaded   : {len(self.chunks)}")
        print(f"  Default strategy: {strategy}")
        print(f"  LLM             : {llm_model} (local transformers, offline mode)\n")


    def query(self, question: str, strategy: str = None) -> dict:
        """
        Full RAG pipeline: retrieve → rerank → generate.

        Args:
            question : the user's question
            strategy : "dense" | "sparse" | "hybrid" (overrides default)

        Returns dict with:
            answer          : the LLM's answer string
            sources         : list of source chunks used
            strategy_used   : which retrieval strategy was used
            retrieved_chunks: raw retrieval results (for evaluation)
        """
        strategy = strategy or self.default_strategy

        # Extract company ticker from question (if mentioned by name or symbol)
        ticker = extract_company_ticker(question)
        if ticker:
            print(f"\n [Company Filter] Detected: {ticker}")

        # ── Step 1: Retrieve top chunks ──────────────────────────────────
        retrieved = retrieve(
            query       = question,
            vectorstore = self.vectorstore,
            chunks      = self.chunks,
            bm25_index  = self.bm25_index,
            strategy    = strategy,
            final_k     = 3,
            ticker      = ticker,
        )

        # ── Step 2: Generate answer from Mistral ─────────────────────────
        result = generate_answer(
            question         = question,
            retrieved_chunks = retrieved,
            llm              = self.llm,
        )

        # ── Step 3: Attach metadata ───────────────────────────────────────
        result["strategy_used"]    = strategy
        result["question"]         = question
        result["retrieved_chunks"] = retrieved

        # ── Step 4: Log to JSONL for evaluation ──────────────────────────
        self._log(result)

        return result


    def _log(self, result: dict) -> None:
        """
        Appends query + result to eval/query_log.jsonl.
        This log becomes your evaluation dataset for Phase 6.
        """
        log_entry = {
            "timestamp"         : datetime.datetime.now().isoformat(),
            "question"          : result["question"],
            "answer"            : result["answer"],
            "strategy_used"     : result["strategy_used"],
            "skipped_generation": result.get("skipped_generation", False),
            "sources"           : result.get("sources", []),
            "retrieved_contexts": [
                r["doc"].page_content for r in result.get("retrieved_chunks", [])
            ],
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")


# ─────────────────────────────────────────────
# MAIN — end-to-end test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = FinSightPipeline()

    test_questions = [
        "What was GOOG's net profit margin and revenue growth in 2023?",
        "Which company had the most negative executive sentiment?",
        "What was BRO's revenue growth in 2025?",
        "Does ARE have a going concern warning?",
        "Did ETR's net income grow despite falling revenue in 2023?",
    ]

    print("\n" + "═" * 65)
    print(" FINSIGHT — END-TO-END TEST")
    print("═" * 65)

    for question in test_questions:
        print(f"\n Question: {question}")
        print("─" * 65)

        result = pipeline.query(question, strategy="hybrid")

        print(f" Answer:\n{result['answer']}")
        print(f"\n Sources used:")
        for s in result["sources"]:
            # CORRECT — matches your actual metadata keys
            print(f"   - {s['filename']} | {s['ticker']} {s['year']}")

    print(f"\n\n All queries logged to eval/query_log.jsonl")
    print(" Phase 5 complete. Run eval/run_eval.py next.")