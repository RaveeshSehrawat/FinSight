"""
src/embeddings.py — Phase 3a
Loads BAAI/bge-small-en-v1.5, embeds all chunks, saves vectors to disk.

Run from finsight/ root:
    python src/embeddings.py
"""

import os
import time
import numpy as np
from tqdm import tqdm

# Disable HuggingFace network access globally (for offline/Streamlit deployment)
# These must be set BEFORE importing transformers or sentence_transformers
# But ONLY if they haven't already been set to 0 by app.py during model download
if os.environ.get('HF_HUB_OFFLINE') != '0':
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
if os.environ.get('TRANSFORMERS_OFFLINE') != '0':
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

from sentence_transformers import SentenceTransformer


# Wrapper to make SentenceTransformer compatible with LangChain APIs
class EmbeddingAdapter:
    """Wraps SentenceTransformer to be compatible with LangChain Chroma/RAGAS."""
    
    def __init__(self, model: SentenceTransformer):
        self.model = model
    
    def embed_documents(self, texts: list) -> list:
        """LangChain API: embed list of documents."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> list:
        """LangChain API: embed a single query."""
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()


# ─────────────────────────────────────────────
# 1. LOAD THE EMBEDDING MODEL
# ─────────────────────────────────────────────

def load_embedding_model() -> EmbeddingAdapter:
    """
    Loads BAAI/bge-small-en-v1.5 from local HuggingFace cache.
    Attempts to download if not cached and online mode is enabled.

    Why bge-small-en-v1.5?
      - Top-ranked on MTEB leaderboard for its size class
      - 384-dimensional vectors (compact but powerful)
      - Fast on CPU — no GPU needed
    """
    print("\n Loading embedding model: BAAI/bge-small-en-v1.5 ...")

    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_name = "BAAI/bge-small-en-v1.5"
    
    try:
        # First, try loading from local cache only (if offline mode is on)
        if os.environ.get('HF_HUB_OFFLINE') == '1' or os.environ.get('TRANSFORMERS_OFFLINE') == '1':
            print("   Attempting to load from cache (offline mode)...")
            model = SentenceTransformer(
                model_name,
                device="cpu",
                cache_folder=cache_dir,
                trust_remote_code=False,
                local_files_only=True,  # Strict cache-only mode
            )
            print(f" [OK] Model loaded from cache")
            return EmbeddingAdapter(model)
    except Exception as e1:
        offline_error = str(e1)
        print(f"   Cache load failed: {str(e1)[:150]}")
        
        # If cache load failed, try downloading (if online mode is allowed)
        if os.environ.get('HF_HUB_OFFLINE') != '1' and os.environ.get('TRANSFORMERS_OFFLINE') != '1':
            print(f"   Online mode enabled - attempting download...")
            try:
                model = SentenceTransformer(
                    model_name,
                    device="cpu",
                    cache_folder=cache_dir,
                    trust_remote_code=False,
                )
                print(f" [OK] Model downloaded and cached")
                return EmbeddingAdapter(model)
            except Exception as e2:
                print(f"   Download also failed: {str(e2)[:150]}")
                raise RuntimeError(
                    f"\n❌ EMBEDDING MODEL LOAD FAILED\n"
                    f"Model: {model_name}\n"
                    f"Cache: {cache_dir}\n"
                    f"Error: {str(e2)[:200]}\n\n"
                    f"Solutions:\n"
                    f"1. On Streamlit Cloud: Wait 15 min for model download, check 'Logs' tab\n"
                    f"2. Locally: Run 'python setup_models.py' to download models\n"
                    f"3. Check internet connection and firewall settings\n"
                ) from e2
        else:
            # Offline but cache missing
            raise RuntimeError(
                f"\n❌ EMBEDDING MODEL NOT CACHED\n"
                f"Model: {model_name}\n"
                f"Cache dir: {cache_dir}\n"
                f"Offline mode: ON (cannot download)\n\n"
                f"Solutions:\n"
                f"1. Locally: Run 'python setup_models.py'\n"
                f"2. Streamlit Cloud: Redeploy (models download on first load)\n"
                f"3. Check ~/.cache/huggingface/hub/ directory exists and is writable\n"
            ) from e1
        raise RuntimeError(
            f"Failed to load embedding model BAAI/bge-small-en-v1.5\n"
            f"This model must be cached before running the app.\n"
            f"Run: python setup_models.py"
        )


# ─────────────────────────────────────────────
# 2. EMBED ALL CHUNKS IN BATCHES
# ─────────────────────────────────────────────

def embed_chunks(chunks: list, model: EmbeddingAdapter, batch_size: int = 32) -> np.ndarray:
    """
    Embeds all chunks in batches of batch_size.

    Returns numpy array of shape (num_chunks, 384).
    Batching keeps memory flat regardless of corpus size.
    """
    print(f"\n Embedding {len(chunks)} chunks in batches of {batch_size} ...")
    start = time.time()

    all_texts      = [chunk.page_content for chunk in chunks]
    batch_embeddings = model.model.encode(
        all_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    elapsed = time.time() - start
    vectors = np.array(batch_embeddings, dtype=np.float32)

    print(f" Embedded {len(chunks)} chunks in {elapsed:.1f}s")
    print(f" Embedding matrix shape: {vectors.shape}")
    return vectors


# ─────────────────────────────────────────────
# 3. SAVE & LOAD VECTORS
# ─────────────────────────────────────────────

def save_embeddings(vectors: np.ndarray, path: str = "data/embeddings.npy") -> None:
    """Saves embedding vectors to disk. Never re-embed unless corpus changes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, vectors)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f" Saved embeddings to {path} ({size_mb:.1f} MB)")

def load_embeddings(path: str = "data/embeddings.npy") -> np.ndarray:
    """Loads previously saved embedding vectors from disk."""
    vectors = np.load(path)
    print(f" Loaded embeddings from {path} — shape: {vectors.shape}")
    return vectors


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import pickle

    CHUNKS_PATH     = "data/chunks.pkl"
    EMBEDDINGS_PATH = "data/embeddings.npy"

    print(" Loading chunks from Phase 2 ...")
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    print(f" Loaded {len(chunks)} chunks")

    model   = load_embedding_model()
    vectors = embed_chunks(chunks, model, batch_size=32)
    save_embeddings(vectors, EMBEDDINGS_PATH)

    print("\n Phase 3a complete. Run vectorstore.py next.")