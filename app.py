"""
app.py — FinSight Streamlit Application (Streamlit Cloud)

For Streamlit Cloud deployment:
    Push to GitHub and connect your repo to Streamlit Cloud

Run locally:
    streamlit run app.py
"""
import os
import sys

# ── DO NOT set offline mode yet - models need to download on first run ───────
# We'll enable it AFTER models are cached
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.expanduser('~/.cache/sentence-transformers')

# ── Pre-cache models if they don't exist (first run only) ────────────────────
def ensure_models_cached():
    """Check if models are cached, run setup if not
    
    On Streamlit Cloud: Downloads models on first app load (~10 min)
    On local: Uses cached models from setup_models.py
    
    CRITICAL: Only sets offline mode AFTER verifying all models are cached!
    """
    import os
    from pathlib import Path
    
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    # Check if models exist in cache
    models_needed = {
        'google/flan-t5-base': 'LLM (flan-t5)',
        'BAAI/bge-small-en-v1.5': 'Embeddings (bge)',
        'cross-encoder/ms-marco-MiniLM-L-6-v2': 'Re-ranker'
    }
    
    def count_cached_models():
        """Count how many models are actually cached"""
        if not os.path.exists(cache_dir):
            return 0
        
        try:
            contents = os.listdir(cache_dir)
            found = 0
            for model_name in models_needed:
                folder_prefix = 'models--' + model_name.replace('/', '--')
                # Check if folder exists AND is not empty (downloading)
                for item in contents:
                    if item.startswith(folder_prefix):
                        item_path = os.path.join(cache_dir, item)
                        # Make sure it's not just an incomplete download
                        if os.path.isdir(item_path) and os.listdir(item_path):
                            found += 1
                            print(f"[SETUP] ✅ Found {models_needed[model_name]}")
                            break
            return found
        except Exception as e:
            print(f"[SETUP] ⚠️  Could not check cache: {e}")
            return 0
    
    # CHECK 1: See what's already cached
    models_found = count_cached_models()
    
    # If ALL models found, enable offline mode and return
    if models_found == len(models_needed):
        print(f"[SETUP] ✅ All {len(models_needed)} models verified - enabling offline mode")
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        return True
    
    # DOWNLOAD PHASE: Some models missing
    print(f"\n[SETUP] ⏳ Missing {len(models_needed) - models_found} model(s)")
    print(f"[SETUP] Cache directory: {cache_dir}")
    print(f"[SETUP] Downloading missing models (this takes 10-15 minutes on first load)...\n")
    
    # CRITICAL: Disable offline mode to allow downloads
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'  # 10 min timeout per file
    
    try:
        os.makedirs(cache_dir, exist_ok=True)
        
        # Try to download FLAN-T5 if missing
        flan_t5_ok = False
        try:
            print(f"[SETUP] 📥 Step 1/3: Downloading FLAN-T5 LLM (~850MB, ~5 min)...")
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            AutoTokenizer.from_pretrained("google/flan-t5-base", trust_remote_code=True)
            AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", trust_remote_code=True)
            print(f"[SETUP] ✅ FLAN-T5 cached successfully\n")
            flan_t5_ok = True
        except Exception as e:
            print(f"[SETUP] ❌ FLAN-T5 download failed")
            print(f"    Error: {str(e)[:150]}\n")
        
        # Try to download BGE embeddings if missing
        bge_ok = False
        try:
            print(f"[SETUP] 📥 Step 2/3: Downloading BGE embeddings (~130MB, ~2 min)...")
            from sentence_transformers import SentenceTransformer
            SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu", trust_remote_code=True)
            print(f"[SETUP] ✅ BGE embeddings cached successfully\n")
            bge_ok = True
        except Exception as e:
            print(f"[SETUP] ❌ BGE embeddings download failed")
            print(f"    Error: {str(e)[:150]}\n")
        
        # Try to download re-ranker if missing
        reranker_ok = False
        try:
            print(f"[SETUP] 📥 Step 3/3: Downloading re-ranker (~85MB, ~1 min)...")
            from sentence_transformers.cross_encoder import CrossEncoder
            CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu", trust_remote_code=True)
            print(f"[SETUP] ✅ Re-ranker cached successfully\n")
            reranker_ok = True
        except Exception as e:
            print(f"[SETUP] ❌ Re-ranker download failed")
            print(f"    Error: {str(e)[:150]}\n")
        
        downloads_succeeded = sum([flan_t5_ok, bge_ok, reranker_ok])
        if downloads_succeeded > 0:
            print(f"[SETUP] ⚠️  PARTIAL SUCCESS - Downloaded {downloads_succeeded}/3 models")
        else:
            print(f"[SETUP] ❌ All 3 models failed to download")
        
    except Exception as e:
        print(f"[SETUP] ❌ Error during download phase: {e}\n")
    
    # VERIFICATION: Check if downloads succeeded
    print(f"[SETUP] Verifying downloaded models...\n")
    models_found_after = count_cached_models()
    
    # Track which models are missing
    missing_models = []
    if os.path.exists(cache_dir):
        try:
            contents = os.listdir(cache_dir)
            if not any(item.startswith('models--google--flan-t5-base') for item in contents if os.listdir(os.path.join(cache_dir, item))):
                missing_models.append('FLAN-T5 (LLM)')
            if not any(item.startswith('models--BAAI--bge-small-en-v1-5') for item in contents if os.listdir(os.path.join(cache_dir, item))):
                missing_models.append('BGE (embeddings)')
            if not any(item.startswith('models--cross-encoder--ms-marco-MiniLM-L-6-v2') for item in contents if os.listdir(os.path.join(cache_dir, item))):
                missing_models.append('Cross-encoder (re-ranker)')
        except:
            pass
    
    if models_found_after == len(models_needed):
        print(f"[SETUP] ✅ SUCCESS: All models downloaded and verified!")
        print(f"[SETUP] Enabling offline mode for subsequent runs...\n")
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        return True
    else:
        # Partial or complete failure
        print(f"[SETUP] ⚠️  Model cache verification:")
        print(f"[SETUP]    - {models_found_after}/{len(models_needed)} models cached")
        if missing_models:
            print(f"[SETUP]    - Missing: {', '.join(missing_models)}")
        print(f"[SETUP] Keeping online mode enabled for next load attempt...\n")
        os.environ['HF_HUB_OFFLINE'] = '0'
        os.environ['TRANSFORMERS_OFFLINE'] = '0'
        return False

# ── Call model check BEFORE importing pipeline ──────────────────────────────
try:
    ensure_models_cached()
except Exception as e:
    print(f"[SETUP] ⚠️  Model check failed: {e}")
    pass  # Continue anyway

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


# ── Global state ─────────────────────────────────────────────────────────────
pipeline = None
conversation_history = []

# ── Pipeline loader ──────────────────────────────────────────────────────────
def load_default_pipeline():
    """Load FinSight pipeline from pre-built dataset"""
    global pipeline
    if pipeline is not None:
        return  # Already loaded
    
    if not os.path.exists("chroma_db") or not os.path.exists("data/embeddings.npy"):
        error_msg = (
            "❌ **Setup required:** Pre-built data not found.\n\n"
            "Please run these setup commands in your terminal:\n\n"
            "```bash\n"
            "python src/ingest.py        # Process CSV data\n"
            "python src/embeddings.py    # Generate embeddings\n"
            "python src/vectorstore.py   # Build vector database\n"
            "```\n\n"
            "Then run Streamlit again: `streamlit run app.py`"
        )
        raise ValueError(error_msg)
    
    from pipeline import FinSightPipeline
    pipeline = FinSightPipeline()

# ── Query handler ────────────────────────────────────────────────────────────
def answer_question(question: str, strategy: str) -> tuple:
    """
    Process a question and return formatted answer with sources
    
    Returns:
        (answer_text, sources_markdown, chat_html)
    """
    global pipeline, conversation_history
    
    if not question.strip():
        return "", "❌ Please enter a question", ""
    
    try:
        # Load pipeline if not already loaded
        if pipeline is None:
            load_default_pipeline()
        
        # Query pipeline
        result = pipeline.query(question.strip(), strategy=strategy)
        
        answer = result["answer"]
        sources = result["sources"]
        strat = result["strategy_used"]
        
        # Format sources
        sources_md = f"**Strategy:** {strat}\n\n"
        if sources:
            sources_md += f"**Sources ({len(sources)} records):**\n\n"
            for i, src in enumerate(sources, 1):
                ticker = src.get("ticker", "N/A")
                year = src.get("year", "N/A")
                sentiment = src.get("sentiment_raw", "N/A")
                score = src.get("score", 0)
                gc_flag = " ⚠️ Going Concern" if src.get("going_concern") == "1" else ""
                text_preview = src.get("text", "")[:300] + "..."
                
                sources_md += (
                    f"{i}. **{ticker} ({year})**\n"
                    f"   - Relevance: {score:.3f}\n"
                    f"   - Sentiment: {sentiment}{gc_flag}\n"
                    f"   - {text_preview}\n\n"
                )
        else:
            sources_md += "❌ No relevant sources found. Try rephrasing with ticker + year."
        
        # Build conversation history for display
        conversation_history.append({"q": question, "a": answer, "strat": strat})
        chat_html = _build_chat_html()
        
        return answer, sources_md, chat_html
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        chat_html = _build_chat_html()
        return "", error_msg, chat_html

def _build_chat_html() -> str:
    """Format conversation history as HTML"""
    html = '<div style="max-height: 400px; overflow-y: auto; padding: 10px; background: #f5f5f5; border-radius: 8px;">'
    
    if not conversation_history:
        html += '<p style="color: #999; text-align: center;"><i>No conversation yet</i></p>'
    else:
        for turn in conversation_history:
            html += (
                f'<div style="margin-bottom: 12px; padding: 8px; background: white; border-radius: 4px;">'
                f'  <b>You:</b> {turn["q"]}<br>'
                f'  <b>FinSight ({turn["strat"]}):</b> {turn["a"][:200]}...'
                f'</div>'
            )
    
    html += '</div>'
    return html

# ── Streamlit UI ────────────────────────────────────────────────────────────

st.set_page_config(page_title="FinSight", layout="wide", initial_sidebar_state="expanded")

# Initialize session state for chat history
if "chat_html" not in st.session_state:
    st.session_state.chat_html = ""

st.markdown("""
# 📊 FinSight
**S&P 500 Financial Q&A powered by RAG**

Ask questions about S&P 500 companies — answers are cited directly from SEC 10-K financial data.
""")

st.markdown("---")

# Sidebar for settings
with st.sidebar:
    st.markdown("### How to use:")
    st.markdown("""
    1. Enter your question below
    2. Select retrieval strategy
    3. Press "Ask FinSight"
    4. View answer with source citations
    """)
    
    st.markdown("### Example questions:")
    st.markdown("""
    - "What was GOOG's net margin in 2023?"
    - "Which companies had a going concern warning?"
    - "Did ETR's net income grow despite falling revenue in 2023?"
    - "What was BRO's revenue growth in 2025?"
    - "Which company had the most negative executive sentiment?"
    """)

# Main content area
st.markdown("### Your Question")

search_col, strategy_col = st.columns([3, 1])

with search_col:
    question_input = st.text_area(
        "Ask about S&P 500 companies",
        placeholder="e.g., What was GOOG's net margin in 2023?",
        height=100,
        label_visibility="collapsed"
    )

with strategy_col:
    strategy = st.selectbox(
        "Strategy",
        options=["dense", "sparse","hybrid"],
        index=0,
        help="hybrid: BM25 + semantic + re-ranker (best) | dense: semantic only | sparse: keyword only"
    )
    st.markdown("")
    submit_btn = st.button("🔍 Ask FinSight", use_container_width=True, type="primary")

st.markdown("---")

# Results display
if submit_btn or "last_question" in st.session_state:
    if submit_btn and question_input.strip():
        st.session_state.last_question = question_input
        
        with st.spinner("Searching and generating answer..."):
            answer_text, sources_md, chat_html = answer_question(question_input, strategy)
            st.session_state.chat_html = chat_html
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Answer")
            st.markdown(answer_text if answer_text else "*Waiting for result...*")
        
        with col2:
            st.markdown("### Sources")
            st.markdown(sources_md)
    
    # Conversation history
    if "last_question" in st.session_state:
        st.markdown("---")
        st.markdown("### Conversation History")
        st.markdown(st.session_state.chat_html, unsafe_allow_html=True)

st.markdown("---")

with st.expander("📚 About FinSight"):
    st.markdown("""
    ### Technology Stack (100% free, runs locally, offline-first)
    - **LLM:** google/flan-t5-base (local inference, no API calls)
    - **Embeddings:** BAAI/bge-small-en-v1.5 (384-dim, cached locally)
    - **Vector DB:** ChromaDB (1,487 S&P 500 records)
    - **Retrieval:** BM25 keyword search + dense semantic search + RRF fusion
    - **Re-ranker:** cross-encoder/ms-marco-MiniLM-L-6-v2 (CPU-only)
    
    ### Pipeline
    ```
    Question → BM25 top-20 + Dense top-20 → RRF fusion → top-10 
    → Cross-encoder re-rank (CPU) → top-3 → T5 generation → Answer
    ```
    
    ### Dataset: S&P 500 Alpha (Kaggle)
    - 1,487 company-year records
    - Financial ratios: Net Margin, ROA, Revenue Growth, Net Income Growth
    - NLP signals: Executive sentiment, MD&A complexity, going concern flags
    - Source: SEC EDGAR + XBRL APIs
    
    ### Features
    - ✅ 100% offline after setup (no internet required)
    - ✅ No API keys or expensive cloud services
    - ✅ All data stays on your machine
    - ✅ CPU-only inference (works on any hardware)
    """)