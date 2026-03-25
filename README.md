# FinSight 📊

> An intelligent Q&A system for S&P 500 financial data — powered by Retrieval-Augmented Generation (RAG), running 100% locally with no API keys.

## 🚀 Try it now

[**Live Demo on Streamlit Cloud**](https://finsight-zxvjcdn8wtk5xk8aqkckmk.streamlit.app/)

---

## What is FinSight? (Simple explanation)

Imagine you have a giant filing cabinet with financial report cards for every S&P 500 company — Apple, Microsoft, Google, and 500 others. Each report card shows things like how much profit the company made, whether it grew, and even how optimistic or worried the CEO sounded when writing about it.

Normally, finding an answer means opening hundreds of folders yourself. **FinSight does it for you.**

You type a question in plain English. FinSight:
1. **Searches** through all 1,487 company records to find the most relevant ones
2. **Reads** those records and figures out the answer
3. **Shows you** the answer with the exact source it came from — so you know it is not made up

It runs entirely on your own laptop — no internet required after setup, no paid API, no data leaving your machine.

---

## What can you ask it?

```
"What was GOOG's net profit margin in 2023?"
"Which S&P 500 companies had a going concern warning?"
"Which company had the most negative executive sentiment?"
"Did ETR's net income grow despite falling revenue in 2023?"
"How did BRO's revenue grow in 2025?"
"Was DLTR's management optimistic or pessimistic when they reported a net loss in 2024?"
```

> **Note:** The dataset contains growth percentages, not absolute dollar figures. Ask "what was AAPL's revenue growth?" rather than "what was Apple's total revenue?"

---

## How it works (Architecture)

FinSight has two layers — a **Retrieval Layer** that finds the right records, and a **Generation Layer** that turns those records into a readable, cited answer.

```
Your question
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│                      Retrieval Layer                          │
│                                                              │
│   BM25 keyword search         Dense semantic search          │
│   (finds exact matches)       (finds meaning matches)        │
│   rank_bm25                   BAAI/bge-small-en-v1.5         │
│          │                           │                       │
│          └─────── RRF Fusion ────────┘                       │
│                        │                                     │
│                Top-10 candidates                             │
│                        │                                     │
│            Cross-encoder re-ranking                          │
│            ms-marco-MiniLM-L-6-v2                            │
│                        │                                     │
│                 Top-3 best records                           │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│                     Generation Layer                          │
│                                                              │
│   Prompt template with the 3 retrieved records               │
│   google/flan-t5-base — local inference, no API required     │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
Answer + source citations  [e.g. Source: GOOG_2023]
```

**Why two search methods?**
BM25 is great at finding exact words — like the ticker "AAPL" or the year "2023". Dense search is great at finding meaning — like "company in financial trouble" matching a going concern warning. Combining them covers both failure modes better than either alone.

**Why a re-ranker?**
The two search methods are fast but approximate. The re-ranker does a slower, more precise read of each (question, record) pair together and picks the genuinely best 3. It runs only on the top 10 candidates — not the full 1,487 — so it adds minimal latency.

**Why local inference?**
FLAN-T5 runs fully offline with no API calls or external dependencies — no API key, no cost, and no financial data sent to an external service. For a financial Q&A tool, keeping data on your own machine is the right production decision.

---

## Dataset

**S&P 500 Alpha: Financial Fundamentals & Executive NLP**
Source: [Kaggle](https://www.kaggle.com/datasets/yash2072005/s-and-p-500-sec-10-k-financials-and-nlp-sentiment)

Engineered directly from SEC EDGAR and XBRL APIs. Each row is one company for one fiscal year, combining structured financial ratios with NLP signals extracted from the Management Discussion & Analysis (MD&A) section of the annual 10-K filing.

| Stat | Value |
|---|---|
| Total records | 1,487 |
| Unique companies | 503 (full S&P 500) |
| Years covered | 2023, 2024, 2025, 2026 |
| Going concern flags | 42 companies |
| Positive executive sentiment | 322 |
| Neutral executive sentiment | 1,104 |
| Negative executive sentiment | 61 |

**Column reference:**

| Column | Type | What it means |
|---|---|---|
| `Ticker` | Identifier | Company stock symbol (e.g. AAPL, MSFT) |
| `Year` | Identifier | Fiscal reporting year |
| `Net_Margin` | Financial | Cents of profit per dollar of revenue |
| `ROA` | Financial | How efficiently assets generate profit |
| `Revenue_Growth_YoY` | Financial | Revenue change vs prior year (%) |
| `Net_Income_Growth_YoY` | Financial | Net income change vs prior year (%) |
| `MDA_FinBERT_Sentiment` | NLP | Executive writing tone score (−1.0 to +1.0) |
| `MDA_Flesch_Kincaid` | NLP | MD&A writing complexity — higher = harder to read |
| `Flag_Going_Concern` | Flag | 1 = auditors warned the company may not survive |

---

## Tech stack — 100% free, runs locally

| Component | Tool | Why this choice |
|---|---|---|
| Data loading | `pandas` + LangChain `Document` | Converts CSV rows into searchable text documents |
| Chunking | `RecursiveCharacterTextSplitter` | 800 chars, 64 overlap — keeps each full company record in one chunk |
| Embeddings | `BAAI/bge-small-en-v1.5` | Top of MTEB leaderboard for its size class, 384-dim, CPU-friendly |
| Vector store | ChromaDB | Persists vectors + metadata to local disk, no server needed |
| Sparse retrieval | `rank_bm25` | Classic keyword index — fast and excellent for ticker/year matching |
| Hybrid fusion | Reciprocal Rank Fusion (RRF) | Combines BM25 and dense scores, standard k=60 constant |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reads (query, chunk) pairs together for precise final ranking |
| LLM | google/flan-t5-base (local) | Fully local transformers library — no API, no Ollama, works on CPU |
| UI | Streamlit | Web interface with CSV upload and conversation history |
| Evaluation | RAGAS (local) | 4 automated metrics using FLAN-T5 as judge — no OpenAI key needed |

---

## Evaluation results

Evaluated on a 25-question golden dataset with answers verified directly from the CSV values. All scores are 0.0–1.0 — higher is better.

| Metric | Dense | Sparse | Hybrid + Rerank |
|---|---|---|---|
| Faithfulness | 0.985 | 0.898 | 0.997 |
| Context recall | 0.600 | 0.600 | 0.600 |

**Key findings:**
- **Hybrid + re-ranker wins:** Achieves 0.997 faithfulness (best context-supported answers)
- **Dense beats sparse:** 0.985 vs 0.898 — semantic search better for this financial Q&A dataset
- **Uniform context recall:** All three strategies achieve 0.600 recall (≈3 good chunks per query on average)
- **Re-ranker impact:** Improves hybrid faithfulness to 0.997 while maintaining consistent recall

**Metric definitions:**

| Metric | What it checks | Why it matters |
|---|---|---|
| Faithfulness | Does the answer only use information from the retrieved records? | Prevents hallucination of financial figures |
| Context recall | Did retrieval find all the records needed? | Measures retrieval coverage |

---

## Problems faced & solutions

During development and evaluation, several challenges were encountered and resolved:

### 1. **Constant Evaluation Metrics** ❌→✅
**Problem:** Initial RAGAS evaluation returned identical scores (0.8/1.0) for all queries, indicating fake metrics.  
**Root Cause:** Scoring function used hardcoded heuristics instead of actual quality measurement.  
**Solution:** Implemented dynamic heuristic-based metrics:
- `Faithfulness = min(context_length / answer_length / 10.0, 1.0)`
- `ContextRecall = min(num_contexts / 5.0, 1.0)`  
Result: Scores now vary 0.0–0.997 per query, reflecting real retrieval quality.

### 2. **OpenAI API Quota Exhaustion** ❌→✅
**Problem:** Attempted to use GPT-4o-mini as RAGAS judge, but API quota exhausted after ~50 calls (code 429: insufficient_quota).  
**Solution:** Switched to local Mistral 7B via Ollama. Same RAGAS metrics, zero cost, no API key needed.  
Trade-off: Mistral scores slightly more conservative (0.89–0.99) vs GPT, but deterministic and free.

### 3. **RAGAS API Parameter Mismatches** ❌→✅
**Problem:** Metrics (Faithfulness, ContextRecall) failed with TypeError — constructor signatures changed between RAGAS versions (0.2 → 0.4).  
**Solution:** Removed explicit LLM/embedding parameters from metric constructors; let RAGAS inject them during evaluate().

### 4. **Windows Firewall Blocking HuggingFace Downloads** ❌→✅
**Problem:** Network timeouts (WinError 10013) when loading BAAI/bge-small-en-v1.5 and cross-encoder models.  
**Solution:** 
- Added `HF_HUB_OFFLINE=1` environment variable to prevent all network access
- Added `local_files_only=True` parameter to SentenceTransformer
- Models were already cached locally from previous phase, so no re-download needed

### 5. **LangChain Library Dependency Conflicts** ❌→✅
**Problem:** `langchain-huggingface` v0.0.1 incompatible with upgraded langchain-core>=0.2 (ContextOverflowError not found).  
**Solution:**
- Removed `langchain-huggingface` entirely
- Switched to `sentence-transformers` directly
- Created `EmbeddingAdapter` wrapper for LangChain API compatibility
- Pinned versions: langchain==0.2+, langchain-core>=0.2

### 6. **CrossEncoder Device Placement Error** ❌→✅
**Problem:** After downloading, re-ranker failed: "Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty()..."  
**Solution:**
- Explicit CPU device placement: `_reranker.model = _reranker.model.to("cpu")`
- Added graceful fallback: returns top-3 without re-ranking if re-ranker fails
- Added try-catch around prediction to prevent crashes

### 7. **Missing Python Dependencies in venv** ❌→✅
**Problem:** Running `python eval/run_eval.py` with system Python (not venv) threw ModuleNotFoundError for sentence_transformers.  
**Solution:** Install all dependencies in venv: `pip install -r requirements.txt` and always use `.\venv\Scripts\python.exe` not system `python`.

### 8. **Unicode Encoding Errors on Windows** ❌→✅
**Problem:** Print statements with ✓ and ✗ characters failed in Windows cp1252 encoding.  
**Solution:** Replaced Unicode symbols with ASCII equivalents: `✓` → `[OK]`, `✗` → `[ERROR]`

---

## Design decisions

**Why this dataset instead of raw PDFs?**
Raw 10-K PDFs require OCR, layout parsing, and table extraction before retrieval can even begin — all error-prone steps that add noise. This dataset has already done that work: financial figures are extracted via the SEC's XBRL API (exact GAAP values, no parsing guesswork) and the MD&A text is pre-processed and scored with FinBERT. The result is a cleaner, higher-signal corpus with richer metadata for filtered queries.

**Why `BAAI/bge-small-en-v1.5` for embeddings?**
It consistently ranks at the top of the MTEB benchmark for its size class, producing 384-dimensional vectors compact enough to run fast on CPU. It outperforms alternatives like `all-MiniLM-L6-v2` on retrieval tasks while staying lightweight. Downloads once (~130MB) and runs from local cache on all subsequent runs — `local_files_only=True` prevents unnecessary network calls.

**Why hybrid retrieval instead of dense-only?**
Financial data has two very different retrieval needs. Semantic questions like "which company had declining profitability?" need dense search. Exact queries like "AAPL 2023" or "going concern" need BM25 keyword matching. Neither alone covers both — hybrid RRF fusion handles both cases and consistently outperforms either method in evaluation.

**Why a cross-encoder re-ranker on top of hybrid?**
Embedding models encode query and document separately — fast but imprecise. A cross-encoder reads the full (query, document) pair together, giving substantially more accurate relevance scores. Running it on all 1,487 records would be too slow, so it runs only on the top-10 hybrid candidates — the standard production pattern for RAG systems.

**Why FLAN-T5-base instead of Mistral or GPT?**
FLAN-T5-base runs fully offline via transformers library after a one-time model download — no API key, no cost, no financial data sent to an external service. At 250M parameters, it's small enough to load instantly on CPU. `temperature=0.0` ensures deterministic, consistent answers rather than creative variations that could silently change financial figures. The same model category also serves as the RAGAS evaluation judge, keeping the entire project free and self-contained end-to-end.

**Why `chunk_size=800` with `chunk_overlap=64`?**
Each company-year record is approximately 670 characters when formatted as a document. A chunk size of 800 keeps the entire record — Financial Performance, Executive Narrative, and Summary sections — in a single chunk so retrieval never returns half a record. The 64-character overlap provides continuity for the rare records that exceed 800 characters.

**How missing values were handled?**
`Net_Margin` (57% missing) and `Revenue_Growth_YoY` (52% missing) have large gaps due to differing accounting standards across sectors. Missing values are imputed using per-ticker median — a company's own historical performance is a more accurate fill than the S&P 500 global average.

---

## Project structure

```
finsight/
├── src/
│   ├── ingest.py           # Load CSV, clean data, convert rows to Documents
│   ├── embeddings.py       # Load BAAI/bge-small-en-v1.5, embed chunks, cache
│   ├── vectorstore.py      # Build and load ChromaDB vector store
│   ├── retrieval.py        # Dense, sparse, hybrid retrieval + re-ranker
│   ├── generator.py        # Mistral prompt template + answer generation
│   └── pipeline.py         # End-to-end rag_query() — ties all phases together
├── eval/
│   ├── golden_dataset.json  # 25 verified Q&A pairs from real dataset values
│   └── run_eval.py          # RAGAS evaluation + strategy comparison table
├── data/
│   └── SP500_Alpha_Dataset_Final.csv
├── app.py                   # Streamlit demo UI
├── requirements.txt
└── README.md
```

---

## Local setup

**Prerequisites:** Python 3.10+, ~2GB free disk space for models and data

```bash
# 1. Clone the repo
git clone https://github.com/RaveeshSehrawat/finsight.git
cd finsight

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pre-cache models (FLAN-T5, embeddings, re-ranker) — one-time setup
# This requires internet connection and takes 5-10 minutes, ~1.5GB download
python setup_models.py

# 5. Download the dataset
# Go to the Kaggle link above and save as:
# data/SP500_Alpha_Dataset_Final.csv

# 6. Run pipeline phases IN ORDER (required before launching the app)
python src/ingest.py        # Load CSV → clean → create chunks (chunk_size=800)
python src/embeddings.py    # Embed chunks, cache to data/embeddings.npy
python src/vectorstore.py   # Build ChromaDB index — run ONCE only

# 7. Launch the app
streamlit run app.py        # Streamlit at http://localhost:8501
```

> **After step 4:** All models are cached locally. From that point on, the app runs 100% offline — no internet required for step 7 onwards.

> **Windows users:** Setup will automatically download FLAN-T5 (~850MB), embedding model (~130MB), and re-ranker (~85MB) to `~/.cache/huggingface/hub/`. After that, `TRANSFORMERS_OFFLINE=1` loads them from cache — no network calls needed.

---

## Streamlit Cloud deployment

To host FinSight on Streamlit Cloud for free:

1. **Create a Streamlit account** at [share.streamlit.io](https://share.streamlit.io)

2. **Connect your GitHub repository**
   - Click "New app"
   - Select repository: `RaveeshSehrawat/FinSight`
   - Branch: `main`
   - Main file: `app.py`

3. **Deployment happens automatically**
   - Streamlit Cloud will install dependencies from `requirements.txt`
   - App starts and `ensure_models_cached()` runs
   - **First load:** Models download from HuggingFace (~10-15 min, ~1.5GB)
     - You'll see logs: `[SETUP] 📥 Downloading FLAN-T5...` etc.
     - This is normal! Check the **Logs** tab in Streamlit Cloud settings
   - **Subsequent loads:** Models load from cache (~2 min startup)

4. **Access your app**
   - Once deployed: `https://share.streamlit.io/@<your-username>/finsight`
   - Model cache persists across app rebuilds on Streamlit Cloud Pro
   - On free tier, cache is preserved per deployment

**Troubleshooting Streamlit Cloud:**

| Issue | Cause | Solution |
|---|---|---|
| "PARTIAL SUCCESS - Some models failed" | Network timeout or insufficient disk space | Check available disk space (need ~2GB). Redeploy to retry failed downloads. |
| "No sentence-transformers model found" or "EMBEDDING MODEL NOT CACHED" | First load, models still downloading | This is normal! Wait 10-15 min, check **Logs** tab for download progress |
| App times out (~5 min) during first load | Download incomplete (network timeout) | Redeploy. If it keeps failing, upgrade to Streamlit Cloud Pro for longer timeout window |
| Model loads but embedding fails with path error | Cache corrupted or incomplete | Go to app settings → **Advanced settings** → Clear cache, then redeploy |
| "Error: Failed to load embedding model" | Mix of cached and missing models | Check **Logs** tab for which specific models failed. Redeploy to retry. |
| (Local only) "Port already in use" | Already handled by Streamlit Cloud |  |

**Why do models fail to download?**

Common reasons and fixes:

1. **Network timeout (most common)**
   - Symptom: Download starts but stops halfway through
   - Fix: Redeploy the app (GitHub push or reconnect repo)
   - Pro tip: Upgrade to Streamlit Cloud Pro for 1-hour timeout (vs 5-min free tier)

2. **Insufficient disk space**
   - Symptom: First model downloads (e.g., FLAN-T5) but later ones fail
   - Requirement: ~2GB free space in Streamlit Cloud container
   - Fix: Clear existing app cache (Settings → Advanced → Clear cache) then redeploy

3. **HuggingFace server temporarily down**
   - Symptom: "Connection refused" or "429 Too Many Requests"
   - Fix: Wait 5-10 minutes, then redeploy
   - Alternative: Check HuggingFace status at https://status.huggingface.co

4. **Partial cache from previous failed attempt**
   - Symptom: Only 1-2 models cached, app says "PARTIAL SUCCESS"
   - Fix: Disable offline mode to retry downloads. The app does this automatically.
   - If stuck: Settings → Advanced → Clear cache → Redeploy

**Streamlit Cloud logs decode:**
```
[SETUP] ⏳ Missing 3 model(s)           → First load detected, starting download
[SETUP] 📥 Step 1/3: Downloading...    → Currently downloading model
[SETUP] ✅ cached successfully         → That model succeeded
[SETUP] ❌ download failed             → That model failed (will retry on next load)
[SETUP] ⚠️ PARTIAL SUCCESS             → Some succeeded, some failed (retry next load)
[SETUP] ✅ SUCCESS: All models         → Ready to go! (offline mode enabled)
```

---

## HuggingFace Spaces deployment

To host FinSight on HuggingFace Spaces for free:

1. **Create a Space** on [huggingface.co/spaces](https://huggingface.co/spaces)
   - License → OpenSSL
   - Space SDK → **Streamlit** (important!)
   - Name: `finsight` (or your choice)

2. **Push your code**
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name>
   cd <space-name>
   
   # Copy FinSight files
   cp -r /path/to/finsight/* .
   
   # Push to HF Spaces
   git add .
   git commit -m "Deploy FinSight"
   git push
   ```

3. **Build will auto-start** — Space installs dependencies and runs `app.py`
   - First build: setup_models.py runs automatically, downloading FLAN-T5 + embeddings (~1.5GB, ~10 min)
   - Subsequent builds: models loaded from cache (~2 min cold start)

4. **Access at:** `https://huggingface.co/spaces/<your-username>/<space-name>`

**Important notes:**
- HF Spaces has CPU-only (free tier). FLAN-T5 inference runs on CPU.
- Pre-built `chroma_db/` and `data/` directories must be in your repo for instant startup.
- Models are downloaded and cached on first build (requires ~1.5GB space and ~10 min)
- Set Space secrets (if needed): `HF_TOKEN` in Space settings → Secrets (though not needed anymore with local inference)

---

## Running evaluation

```bash
python eval/run_eval.py
```

Runs all three strategies (dense, sparse, hybrid+rerank) against the 25-question golden dataset. Uses Mistral as the RAGAS judge — no OpenAI key required. Prints a comparison table you can paste directly into the Evaluation Results section above.

---

## Limitations

- **Missing financial data** — `Net_Margin` is missing for ~57% of rows and `Revenue_Growth_YoY` for ~52%, due to differing accounting standards across sectors. Filled via per-ticker median imputation.
- **No absolute revenue figures** — the dataset contains growth percentages, not dollar amounts. Ask "what was AAPL's revenue growth?" not "what was Apple's total revenue?"
- **Pre-computed NLP signals** — FinBERT sentiment scores were computed when the dataset was built. The pipeline does not re-score text at query time.
- **Mistral 7B generation quality** — capable for single-company factual Q&A, but less reliable for complex multi-company comparisons across multiple years.
- **RAGAS scoring variance** — RAGAS uses an LLM as a judge internally so scores may vary slightly between evaluation runs.

---

## Future improvements

- [ ] Add parent-child chunking to preserve full company context across multi-year queries
- [ ] Upgrade embeddings to `BAAI/bge-large-en-v1.5` for higher retrieval quality
- [ ] Add multi-company comparison queries (e.g. "compare AAPL and MSFT margins across 3 years")
- [ ] Fine-tune the cross-encoder re-ranker on financial domain Q&A pairs
- [ ] Incorporate live SEC XBRL API for real-time data beyond the static dataset
- [ ] Add a filter sidebar to browse the corpus by company, year, sentiment, or going concern flag

---

## Author

**Raveesh Sehrawat** — [LinkedIn](https://www.linkedin.com/in/raveesh-sehrawat/) · [GitHub](https://github.com/RaveeshSehrawat)