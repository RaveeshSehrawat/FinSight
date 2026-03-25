"""
src/generator.py — Phase 5a

Loads LLM using local transformers library with offline mode for Streamlit Cloud.
No external API calls, no HuggingFace Inference, no Ollama required.

Uses T5 seq2seq models for text generation (google/flan-t5-base by default).

Run standalone to test generation in isolation:
    python src/generator.py
"""

import os
import torch
from typing import Any, Callable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# IMPORTANT: DO NOT force offline mode here - let app.py control it!
# Only set defaults if not already configured by app.py
if "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"
if "TRANSFORMERS_OFFLINE" not in os.environ:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM


# ─────────────────────────────────────────────
# 1. LOAD THE LLM (Local Transformers with Offline Mode)
# ─────────────────────────────────────────────

def load_llm(model_name: str = "google/flan-t5-base", temperature: float = 0.0, max_tokens: int = 300):
    """
    Loads LLM using local transformers library with offline/online mode.
    
    Supports both seq2seq (T5, FLAN-T5) and causal (GPT-style) models.
    No external API calls, no HuggingFace Inference required.
    
    Args:
        model_name: HuggingFace model identifier (must be in cache)
        temperature: 0.0 = deterministic (best for financial Q&A)
        max_tokens: Maximum length for generated response
    
    Returns:
        LangChain RunnableLambda wrapper for the model
    """
    
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    offline_mode = os.environ.get('HF_HUB_OFFLINE') == '1'
    
    print(f"\n[LOAD_LLM] Loading {model_name}")
    print(f"[LOAD_LLM] Offline mode: {offline_mode}")
    print(f"[LOAD_LLM] Cache directory: {cache_dir}")
    
    # Try to load with local_files_only first
    try:
        print(f"[LOAD_LLM] Attempting to load from cache...")
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
            trust_remote_code=True
        )
        print(f"[LOAD_LLM]   ✓ Tokenizer loaded from cache")
        
        # Load model (auto-detect T5 vs causal)
        # IMPORTANT: Use device_map="cpu" to avoid meta tensor issues
        if "t5" in model_name.lower():
            print(f"[LOAD_LLM]   Detected T5 seq2seq model")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                local_files_only=True,
                trust_remote_code=True,
                device_map="cpu",  # Load directly to CPU, avoid meta device
                low_cpu_mem_usage=True  # Use flash attention if available
            )
        else:
            print(f"[LOAD_LLM]   Detected causal model")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                local_files_only=True,
                trust_remote_code=True,
                device_map="cpu",  # Load directly to CPU, avoid meta device
                low_cpu_mem_usage=True  # Use flash attention if available
            )
        
        model.eval()  # Set to eval mode (no dropout)
        print(f"[LOAD_LLM]   ✓ Model loaded from cache on CPU")
        
        # Create inference function
        def call_llm(prompt: str) -> str:
            """Runs inference on the prompt."""
            # Convert prompt to string if it's a LangChain ChatPromptValue
            prompt_str = str(prompt) if not isinstance(prompt, str) else prompt
            
            # Tokenize
            inputs = tokenizer(
                prompt_str,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            # Move inputs to CPU
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            
            # Generate with no_grad for efficiency
            with torch.no_grad():
                if "t5" in model_name.lower():
                    # T5 seq2seq: encoder-decoder
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_length=max_tokens + 50,
                        temperature=temperature or 0.7,
                        do_sample=temperature > 0,
                        top_p=0.95
                    )
                else:
                    # Causal model: decoder-only
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_new_tokens=max_tokens,
                        temperature=temperature or 0.7,
                        do_sample=temperature > 0,
                        top_p=0.95
                    )
            
            # Decode output
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result.strip()
        
        # Wrap in RunnableLambda for LangChain compatibility
        llm = RunnableLambda(call_llm)
        print(f"[LOAD_LLM] ✅ Local model ready: {model_name}")
        return llm
    
    except Exception as e1:
        print(f"[LOAD_LLM] ⚠️  Cache load failed: {str(e1)[:100]}")
        
        # If offline mode is enabled and cache failed, TRY DOWNLOAD
        if offline_mode:
            print(f"[LOAD_LLM] Offline mode ON but cache not found. Attempting download anyway...")
            try:
                # Temporarily disable offline mode for download
                os.environ["HF_HUB_OFFLINE"] = "0"
                os.environ["TRANSFORMERS_OFFLINE"] = "0"
                
                print(f"[LOAD_LLM] Downloading {model_name} (~5 min, ~850MB)...")
                
                # Load tokenizer with downloads enabled
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                print(f"[LOAD_LLM]   ✓ Tokenizer downloaded")
                
                # Load model with device_map to avoid meta tensor issues
                if "t5" in model_name.lower():
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        device_map="cpu",
                        low_cpu_mem_usage=True
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        device_map="cpu",
                        low_cpu_mem_usage=True
                    )
                
                model.eval()
                print(f"[LOAD_LLM]   ✓ Model downloaded and loaded on CPU")
                
                # Create inference function (same as before)
                def call_llm(prompt: str) -> str:
                    prompt_str = str(prompt) if not isinstance(prompt, str) else prompt
                    inputs = tokenizer(
                        prompt_str,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    )
                    inputs = {k: v.to("cpu") for k, v in inputs.items()}
                    with torch.no_grad():
                        if "t5" in model_name.lower():
                            outputs = model.generate(
                                inputs["input_ids"],
                                max_length=max_tokens + 50,
                                temperature=temperature or 0.7,
                                do_sample=temperature > 0,
                                top_p=0.95
                            )
                        else:
                            outputs = model.generate(
                                inputs["input_ids"],
                                max_new_tokens=max_tokens,
                                temperature=temperature or 0.7,
                                do_sample=temperature > 0,
                                top_p=0.95
                            )
                    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    return result.strip()
                
                llm = RunnableLambda(call_llm)
                print(f"[LOAD_LLM] ✅ Model successfully downloaded and loaded!")
                
                # Re-enable offline mode for future operations
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                
                return llm
                
            except Exception as e2:
                print(f"[LOAD_LLM] ❌ Download also failed: {str(e2)[:100]}")
                raise RuntimeError(
                    f"\n❌ FAILED TO LOAD MODEL: {model_name}\n\n"
                    f"✓ First attempted: Load from cache → {str(e1)[:80]}\n"
                    f"✓ Then attempted: Download online → {str(e2)[:80]}\n\n"
                    f"Solutions:\n"
                    f"1. On Streamlit Cloud: Redeploy (models download on first load)\n"
                    f"2. Locally: Run 'python setup_models.py'\n"
                    f"3. Check cache directory: {cache_dir}\n"
                    f"4. Check internet connection and HuggingFace status\n"
                ) from e2
        else:
            # Online mode but still failed
            raise RuntimeError(
                f"\n❌ FAILED TO LOAD MODEL: {model_name}\n\n"
                f"Error: {str(e1)[:150]}\n\n"
                f"Solutions:\n"
                f"1. Check cache directory: {cache_dir}\n"
                f"2. Verify model name is correct\n"
                f"3. Run 'python setup_models.py' to download models\n"
                f"4. Check internet connection\n"
            ) from e1


# ─────────────────────────────────────────────
# 2. PROMPT TEMPLATE
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are FinSight, a financial data analyst assistant.

You have been given structured financial data records from SEC 10-K filings.
Each record contains exact financial metrics for a specific company and year.

CRITICAL RULES:
1. You MUST answer using the data in the context below. The data is already extracted and verified.
2. NEVER say "I cannot find" or "I don't have enough context" — the answer IS in the context.
3. Always state the specific number or value from the context in your answer.
4. Always end with: [Source: TICKER_YEAR]
5. Keep answers short and factual — one or two sentences maximum.

EXAMPLE:
Context: Company: CPB | Fiscal Year: 2024 ... Net Profit Margin: 5.88%
Question: What was CPB's net profit margin in 2024?
Answer: CPB reported a net profit margin of 5.88% in fiscal year 2024. [Source: CPB_2024]

Financial data:
──────────────────────────────────
{context}
──────────────────────────────────
"""


def build_prompt() -> ChatPromptTemplate:
    """Returns the ChatPromptTemplate used for every RAG query."""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])


# ─────────────────────────────────────────────
# 3. FORMAT CONTEXT FROM RETRIEVED CHUNKS
# ─────────────────────────────────────────────

def format_context(retrieved_chunks: list, max_chars: int = 3500) -> str:
    """
    Formats retrieved chunks into a clean context string for the prompt.

    Each chunk is labelled with its ticker and year so the LLM
    can cite it correctly in the answer.

    max_chars: rough limit to stay within Mistral's context window.
    At ~4 chars per token, 3500 chars ≈ 875 tokens of context.
    This leaves plenty of room for the system prompt + answer.
    """
    context_parts = []
    total_chars   = 0

    for i, result in enumerate(retrieved_chunks):
        doc     = result["doc"]
        ticker  = doc.metadata.get("ticker",  doc.metadata.get("company", "UNKNOWN")).upper()
        year    = doc.metadata.get("year",    "?")
        text    = doc.page_content.strip()

        chunk_str = f"[Source {i+1}: {ticker}_{year}]\n{text}\n"

        if total_chars + len(chunk_str) > max_chars:
            break

        context_parts.append(chunk_str)
        total_chars += len(chunk_str)

    return "\n".join(context_parts)


# ─────────────────────────────────────────────
# 4. GENERATE AN ANSWER
# ─────────────────────────────────────────────

def generate_answer(
    question        : str,
    retrieved_chunks: list,
    llm             : Any,
    min_score       : float = 0.25,
) -> dict:
    """
    Generates a grounded answer from Mistral given a question + retrieved chunks.

    Steps:
        1. Guard: if no chunks or top chunk score is too low, return fallback
        2. Format retrieved chunks into a context string
        3. Build prompt and invoke Mistral
        4. Return answer string + structured sources list

    min_score: relevance threshold — only applied for dense retrieval
    (cosine similarity 0–1). RRF hybrid scores are tiny floats ~0.03,
    so the threshold is skipped for those.

    Returns:
        {
            "answer"             : str,
            "sources"            : list of dicts,
            "skipped_generation" : bool
        }
    """

    # ── Guard: no chunks at all ───────────────────────────────────────────
    if not retrieved_chunks:
        return {
            "answer"             : "I couldn't find relevant information in the loaded dataset.",
            "sources"            : [],
            "skipped_generation" : True,
        }

    # ── Guard: low relevance (dense retrieval only) ───────────────────────
    top_score  = retrieved_chunks[0].get("score", 1.0)
    retriever  = retrieved_chunks[0].get("retriever", "hybrid")
    if retriever == "dense" and top_score < min_score:
        return {
            "answer"             : (
                "I couldn't find relevant information for that question. "
                "Try rephrasing or asking about a specific company ticker and year."
            ),
            "sources"            : [],
            "skipped_generation" : True,
        }

    # ── Format context + build chain ─────────────────────────────────────
    context  = format_context(retrieved_chunks)
    prompt   = build_prompt()
    chain    = prompt | llm

    # ── Call Mistral ──────────────────────────────────────────────────────
    response = chain.invoke({
        "context" : context,
        "question": question,
    })
    answer = response.content if hasattr(response, "content") else str(response)

    # ── Build structured sources list for the UI ──────────────────────────
    sources = []
    for r in retrieved_chunks:
        doc    = r["doc"]
        ticker = doc.metadata.get("ticker", doc.metadata.get("company", "?")).upper()
        year   = doc.metadata.get("year", "?")
        sources.append({
            "ticker"         : ticker,
            "year"           : year,
            "filename"       : doc.metadata.get("filename", f"{ticker}_{year}_10K.txt"),
            "going_concern"  : doc.metadata.get("going_concern", "0"),
            "sentiment_raw"  : doc.metadata.get("sentiment_raw", "0"),
            "text"           : doc.page_content[:350].strip(),
            "score"          : round(r.get("score", 0), 4),
        })

    return {
        "answer"             : answer,
        "sources"            : sources,
        "skipped_generation" : False,
    }


# ─────────────────────────────────────────────
# MAIN — test generation without running full pipeline
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Mock a retrieved chunk using real data from the dataset
    mock_chunks = [
        {
            "doc": type("Doc", (), {
                "page_content": (
                    "Company: AAPL | Fiscal Year: 2023\n\n"
                    "--- Financial Performance ---\n"
                    "Net Profit Margin      : 25.31%\n"
                    "Return on Assets (ROA) : 28.29%\n"
                    "Revenue Growth YoY     : -2.80%\n"
                    "Net Income Growth YoY  : -2.81%\n\n"
                    "--- Executive Narrative (MD&A Analysis) ---\n"
                    "FinBERT Sentiment Score : Negative (-0.248)\n"
                    "Writing Complexity      : Grade 14.8 — Complex (college level)\n"
                    "Auditor Warning         : No going concern warning issued."
                ),
                "metadata": {
                    "ticker"        : "AAPL",
                    "year"          : "2023",
                    "going_concern" : "0",
                    "sentiment_raw" : "-0.2483",
                    "source"        : "AAPL_2023",
                    "filename"      : "AAPL_2023_10K.txt",
                    "chunk_id"      : "0",
                },
            })(),
            "score"    : 0.88,
            "chunk_id" : "0",
            "retriever": "dense",
        }
    ]

    print("[TEST] Starting local LLM inference test...")
    try:
        llm    = load_llm(model_name="google/flan-t5-base")
        result = generate_answer("What was Apple's net profit margin in 2023?", mock_chunks, llm)
        
        print("\n✓ Answer:")
        print(f"  {result['answer']}")
        print("\n✓ Sources used:")
        for s in result["sources"]:
            print(f"  - {s['ticker']} {s['year']} | sentiment: {s['sentiment_raw']} | score: {s['score']}")
        
        print("\n[TEST] ✅ Phase 5a complete - local inference working!")
    except Exception as e:
        print(f"\n[TEST] ❌ Error: {e}")
        raise
