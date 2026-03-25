"""
setup_models.py — Pre-download models for Streamlit/HuggingFace Spaces deployment

This script should run ONCE during Space build time to cache all required models.
It's called from the setup.sh script or can be run manually.

Models downloaded:
  - google/flan-t5-base (LLM for text generation) ~850MB
  - BAAI/bge-small-en-v1.5 (embedding model) ~130MB
  - cross-encoder/ms-marco-MiniLM-L-6-v2 (re-ranker model) ~85MB
"""

import os
import sys
from pathlib import Path

def setup_llm_model():
    """Download and cache the LLM (T5) model"""
    print("\n" + "="*70)
    print("Setting up LLM model for Streamlit deployment...")
    print("="*70 + "\n")
    
    # Set environment variable to allow downloads during setup
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        model_name = "google/flan-t5-base"
        print(f"📥 Downloading {model_name}...")
        print("(This may take 3-5 minutes for ~850MB)")
        
        # Download to default HuggingFace cache location
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model = model.to("cpu")
        
        print(f"\n✅ LLM model downloaded successfully!")
        print(f"   ✓ Model: {model_name}")
        print(f"   ✓ Cached at: ~/.cache/huggingface/hub/")
        
        # Verify it can be loaded
        test_input = tokenizer("test", return_tensors="pt")
        test_output = model.generate(test_input["input_ids"], max_length=20)
        result = tokenizer.decode(test_output[0], skip_special_tokens=True)
        print(f"   ✓ Test generation result: '{result}'")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR downloading LLM model: {e}")
        print("\nThis is expected if running in an environment without internet.")
        print("The model must be pre-cached before deployment to Streamlit Cloud.")
        return False

def setup_embedding_model():
    """Download and cache the embedding model"""
    print("\n" + "="*70)
    print("Setting up embedding model for Streamlit deployment...")
    print("="*70 + "\n")
    
    # Set environment variable to allow downloads during setup
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = "BAAI/bge-small-en-v1.5"
        print(f"📥 Downloading {model_name}...")
        print("(This may take 2-3 minutes for ~130MB)")
        
        # Download to default HuggingFace cache location
        model = SentenceTransformer(model_name, device="cpu")
        
        print(f"\n✅ Embedding model downloaded successfully!")
        print(f"   ✓ Model: {model_name}")
        print(f"   ✓ Embedding dimension: {model.get_sentence_embedding_dimension()}")
        print(f"   ✓ Cached at: ~/.cache/huggingface/hub/")
        
        # Verify it can be loaded
        test_embedding = model.encode("test", normalize_embeddings=True)
        print(f"   ✓ Test embedding shape: {test_embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR downloading embedding model: {e}")
        print("\nThis is expected if running in an environment without internet.")
        print("The model must be pre-cached before deployment to Streamlit Cloud.")
        return False


def setup_reranker_model():
    """Download and cache the re-ranker model"""
    print("\n" + "="*70)
    print("Setting up re-ranker model for Streamlit deployment...")
    print("="*70 + "\n")
    
    # Ensure downloads are enabled
    os.environ['HF_HUB_OFFLINE'] = '0'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    
    try:
        from sentence_transformers.cross_encoder import CrossEncoder
        
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        print(f"📥 Downloading {model_name}...")
        print("(This may take 1-2 minutes for ~85MB)")
        
        # Download to default HuggingFace cache location
        model = CrossEncoder(model_name, device="cpu")
        model.model = model.model.to("cpu")  # Ensure CPU placement
        
        print(f"\n✅ Re-ranker model downloaded successfully!")
        print(f"   ✓ Model: {model_name}")
        print(f"   ✓ Cached at: ~/.cache/huggingface/hub/")
        
        # Quick test
        test_scores = model.predict([("test query", "test passage")])
        print(f"   ✓ Test score shape: {test_scores.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR downloading re-ranker model: {e}")
        print("\nThis is expected if running in an environment without internet.")
        return False


if __name__ == "__main__":
    print("🚀 PRE-CACHING MODELS FOR STREAMLIT CLOUD DEPLOYMENT\n")
    
    llm_success = setup_llm_model()
    embedding_success = setup_embedding_model()
    reranker_success = setup_reranker_model()
    
    print("\n" + "="*70)
    if llm_success and embedding_success and reranker_success:
        print("✅ ALL MODELS CACHED SUCCESSFULLY!")
        print("   Your Streamlit Cloud app is ready to run!")
        sys.exit(0)
    else:
        print("⚠️  PARTIAL SUCCESS - Some models failed to download")
        print("   The app may still work if running in cached/offline mode")
        sys.exit(0 if (llm_success or embedding_success or reranker_success) else 1)
    print("="*70)
