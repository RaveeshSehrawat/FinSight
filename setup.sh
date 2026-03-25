#!/bin/bash
# setup.sh — Run during Streamlit Cloud build to pre-cache models
echo "🔧 Starting Streamlit Cloud setup..."
echo "📥 Pre-downloading HuggingFace models..."

# Run the Python setup script to download models
python setup_models.py

echo "✅ Setup complete!"
