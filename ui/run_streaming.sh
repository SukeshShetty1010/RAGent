#!/bin/bash
# Run the RAGent Conversational Streaming UI

# Navigate to project root
cd "$(dirname "$0")/.."

# Set PYTHONPATH to include project root
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Run the streamlit app
streamlit run ui/app_streaming.py --server.port 8501
