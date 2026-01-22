#!/bin/bash

# Run evaluation script

set -e

echo "Starting LLM Evaluation Framework..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found"
    echo "Please copy .env.example to .env and configure it"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Download NLTK data
python3 -c "import nltk; nltk.download('punkt', quiet=True)" || true

# Run evaluation
echo "Running evaluation..."
python main.py

echo "Evaluation complete!"
