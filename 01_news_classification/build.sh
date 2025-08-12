#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create NLTK data directory
mkdir -p /opt/render/nltk_data

# Download required NLTK data
python -m nltk.downloader -d /opt/render/nltk_data punkt punkt_tab stopwords wordnet omw-1.4



