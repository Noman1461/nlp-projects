#!/usr/bin/env bash
# Install Python dependencies
pip install -r requirements.txt

# Download NLTK data to a directory inside the Render persistent build area
python -m nltk.downloader -d /opt/render/nltk_data punkt stopwords wordnet


