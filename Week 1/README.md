# Week 1: Environment, Data Ingestion & Preprocessing

## Project
RAG (Retrieval-Augmented Generation) assistant built during the Parallax Labs internship.

## Progress this week
- Set up the repo, virtual environment, .gitignore and locked requirements.txt
- Verified the environment (scripts/verify_env.py)
- Downloaded 6,000 Wikipedia articles (English)
- Wrote cleaning functions: HTML stripping, Unicode normalization, whitespace cleanup, language filtering
- Wrote 9 unit tests (all passing)
- Built the clean dataset: 5,920 documents

## Hardware note
My laptop has only Intel Iris Xe graphics (no NVIDIA GPU), so CUDA is not available.
verify_env.py reports this as a warning and everything runs on CPU.
GPU-heavy steps in later weeks will use Google Colab.

## Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

## How to run
python scripts/verify_env.py
python scripts/download_data.py
python -m scripts.build_clean_corpus
python -m pytest

## Dataset
- Source: Wikipedia (wikimedia/wikipedia, 20231101.en) via Hugging Face
- Raw: 6,000 documents -> after short-doc filter: 5,940 -> after English filter: 5,920 -> after duplicate removal: 5,920
- Raw and processed data are git-ignored. Run the two scripts above to recreate them.

## Folder structure
Week 1/
  scripts/   verify_env.py, download_data.py, build_clean_corpus.py
  src/       preprocessing.py
  tests/     test_preprocessing.py