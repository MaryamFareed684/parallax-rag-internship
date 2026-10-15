# Week 2: Chunking, Embeddings, ChromaDB & Retrieval Benchmark

## What I built
A pipeline that takes the clean Wikipedia corpus from Week 1 (5,920 documents),
splits it into chunks, embeds the chunks, stores them in ChromaDB, and searches
them by meaning. I also measured embedding speed and search latency.

## Pipeline
clean_corpus.parquet -> chunking -> embeddings (all-MiniLM-L6-v2, 384 dims)
-> ChromaDB (persistent, cosine distance) -> semantic search -> latency benchmark

## Chunking strategy
- Method: recursive splitting, written from scratch in src/chunking.py
- Separators tried in order: paragraph break, line break, sentence end (". "), space,
  and finally a hard cut by characters
- Chunk size: 500 characters. Overlap: 50 characters
- Why 500 / 50: small chunks give precise search results, and the overlap keeps
  a sentence from being lost when it falls on a border
- Note: the Week 1 cleaning turned all new lines into spaces, so in practice
  splitting happens at sentence ends and spaces
- Chunks per document: capped at the first 5 (see Decisions below)

## Embeddings
- Model: all-MiniLM-L6-v2 (384 dimensions), batch size 64, normalized vectors
- Hardware: CPU only (Intel Iris Xe laptop, no NVIDIA GPU)

| Run | Documents | Chunks | Embedding time | Speed |
|---|---|---|---|---|
| Test | 200 | 14,888 | 824 s | 18.1 chunks/s |
| Final (max 5 chunks/doc) | 5,920 | 26,098 | 1,271 s | 20.5 chunks/s |

Other timings of the final run: chunking 3.95 s, model load 6.6 s.
All runs are logged in results/embedding_log.jsonl.

## Decisions
- Without a limit, the corpus makes 180,541 chunks. At about 20 chunks/s on my CPU
  that is roughly 2.5 hours, so I kept only the first 5 chunks of each article.
  This still covers all 5,920 documents, and the start of a Wikipedia article
  usually holds its main facts.

## ChromaDB
- Persistent client, folder data/chroma_db (git-ignored), collection "wiki_chunks"
- Distance: cosine (smaller = more similar)
- Stored per chunk: id, text, embedding, metadata (doc_id, title, chunk_index)
- Ingestion of 26,098 chunks took about 43 s; re-running gave the same count (26,098)

## Edge cases handled
- Empty or whitespace-only chunks are skipped
- Duplicate chunk ids are skipped, and upsert makes re-running safe
- Lists of different lengths raise a clear error
- Data is sent to ChromaDB in batches of 1000
- Empty or non-text query raises an error; k <= 0 raises an error
- k larger than the number of stored chunks is reduced to the collection size
- Searching an empty collection returns an empty list
- Invalid chunk settings (chunk_size <= 0, overlap >= chunk_size) raise an error

## Retrieval benchmark
12 queries in 5 categories, each run 5 times after a warm-up search. Results are in
results/retrieval_latency.csv and results/retrieval_latency_by_k.csv.

- Overall (60 searches): mean 11.7 ms, p50 11.3 ms, p95 15.9 ms
- Embedding the query takes about 5-8 ms of the total; the rest is ChromaDB search
- Long queries are slowest (about 15 ms); short queries about 10-11 ms
- k=1: 10.4 ms, k=5: 11.5 ms, k=10: 10.6 ms, k=50: 15.2 ms
- Top-1 distance: on-topic queries 0.25-0.46, typo queries about 0.52-0.54,
  off-topic queries 0.60-0.67

## Example results and limitations
- "who invented the telephone" -> Alexander Graham Bell (distance 0.254)
- "history of the Roman Empire" -> a TV series called "Empire" ranked above Augustus.
  The model matched the word "Empire" more than the topic. Retrieval is based on
  meaning of short chunks, so it can be fooled by shared words.
- Off-topic queries still return something, only with larger distances. A distance
  threshold will be needed later so the assistant can say it does not know.

## How to run
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
(copy data/processed/clean_corpus.parquet from Week 1 into Week 2/data/processed/)
python -m scripts.embed_chunks --max-chunks-per-doc 5
python -m scripts.ingest_chroma
python -m scripts.search_demo "who invented the telephone"
python -m scripts.benchmark_retrieval
python -m pytest

## Tests
13 tests pass (7 for chunking, 6 for ChromaDB ingestion and search edge cases).

## Folder structure
Week 2/
  src/       chunking.py, vector_store.py, search.py
  scripts/   embed_chunks.py, ingest_chroma.py, search_demo.py, benchmark_retrieval.py
  tests/     test_chunking.py, test_vector_store.py
  results/   embedding_log.jsonl, retrieval_latency.csv, retrieval_latency_by_k.csv