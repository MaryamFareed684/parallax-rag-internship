import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.chunking import chunk_text

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
OVERLAP = 50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--max-chunks-per-doc", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_parquet("data/processed/clean_corpus.parquet")
    if args.limit_docs:
        df = df.head(args.limit_docs)
    print("Documents:", len(df))

    # 1. Chunking
    t0 = time.perf_counter()
    rows = []
    for doc in df.itertuples():
        chunks = chunk_text(doc.text, CHUNK_SIZE, OVERLAP)
        if args.max_chunks_per_doc:
            chunks = chunks[:args.max_chunks_per_doc]
        for i, chunk in enumerate(chunks):
            rows.append({
                "chunk_id": f"{doc.id}_{i}",
                "doc_id": str(doc.id),
                "title": doc.title,
                "chunk_index": i,
                "text": chunk,
            })
    chunk_time = time.perf_counter() - t0
    chunks_df = pd.DataFrame(rows)
    print(f"Chunks: {len(chunks_df)} (chunking took {chunk_time:.2f}s)")

    # 2. Load the model
    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME)
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.2f}s")

    # 3. Embed
    t0 = time.perf_counter()
    embeddings = model.encode(
        chunks_df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embed_time = time.perf_counter() - t0
    speed = len(chunks_df) / embed_time
    print(f"Embedding took {embed_time:.2f}s ({speed:.1f} chunks/second)")
    print("Embedding shape:", embeddings.shape)

    # 4. Save
    chunks_df.to_parquet("data/processed/chunks.parquet", index=False)
    np.save("data/processed/embeddings.npy", embeddings)

    # 5. Log this run (one line per run)
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL_NAME,
        "documents": len(df),
        "chunks": len(chunks_df),
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "chunking_seconds": round(chunk_time, 2),
        "model_load_seconds": round(load_time, 2),
        "embedding_seconds": round(embed_time, 2),
        "chunks_per_second": round(speed, 1),
        "embedding_dim": int(embeddings.shape[1]),
    }
    with open("results/embedding_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print("Logged to results/embedding_log.jsonl")


if __name__ == "__main__":
    main()