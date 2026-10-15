import time

import numpy as np
import pandas as pd

from src.vector_store import get_collection, ingest

chunks = pd.read_parquet("data/processed/chunks.parquet")
embeddings = np.load("data/processed/embeddings.npy")

if len(chunks) != len(embeddings):
    raise SystemExit("chunks.parquet and embeddings.npy do not match. Re-run embed_chunks.")

collection = get_collection()
print("Already in collection:", collection.count())

metadatas = [
    {"doc_id": r.doc_id, "title": r.title, "chunk_index": int(r.chunk_index)}
    for r in chunks.itertuples()
]

t0 = time.perf_counter()
stored = ingest(
    collection,
    chunks["chunk_id"].tolist(),
    chunks["text"].tolist(),
    embeddings,
    metadatas,
)
elapsed = time.perf_counter() - t0

print(f"Ingested {stored} chunks in {elapsed:.1f}s")
print("Collection now has", collection.count(), "chunks")