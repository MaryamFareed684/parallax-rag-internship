import json
import time

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

from src.vector_store import get_collection, ingest

COLLECTION_NAME = "wiki_chunks_nlp"
DB_PATH = "data/chroma_db"
PLACE_LABELS = {"GPE", "LOC", "FAC"}

chunks = pd.read_parquet("data/processed/chunks.parquet")
embeddings = np.load("data/processed/embeddings.npy")
topics = pd.read_parquet("data/processed/docs_topics.parquet")[
    ["doc_id", "topic_id", "topic_label", "topic_source", "topic_similarity"]]
chunks["doc_id"] = chunks["doc_id"].astype(str)
topics["doc_id"] = topics["doc_id"].astype(str)
chunks = chunks.merge(topics, on="doc_id", how="left")

if chunks["topic_id"].isna().any():
    raise SystemExit("Some chunks have no topic. Run scripts.run_topics again.")
if len(chunks) != len(embeddings):
    raise SystemExit("chunks.parquet and embeddings.npy do not match.")

try:
    nlp = spacy.load("en_core_web_sm",
                     disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
except OSError:
    raise SystemExit("Run first: python -m spacy download en_core_web_sm")

# 1. Named entity recognition on every chunk
start = time.perf_counter()
flags = {"has_person": [], "has_org": [], "has_place": [], "has_date": [], "n_entities": []}
texts = chunks["text"].tolist()
for doc in tqdm(nlp.pipe(texts, batch_size=128), total=len(texts)):
    labels = [ent.label_ for ent in doc.ents]
    flags["has_person"].append("PERSON" in labels)
    flags["has_org"].append("ORG" in labels)
    flags["has_place"].append(any(label in PLACE_LABELS for label in labels))
    flags["has_date"].append("DATE" in labels)
    flags["n_entities"].append(len(labels))
ner_seconds = time.perf_counter() - start
for name, values in flags.items():
    chunks[name] = values

summary = {
    "chunks": len(chunks),
    "ner_seconds": round(ner_seconds, 1),
    "chunks_per_second": round(len(chunks) / ner_seconds, 1),
    "share_has_person": round(float(chunks["has_person"].mean()), 3),
    "share_has_org": round(float(chunks["has_org"].mean()), 3),
    "share_has_place": round(float(chunks["has_place"].mean()), 3),
    "share_has_date": round(float(chunks["has_date"].mean()), 3),
}
print(summary)
with open("results/ner_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
chunks.to_parquet("data/processed/chunks_nlp.parquet", index=False)

# 2. New ChromaDB collection with the NLP metadata
metadatas = []
for r in chunks.itertuples():
    metadatas.append({
        "doc_id": str(r.doc_id),
        "title": str(r.title),
        "chunk_index": int(r.chunk_index),
        "topic_id": int(r.topic_id),
        "topic_label": str(r.topic_label),
        "topic_source": str(r.topic_source),
        "topic_similarity": float(r.topic_similarity),
        "has_person": bool(r.has_person),
        "has_org": bool(r.has_org),
        "has_place": bool(r.has_place),
        "has_date": bool(r.has_date),
        "n_entities": int(r.n_entities),
    })

collection = get_collection(path=DB_PATH, name=COLLECTION_NAME)
start = time.perf_counter()
stored = ingest(collection, chunks["chunk_id"].tolist(), chunks["text"].tolist(),
                embeddings, metadatas)
print(f"Stored {stored} chunks in {time.perf_counter() - start:.1f}s")
print("Collection now has", collection.count(), "chunks")