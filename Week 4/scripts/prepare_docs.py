import numpy as np
import pandas as pd

LEAD_CHARS = 2500   # the start of an article says what it is about

chunks = pd.read_parquet("data/processed/chunks.parquet")
embeddings = np.load("data/processed/embeddings.npy")
corpus = pd.read_parquet("data/processed/clean_corpus.parquet")

# 1. One embedding per document = average of its chunk embeddings
frame = pd.DataFrame(embeddings)
frame["doc_id"] = chunks["doc_id"].astype(str).values
doc_emb = frame.groupby("doc_id", sort=False).mean()
vectors = doc_emb.to_numpy(dtype="float32")
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# 2. A matching table of documents, in the same order
corpus["doc_id"] = corpus["id"].astype(str)
docs = corpus.set_index("doc_id").loc[doc_emb.index].reset_index()
docs["lead_text"] = docs["text"].str[:LEAD_CHARS]
docs["word_count"] = docs["text"].str.split().str.len()
docs = docs[["doc_id", "title", "lead_text", "word_count"]]

print("Documents:", len(docs), "| embedding matrix:", vectors.shape)
print(docs["word_count"].describe())
for t in (50, 100, 150, 200):
    print(f"Documents under {t} words:", int((docs["word_count"] < t).sum()))

docs.to_parquet("data/processed/docs.parquet", index=False)
np.save("data/processed/doc_embeddings.npy", vectors)
print("Saved docs.parquet and doc_embeddings.npy")