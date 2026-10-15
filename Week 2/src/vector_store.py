import chromadb

DB_PATH = "data/chroma_db"
COLLECTION_NAME = "wiki_chunks"
BATCH_SIZE = 1000


def get_collection(path=DB_PATH, name=COLLECTION_NAME):
    """Open (or create) the ChromaDB collection stored on disk."""
    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )


def ingest(collection, ids, texts, embeddings, metadatas, batch_size=BATCH_SIZE):
    """Add chunks to ChromaDB in batches. Safe to run again (uses upsert).

    embeddings must be a numpy array with one row per chunk.
    Returns how many chunks were stored.
    """
    n = len(ids)
    if not (n == len(texts) == len(embeddings) == len(metadatas)):
        raise ValueError("ids, texts, embeddings and metadatas must have the same length")

    # Edge cases: skip empty chunks and duplicate ids
    seen = set()
    keep = []
    for i in range(n):
        if not texts[i] or not texts[i].strip():
            continue
        if ids[i] in seen:
            continue
        seen.add(ids[i])
        keep.append(i)

    # ChromaDB limits how much can be sent at once, so we send batches
    for start in range(0, len(keep), batch_size):
        part = keep[start:start + batch_size]
        collection.upsert(
            ids=[ids[i] for i in part],
            documents=[texts[i] for i in part],
            embeddings=[embeddings[i].tolist() for i in part],
            metadatas=[metadatas[i] for i in part],
        )
    return len(keep)