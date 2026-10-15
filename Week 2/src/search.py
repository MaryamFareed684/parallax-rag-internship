from sentence_transformers import SentenceTransformer

from src.vector_store import get_collection

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model():
    """Load the embedding model only once, then reuse it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def search(query, k=5, collection=None, model=None):
    """Return the k chunks closest in meaning to the query."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if collection is None:
        collection = get_collection()
    total = collection.count()
    if total == 0:
        return []              # empty database: nothing to find
    k = min(k, total)          # asking for more than we have would be an error

    if model is None:
        model = get_model()
    query_vector = model.encode([query.strip()], normalize_embeddings=True)

    result = collection.query(query_embeddings=query_vector.tolist(), n_results=k)

    hits = []
    for i in range(len(result["ids"][0])):
        meta = result["metadatas"][0][i]
        hits.append({
            "id": result["ids"][0][i],
            "title": meta["title"],
            "chunk_index": meta["chunk_index"],
            "distance": result["distances"][0][i],
            "text": result["documents"][0][i],
        })
    return hits