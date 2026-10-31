import pandas as pd

from src.search import get_model
from src.vector_store import get_collection

NLP_COLLECTION = "wiki_chunks_nlp"


def build_where(topic_id=None, has_person=None, has_org=None, has_place=None,
                has_date=None, min_similarity=None, model_topics_only=False):
    """Turn simple options into a ChromaDB filter (or None for no filter)."""
    conditions = []
    if topic_id is not None:
        conditions.append({"topic_id": int(topic_id)})
    for name, value in (("has_person", has_person), ("has_org", has_org),
                        ("has_place", has_place), ("has_date", has_date)):
        if value is not None:
            conditions.append({name: bool(value)})
    if min_similarity is not None:
        conditions.append({"topic_similarity": {"$gte": float(min_similarity)}})
    if model_topics_only:
        conditions.append({"topic_source": "model"})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def find_topic(keyword, path="results/topics.csv"):
    """Find the biggest topic whose label contains the keyword."""
    topics = pd.read_csv(path)
    found = topics[topics["label"].str.contains(keyword, case=False, regex=False)]
    if found.empty:
        raise ValueError(f"No topic label contains '{keyword}'")
    return int(found.sort_values("docs_total", ascending=False).iloc[0]["topic_id"])


def filtered_search(query, k=5, where=None, collection=None, model=None):
    """Semantic search that only looks at chunks matching the filter."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if k <= 0:
        raise ValueError("k must be greater than 0")
    if collection is None:
        collection = get_collection(name=NLP_COLLECTION)
    total = collection.count()
    if total == 0:
        return []
    k = min(k, total)
    if model is None:
        model = get_model()
    vector = model.encode([query.strip()], normalize_embeddings=True)
    result = collection.query(query_embeddings=vector.tolist(), n_results=k, where=where)

    hits = []
    for i in range(len(result["ids"][0])):
        hits.append({
            "id": result["ids"][0][i],
            "title": result["metadatas"][0][i]["title"],
            "distance": result["distances"][0][i],
            "text": result["documents"][0][i],
            "metadata": result["metadatas"][0][i],
        })
    return hits