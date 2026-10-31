import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from umap import UMAP

SHORT_WORDS = 100        # documents with fewer words count as "short"
MIN_CLUSTER_SIZE = 25    # bigger number = fewer, broader topics
SEED = 42
NOISE_WORDS = ["aligncenter", "colspan", "rowspan", "nowrap", "bgcolor"]


docs = pd.read_parquet("data/processed/docs.parquet")
emb = np.load("data/processed/doc_embeddings.npy")
docs["is_short"] = docs["word_count"] < SHORT_WORDS
core = ~docs["is_short"].to_numpy()
core_idx = np.flatnonzero(core)
print(f"Documents: {len(docs)} | short (< {SHORT_WORDS} words): {int((~core).sum())}")

# 1. Fit the topic model on the longer documents only
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                  metric="cosine", random_state=SEED)
hdbscan_model = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric="euclidean",
                        cluster_selection_method="eom", prediction_data=True)
vectorizer = CountVectorizer(stop_words=list(ENGLISH_STOP_WORDS) + NOISE_WORDS,
                             ngram_range=(1, 2), strip_accents="unicode",
                             token_pattern=r"(?u)\b[^\W\d_]{3,}\b")
model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model,
                 vectorizer_model=vectorizer, calculate_probabilities=False, verbose=True)

texts = docs.loc[core, "lead_text"].tolist()
fitted, _ = model.fit_transform(texts, embeddings=emb[core])
fitted = np.array(fitted)

# 2. Center (average embedding) of every real topic
topic_ids = sorted(t for t in set(fitted.tolist()) if t != -1)
centroids = np.vstack([emb[core_idx[fitted == t]].mean(axis=0) for t in topic_ids])
centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

# 3. Nearest topic for every document (used for outliers and short documents)
sims = emb @ centroids.T
nearest_topic = np.array(topic_ids)[sims.argmax(axis=1)]

model_topic = np.full(len(docs), -1)
model_topic[core_idx] = fitted
in_model = model_topic != -1

topic_final = nearest_topic.copy()
topic_final[in_model] = model_topic[in_model]

source = np.array(["short_reassigned"] * len(docs), dtype=object)
source[in_model] = "model"
source[core & (model_topic == -1)] = "outlier_reassigned"

index_of = {t: i for i, t in enumerate(topic_ids)}
assigned_sim = sims[np.arange(len(docs)), [index_of[t] for t in topic_final]]

# 4. Readable labels and the results table
labels = {t: ", ".join(word for word, _ in model.get_topic(t)[:4]) for t in topic_ids}
docs["topic_id"] = topic_final
docs["topic_label"] = docs["topic_id"].map(labels)
docs["topic_source"] = source
docs["topic_similarity"] = assigned_sim.round(3)
docs.drop(columns="lead_text").to_parquet("data/processed/docs_topics.parquet", index=False)

rows = []
for t in topic_ids:
    members = docs[docs["topic_id"] == t]
    rows.append({
        "topic_id": t,
        "label": labels[t],
        "docs_model": int((members["topic_source"] == "model").sum()),
        "docs_total": len(members),
        "example_titles": " | ".join(members.nlargest(5, "topic_similarity")["title"]),
    })
pd.DataFrame(rows).sort_values("docs_total", ascending=False).to_csv(
    "results/topics.csv", index=False, encoding="utf-8")

# 5. Summary
n_core = int(core.sum())
n_out = int((core & (model_topic == -1)).sum())
print(f"\nTopics found: {len(topic_ids)}")
print(f"Outliers among long documents: {n_out} of {n_core} ({100 * n_out / n_core:.1f}%)")
print(docs["topic_source"].value_counts().to_string())
print("\nAverage similarity to topic center:")
print(docs.groupby("topic_source")["topic_similarity"].mean().round(3).to_string())
print("\nSaved data/processed/docs_topics.parquet and results/topics.csv")