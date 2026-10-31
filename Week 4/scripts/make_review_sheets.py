import pandas as pd

SEED = 42
PER_GROUP = 20

docs = pd.read_parquet("data/processed/docs_topics.parquet")
lead = pd.read_parquet("data/processed/docs.parquet")[["doc_id", "lead_text"]]
docs = docs.merge(lead, on="doc_id")
topics = pd.read_csv("results/topics.csv")

# Sheet 1: one row per topic
rows = []
for t in topics.itertuples():
    members = docs[(docs["topic_id"] == t.topic_id) & (docs["topic_source"] == "model")]
    random_titles = members.sample(min(5, len(members)), random_state=SEED)["title"].tolist()
    rows.append({
        "topic_id": t.topic_id,
        "label": t.label,
        "docs_total": t.docs_total,
        "top_titles": t.example_titles,
        "random_titles": " | ".join(random_titles),
        "rating": "",
        "notes": "",
    })
pd.DataFrame(rows).to_csv("results/topic_review.csv", index=False, encoding="utf-8-sig")

# Sheet 2: does the assigned topic fit? (three groups of documents)
parts = []
for source in ("model", "outlier_reassigned", "short_reassigned"):
    part = docs[docs["topic_source"] == source].sample(PER_GROUP, random_state=SEED)
    parts.append(part)
sample = pd.concat(parts)
sheet = pd.DataFrame({
    "group": sample["topic_source"],
    "doc_id": sample["doc_id"],
    "title": sample["title"],
    "assigned_topic": sample["topic_label"],
    "text_start": sample["lead_text"].str[:250],
    "judgment": "",
})
sheet.sample(frac=1, random_state=SEED).to_csv(
    "results/assignment_review.csv", index=False, encoding="utf-8-sig")
print("Created results/topic_review.csv and results/assignment_review.csv")