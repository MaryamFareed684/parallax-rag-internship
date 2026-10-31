import csv
import statistics
import time

from src.filtered_search import build_where, filtered_search, find_topic
from src.search import get_model
from src.vector_store import get_collection
from src.filtered_search import NLP_COLLECTION

RUNS = 5
QUERIES = [
    "who invented the telephone", "how do plants make energy from sunlight",
    "history of the Roman Empire", "famous football players",
    "election results in Australia", "chemical elements and atoms",
    "songs and albums by bands", "rivers and villages in England",
]
music = find_topic("album")
SCENARIOS = [
    ("no filter", None, lambda m: True),
    ("topic (music)", build_where(topic_id=music), lambda m: m["topic_id"] == music),
    ("has_person", build_where(has_person=True), lambda m: m["has_person"]),
    ("place + date", build_where(has_place=True, has_date=True),
     lambda m: m["has_place"] and m["has_date"]),
    ("similarity >= 0.5", build_where(min_similarity=0.5), lambda m: m["topic_similarity"] >= 0.5),
]

collection = get_collection(name=NLP_COLLECTION)
model = get_model()
filtered_search("warm up", k=3, collection=collection, model=model)

rows = []
for name, where, rule in SCENARIOS:
    times, returned, violations = [], 0, 0
    for query in QUERIES:
        for _ in range(RUNS):
            start = time.perf_counter()
            hits = filtered_search(query, k=5, where=where, collection=collection, model=model)
            times.append((time.perf_counter() - start) * 1000)
        returned += len(hits)
        violations += sum(1 for h in hits if not rule(h["metadata"]))
    rows.append({"scenario": name, "mean_ms": round(statistics.mean(times), 1),
                 "p95_ms": round(statistics.quantiles(times, n=20)[-1], 1),
                 "avg_results": round(returned / len(QUERIES), 1), "violations": violations})
    print(rows[-1])

with open("results/filter_latency.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print("Saved results/filter_latency.csv  (violations must be 0)")