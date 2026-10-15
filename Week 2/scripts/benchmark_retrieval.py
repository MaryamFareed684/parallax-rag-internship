import csv
import statistics
import time

from src.search import get_model, search
from src.vector_store import get_collection

RUNS = 5

QUERIES = [
    ("short keyword", "photosynthesis"),
    ("short keyword", "Napoleon"),
    ("very short", "dog"),
    ("question", "who invented the telephone"),
    ("question", "how do plants make energy from sunlight"),
    ("question", "why did the Roman Empire fall"),
    ("long", "I want to understand how the French Revolution started, what the main "
             "causes were, and how it changed the government of France afterwards"),
    ("long", "Explain the difference between a virus and a bacterium and how each "
             "one causes disease in the human body"),
    ("typo", "photosyntesis in plnts"),
    ("typo", "who invnted the telefone"),
    ("off-topic", "best pizza recipe with extra cheese"),
    ("off-topic", "how to fix a flat bicycle tire"),
]

K_VALUES = [1, 5, 10, 50]


def timed_ms(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1000


def main():
    collection = get_collection()
    model = get_model()
    print("Chunks in database:", collection.count())

    # Warm-up: the first search is always slower
    search("warm up", k=5, collection=collection, model=model)

    rows = []
    all_times = []
    for category, query in QUERIES:
        totals, embeds = [], []
        hits = []
        for _ in range(RUNS):
            hits, total_ms = timed_ms(search, query, k=5, collection=collection, model=model)
            _, embed_ms = timed_ms(model.encode, [query], normalize_embeddings=True)
            totals.append(total_ms)
            embeds.append(embed_ms)
        all_times.extend(totals)
        rows.append({
            "category": category,
            "query": query,
            "mean_ms": round(statistics.mean(totals), 1),
            "min_ms": round(min(totals), 1),
            "max_ms": round(max(totals), 1),
            "embed_ms": round(statistics.mean(embeds), 1),
            "top1_title": hits[0]["title"],
            "top1_distance": round(hits[0]["distance"], 3),
        })

    # Different numbers of results (k)
    k_rows = []
    for k in K_VALUES:
        times = [timed_ms(search, "who invented the telephone", k=k,
                          collection=collection, model=model)[1] for _ in range(RUNS)]
        k_rows.append({"k": k, "mean_ms": round(statistics.mean(times), 1)})

    # Save results
    with open("results/retrieval_latency.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with open("results/retrieval_latency_by_k.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "mean_ms"])
        writer.writeheader()
        writer.writerows(k_rows)

    # Print a readable summary
    print(f"\n{'category':<14}{'mean ms':>9}{'embed ms':>10}{'dist':>8}  query")
    for r in rows:
        print(f"{r['category']:<14}{r['mean_ms']:>9}{r['embed_ms']:>10}"
              f"{r['top1_distance']:>8}  {r['query'][:45]}")

    p50 = statistics.median(all_times)
    p95 = statistics.quantiles(all_times, n=20)[-1]
    print(f"\nOverall over {len(all_times)} searches: "
          f"mean {statistics.mean(all_times):.1f} ms, p50 {p50:.1f} ms, p95 {p95:.1f} ms")

    print("\nLatency by number of results (k):")
    for r in k_rows:
        print(f"  k={r['k']:<3} {r['mean_ms']} ms")
    print("\nSaved to results/retrieval_latency.csv and results/retrieval_latency_by_k.csv")


if __name__ == "__main__":
    main()