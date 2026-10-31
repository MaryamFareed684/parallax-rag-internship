import pandas as pd

labels = pd.read_csv("results/ner_labels.csv", dtype=str, encoding="utf-8-sig").fillna("")
labels = labels[labels["person"].isin(["yes", "no"])]
pred = pd.read_parquet("data/processed/chunks_nlp.parquet")[["chunk_id", "has_person", "has_place"]]
data = labels.merge(pred, on="chunk_id")
print(f"Chunks labeled by hand: {len(data)}\n")


def report(name, truth, guess):
    tp = int((truth & guess).sum())
    fp = int((~truth & guess).sum())
    fn = int((truth & ~guess).sum())
    tn = int((~truth & ~guess).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else float("nan")
    print(f"{name}: accuracy {100 * (tp + tn) / len(truth):.0f}% | precision {precision:.2f} "
          f"| recall {recall:.2f} | F1 {f1:.2f}   (TP {tp}, FP {fp}, FN {fn}, TN {tn})")


report("has_person", data["person"] == "yes", data["has_person"].astype(bool))
report("has_place ", data["place"] == "yes", data["has_place"].astype(bool))

for column, flag in (("person", "has_person"), ("place", "has_place")):
    wrong = data[(data[column] == "yes") != data[flag].astype(bool)]
    if len(wrong):
        print(f"\nMistakes for {flag}:")
        for r in wrong.itertuples():
            print(f"  - {r.title}: you said {getattr(r, column)}, spaCy said {getattr(r, flag)}")