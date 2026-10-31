import pandas as pd

N = 25
SEED = 7
PATH = "results/ner_labels.csv"

try:
    labels = pd.read_csv(PATH, dtype=str, encoding="utf-8-sig").fillna("")
except FileNotFoundError:
    chunks = pd.read_parquet("data/processed/chunks_nlp.parquet")
    labels = chunks.sample(N, random_state=SEED)[["chunk_id", "title", "text"]].copy()
    labels["text"] = labels["text"].str[:600]
    labels["person"] = ""
    labels["place"] = ""
    labels = labels.reset_index(drop=True).astype(str)

todo = labels.index[labels["person"].str.strip() == ""].tolist()
print("Question 1 = PERSON: does the text name at least one real person?")
print("Question 2 = PLACE : does it name a country, city, region or other place?")
print(f"{len(todo)} chunks left. Type two letters, like yn  (y = yes, n = no), or q to quit\n")

for n, i in enumerate(todo, 1):
    row = labels.loc[i]
    print("-" * 70)
    print(f"[{n}/{len(todo)}] {row['title']}")
    print(row["text"])
    while True:
        answer = input("person place > ").strip().lower()
        if answer == "q":
            labels.to_csv(PATH, index=False, encoding="utf-8-sig")
            print("Saved.")
            raise SystemExit
        if len(answer) == 2 and set(answer) <= {"y", "n"}:
            labels.loc[i, "person"] = "yes" if answer[0] == "y" else "no"
            labels.loc[i, "place"] = "yes" if answer[1] == "y" else "no"
            break
        print("Type exactly two letters made of y and n, for example: yn")
    labels.to_csv(PATH, index=False, encoding="utf-8-sig")
print("All done and saved.")