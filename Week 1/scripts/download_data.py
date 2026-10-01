import json
from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

count = 0
with open("data/raw/raw.jsonl", "w", encoding="utf-8") as f:
    for row in ds:
        f.write(json.dumps(
            {"id": row["id"], "title": row["title"], "text": row["text"]}
        ) + "\n")
        count += 1
        if count >= 6000:
            break

print(count, "documents saved")