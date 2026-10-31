import sys

import pandas as pd

MODES = {
    "topics": ("results/topic_review.csv", "rating", {"g": "good", "m": "mixed", "b": "bad"}),
    "assignments": ("results/assignment_review.csv", "judgment", {"y": "yes", "p": "partly", "n": "no"}),
}

mode = sys.argv[1] if len(sys.argv) > 1 else ""
if mode not in MODES:
    raise SystemExit("Use: python -m scripts.rate topics   or   python -m scripts.rate assignments")
path, column, options = MODES[mode]

df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
todo = df.index[df[column].str.strip() == ""].tolist()
keys = ", ".join(f"{k}={v}" for k, v in options.items())
print(f"{len(todo)} rows left. Type {keys}, s=skip, q=quit and save\n")

for count, i in enumerate(todo, 1):
    row = df.loc[i]
    print("-" * 70)
    print(f"[{count}/{len(todo)}]")
    if mode == "topics":
        print("Topic label:", row["label"])
        print("Top titles :", row["top_titles"])
        print("Random     :", row["random_titles"])
    else:
        print("Title         :", row["title"])
        print("Assigned topic:", row["assigned_topic"])
        print("Text start    :", row["text_start"])
    while True:
        answer = input("> ").strip().lower()
        if answer in options:
            df.loc[i, column] = options[answer]
            if mode == "topics":
                df.loc[i, "notes"] = input("note (optional, Enter to skip): ").strip()
            break
        if answer == "s":
            break
        if answer == "q":
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print("Saved. Run the same command again to continue.")
            raise SystemExit
        print("Please type one of:", ", ".join(options), "s or q")
    df.to_csv(path, index=False, encoding="utf-8-sig")
print("All done and saved.")