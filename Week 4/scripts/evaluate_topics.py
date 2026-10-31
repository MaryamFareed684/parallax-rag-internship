import pandas as pd


def read(path):
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"Could not read {path}")


topics = read("results/topic_review.csv")
topics["rating"] = topics["rating"].astype(str).str.strip().str.lower()
rated = topics[topics["rating"].isin(["good", "mixed", "bad"])]
if rated.empty:
    raise SystemExit("No topics rated yet. Run: python -m scripts.rate topics")

coverage = 100 * rated["docs_total"].sum() / topics["docs_total"].sum()
print(f"=== Topic quality ({len(rated)} of {len(topics)} topics rated by hand) ===")
print(f"The rated topics contain {coverage:.0f}% of all documents")
counts = rated["rating"].value_counts()
for r in ("good", "mixed", "bad"):
    n = int(counts.get(r, 0))
    print(f"{r:<6} {n:>3}  ({100 * n / len(rated):.0f}%)")

sheet = read("results/assignment_review.csv")
sheet["judgment"] = sheet["judgment"].astype(str).str.strip().str.lower()
judged = sheet[sheet["judgment"].isin(["yes", "partly", "no"])]
if judged.empty:
    raise SystemExit("No assignments judged yet. Run: python -m scripts.rate assignments")

print(f"\n=== Does the assigned topic fit? ({len(judged)} of {len(sheet)} documents judged) ===")
print(f"{'group':<20}{'docs':>5}{'yes':>6}{'partly':>8}{'no':>5}{'strict':>9}{'lenient':>9}")
for group, part in judged.groupby("group"):
    yes = int((part["judgment"] == "yes").sum())
    partly = int((part["judgment"] == "partly").sum())
    no = int((part["judgment"] == "no").sum())
    print(f"{group:<20}{len(part):>5}{yes:>6}{partly:>8}{no:>5}"
          f"{100 * yes / len(part):>8.0f}%{100 * (yes + partly) / len(part):>8.0f}%")
yes_all = int((judged["judgment"] == "yes").sum())
print(f"{'all':<20}{len(judged):>5}{'':>6}{'':>8}{'':>5}{100 * yes_all / len(judged):>8.0f}%")
print("strict = only 'yes' counts; lenient = 'yes' and 'partly' count")
print("Small samples (about 10 per group): treat these percentages as rough.")