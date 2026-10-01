import pandas as pd
from src.preprocessing import clean_text, is_english

df = pd.read_json("data/raw/raw.jsonl", lines=True)
print("Raw documents:", len(df))

# 1. Clean the text
df["text"] = df["text"].apply(clean_text)

# 2. Remove very short documents
df = df[df["text"].str.len() > 200]
print("After removing short docs:", len(df))

# 3. Keep English only (check the first 1000 characters to be fast)
df = df[df["text"].apply(lambda t: is_english(t[:1000]))]
print("After language filter:", len(df))

# 4. Remove duplicates
df = df.drop_duplicates(subset="text")
print("After removing duplicates:", len(df))

# 5. Save
df.to_parquet("data/processed/clean_corpus.parquet", index=False)
print("Saved", len(df), "documents to data/processed/clean_corpus.parquet")