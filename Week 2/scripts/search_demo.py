import sys

from src.search import search

query = " ".join(sys.argv[1:]) or "What is the capital of France?"
for hit in search(query, k=3):
    print(f"[{hit['distance']:.3f}] {hit['title']} (chunk {hit['chunk_index']})")
    print("   ", hit["text"][:200], "...\n")