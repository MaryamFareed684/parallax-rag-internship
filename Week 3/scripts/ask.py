import sys

from src.rag import answer_question

question = " ".join(sys.argv[1:]) or input("Question: ")
r = answer_question(question)

print(f"\nStatus: {r['status']}")
print(r["answer"])
for s in r["sources"]:
    print(f"  [{s['n']}] {s['title']} (distance {s['distance']})")
print(f"\nretrieval {r['retrieval_ms']} ms | generation {r['generation_ms']} ms | total {r['total_ms']} ms")