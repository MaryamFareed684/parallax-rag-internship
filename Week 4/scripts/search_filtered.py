import argparse

from src.filtered_search import build_where, filtered_search, find_topic

parser = argparse.ArgumentParser()
parser.add_argument("query")
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--topic", help="a word from a topic label, for example: music")
parser.add_argument("--person", action="store_true")
parser.add_argument("--org", action="store_true")
parser.add_argument("--place", action="store_true")
parser.add_argument("--date", action="store_true")
parser.add_argument("--min-sim", type=float, help="minimum topic similarity, for example 0.5")
args = parser.parse_args()

where = build_where(
    topic_id=find_topic(args.topic) if args.topic else None,
    has_person=True if args.person else None,
    has_org=True if args.org else None,
    has_place=True if args.place else None,
    has_date=True if args.date else None,
    min_similarity=args.min_sim,
)
print("filter:", where)
hits = filtered_search(args.query, k=args.k, where=where)
if not hits:
    print("No results match this filter.")
for h in hits:
    m = h["metadata"]
    print(f"[{h['distance']:.3f}] {h['title']} | topic: {m['topic_label']} "
          f"| person={m['has_person']} place={m['has_place']}")
    print("   ", h["text"][:150], "...\n")