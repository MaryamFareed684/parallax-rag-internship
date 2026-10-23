import time

from src.rag import answer_question

for q in ["What is albedo?", "What is anarchism?"]:
    for attempt in range(1, 6):
        r = answer_question(q, log=False)
        g = r.get("grounding") or {}
        print(f"{q} | try {attempt} | status={r['status']} | model={r.get('model')} "
              f"| retries={r.get('retries')} | reasons={g.get('reasons')} | support={g.get('support')}")
        if r["status"] == "flagged":
            print("   RAW ANSWER:", r.get("raw_answer"))
        time.sleep(2)