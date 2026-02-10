import json
import os
import requests

API_BASE = os.getenv("API_BASE", "http://localhost:8001")

def main():
    with open("eval/questions.json", "r", encoding="utf-8") as f:
        qs = json.load(f)

    results = []
    for item in qs:
        q = item["question"]
        expected = [k.lower() for k in item["expected_keywords"]]

        r = requests.post(f"{API_BASE}/ask", json={"question": q})
        r.raise_for_status()
        data = r.json()

        answer = (data.get("answer") or "").lower()
        sources_text = " ".join([s.get("text","") for s in data.get("sources", [])]).lower()

        hit = any(k in answer or k in sources_text for k in expected)
        results.append({"question": q, "hit": hit, "expected": expected})

    total = len(results)
    hits = sum(1 for r in results if r["hit"])
    print(f"Eval hit-rate: {hits}/{total}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
