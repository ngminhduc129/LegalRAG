"""Trace which lines produced false-positive article numbers in nd_12_2022 and nd_70_2023."""
import sys, json
sys.path.insert(0, "src")
from pathlib import Path

# Reconstruct: find chunks whose article number is suspicious
# nd_12_2022: article 110 (false positive)
# nd_70_2023: article 23 (false positive)

for doc_id, suspicious in [("nd_12_2022", [110]), ("nd_70_2023", [23])]:
    path = Path(f"data/chunks/run16_ocr/by_document/{doc_id}.jsonl")
    print(f"\n=== {doc_id} ===")
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            art = chunk.get("article", "")
            if not art:
                continue
            try:
                num = int(art.replace("Điều ", ""))
            except ValueError:
                continue
            if num in suspicious:
                print(f"  article={art} chunk_type={chunk['chunk_type']} page={chunk['page_start']}")
                print(f"  content: {chunk['content'][:200]}")
                print()