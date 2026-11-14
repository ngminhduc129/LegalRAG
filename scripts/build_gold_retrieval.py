#!/usr/bin/env python3
"""Sinh tập gold retrieval từ corpus chunks đã build (run-scoped).

Quy tac tu AGENTS.md:
- Khong bi so lieu: moi ref trong 'relevant' phai truy nguc ve mot chunk thuc su
  co trong corpus (document_id + article + clause).
- Khong do du lieu phap luat khi khong chac.
- Mot chay co run_id duyiet, khong ghi de ket qua cu.

Chay:
    python scripts/build_gold_retrieval.py \
        --chunks data/chunks/full_gold_20261114/all_legal_aware.jsonl \
        --out data/qa/gold_retrieval.jsonl \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from legal_rag.corpus.registry import load_registry

TARGETS: Dict[str, Dict[str, int]] = {
    "bo_luat": {"dev": 60, "test": 180},
    "nghi_dinh": {"dev": 25, "test": 75},
    "thong_tu": {"dev": 10, "test": 30},
    "quyet_dinh": {"dev": 0, "test": 0},
    "cross": {"dev": 5, "test": 15},
}


def normalize_article(value: Any) -> str:
    """Tra ve so Dieu rieng (\"70\") tu \"Dieu 70\" hoac 70."""
    if value is None:
        return ""
    m = re.search(r"(\d+)", str(value))
    return m.group(1) if m else str(value).strip()


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def snippet_of(content: str, max_chars: int = 160) -> str:
    """Lay cau dau tien co nghia cua chunk de lam co so cau hoi."""
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text or content[:max_chars]


def make_question(chunk: Dict[str, Any]) -> str:
    article = chunk.get("article") or ""
    clause = chunk.get("clause")
    doc_name = chunk.get("document_name") or chunk.get("document_id") or ""
    content = chunk.get("content") or ""
    snippet = snippet_of(content)
    if clause:
        return f"Theo {article} Khoan {clause} cua {doc_name}, {snippet}"
    return f"Theo {article} cua {doc_name}, {snippet}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", required=True, help="Duong dan all_<strategy>.jsonl cua run chunking")
    ap.add_argument("--out", required=True, help="Duong dan file gold retrieval JSONL dau ra")
    ap.add_argument("--seed", type=int, default=42, help="Random seed de chia dev/test duoc lap lap")
    args = ap.parse_args()

    chunks_path = Path(args.chunks)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    registry = load_registry()
    doc_type_by_id = {d.document_id: d.profile for d in registry.documents}
    doc_name_by_id = {d.document_id: d.document_name for d in registry.documents}

    chunks = load_chunks(chunks_path)
    eligible = [c for c in chunks if c.get("retrieval_eligible") is not False]
    print(f"Tong chunks: {len(chunks)} | retrieval_eligible: {len(eligible)}")

    # Nhom chunks theo document_id, chi lay cac chunk co article (loai preamble/appendix khong co Dieu)
    by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in eligible:
        article = normalize_article(c.get("article"))
        if not article:
            continue
        by_doc[c["document_id"]].append(c)

    # Nhom theo document_type (profile tu registry)
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for did, cs in by_doc.items():
        dtype = doc_type_by_id.get(did, "unknown")
        by_type[dtype].extend(cs)

    print("Chunks co article boi loai van ban:")
    for dtype, cs in sorted(by_type.items()):
        docs = sorted({c["document_id"] for c in cs})
        print(f"  {dtype:12s} chunks={len(cs):5d} documents={docs}")

    rng = random.Random(args.seed)
    records: List[Dict[str, Any]] = []
    used_chunk_indices: set[int] = set()

    def take(pool: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
        """Lay n chunk khac nhau tu pool (khong trung chunk da su dung)."""
        avail = [c for c in pool if id(c) not in used_chunk_indices]
        if len(avail) < n:
            print(f"  [canh bao] chi con {len(avail)}/{n} chunk chua su dung trong nhom")
            chosen = avail
        else:
            chosen = rng.sample(avail, n)
        for c in chosen:
            used_chunk_indices.add(id(c))
        return chosen

    counter = 0

    def emit(doc_id: str, chunk: Dict[str, Any], split: str, qtype: str, notes: str):
        nonlocal counter
        counter += 1
        article = normalize_article(chunk.get("article"))
        clause = chunk.get("clause")
        ref: Dict[str, Any] = {
            "document_id": doc_id,
            "article": article,
        }
        if clause:
            ref["clause"] = str(clause)
        records.append({
            "id": f"gold_{doc_id}_{counter:04d}",
            "question": make_question(chunk),
            "relevant": [ref],
            "split": split,
            "question_type": qtype,
            "notes": notes,
        })

    # --- Nhom 1: theo loai van ban (bo_luat, nghi_dinh, thong_tu, quyet_dinh) ---
    for dtype, targets in TARGETS.items():
        if dtype == "cross":
            continue
        pool = by_type.get(dtype, [])
        if not pool:
            print(f"  [canh bao] khong co chunk cho loai {dtype}, bo qua")
            continue
        for split in ("dev", "test"):
            n = targets.get(split, 0)
            if n <= 0:
                continue
            chosen = take(pool, n)
            for i, c in enumerate(chosen):
                did = c["document_id"]
                dname = doc_name_by_id.get(did, did)
                emit(
                    did,
                    c,
                    split,
                    f"{dtype}_factoid",
                    f"sinh tu chunk {c.get('chunk_id')} ({dname} {c.get('article')} khoan {c.get('clause')})",
                )

    # --- Nhom 2: cross-document (2 document khac nhau, 1 cau hoi, 2 ref) ---
    def emit_cross(c1: Dict[str, Any], c2: Dict[str, Any], split: str):
        nonlocal counter
        if c1["document_id"] == c2["document_id"]:
            return
        refs = []
        for c in (c1, c2):
            ref = {
                "document_id": c["document_id"],
                "article": normalize_article(c.get("article")),
            }
            if c.get("clause"):
                ref["clause"] = str(c["clause"])
            refs.append(ref)
        counter += 1
        records.append({
            "id": f"gold_cross_{counter:04d}",
            "question": (
                f"So sanh {c1.get('article')} cua {c1.get('document_name', c1['document_id'])} "
                f"va {c2.get('article')} cua {c2.get('document_name', c2['document_id'])}: "
                f"{snippet_of(c1.get('content',''))[:80]} ... {snippet_of(c2.get('content',''))[:80]}"
            ),
            "relevant": refs,
            "split": split,
            "question_type": "cross_document_factoid",
            "notes": f"cross-document: {c1['document_id']} {c1.get('article')} + {c2['document_id']} {c2.get('article')}",
        })

    # Pool tat ca chunk chua su dung, sap xep ngau nhien de tao cap cross-document du so luong.
    all_avail = [c for cs in by_type.values() for c in cs if id(c) not in used_chunk_indices]
    rng.shuffle(all_avail)

    for split in ("dev", "test"):
        target = TARGETS["cross"].get(split, 0)
        made = 0
        i = 0
        while made < target and i + 1 < len(all_avail):
            c1 = all_avail[i]
            c2 = all_avail[i + 1]
            i += 2
            if c1["document_id"] == c2["document_id"]:
                # tra lai c2 vao cuoi pool de tim cap khac document
                all_avail.append(c2)
                continue
            # danh dau da su dung
            used_chunk_indices.add(id(c1))
            used_chunk_indices.add(id(c2))
            emit_cross(c1, c2, split)
            made += 1
        if made < target:
            print(f"  [canh bao] cross-document {split}: chi tao duoc {made}/{target} cau hoi")

    # Sap xep theo split va id, xoa trung lap id
    seen_ids = set()
    unique: List[Dict[str, Any]] = []
    for r in records:
        if r["id"] in seen_ids:
            continue
        seen_ids.add(r["id"])
        unique.append(r)
    unique.sort(key=lambda r: (r["split"], r["id"]))

    with out_path.open("w", encoding="utf-8") as handle:
        for r in unique:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")

    dev_n = sum(1 for r in unique if r["split"] == "dev")
    test_n = sum(1 for r in unique if r["split"] == "test")
    print(f"\nDa ghi {len(unique)} cau hoi -> {out_path}")
    print(f"  dev={dev_n}  test={test_n}")
    print(f"  targets: dev={sum(t['dev'] for t in TARGETS.values())} test={sum(t['test'] for t in TARGETS.values())}")


if __name__ == "__main__":
    main()
