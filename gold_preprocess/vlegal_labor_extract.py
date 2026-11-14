#!/usr/bin/env python3
"""Trích bản ghi liên quan Bộ luật Lao động từ VLegal-Bench -> gold nháp (jsonl).

Mình chưa thấy schema thật của từng tác vụ nên script dò tên trường theo danh sách
khả dĩ (Q_KEYS / A_KEYS / ID_KEYS) và LUÔN giữ nguyên bản ghi gốc trong trường "raw".
Chạy --inspect trước để xem tên trường, sửa 3 danh sách đó nếu dò sai.

Cách chạy (PowerShell, đứng ở E:\\LegalRAG):
  python gold_preprocess\\vlegal_labor_extract.py --inspect
  python gold_preprocess\\vlegal_labor_extract.py --tasks 3.1 3.5 2.2
  python gold_preprocess\\vlegal_labor_extract.py            # mọi tác vụ
"""
import argparse
import json
import re
from pathlib import Path

# ---- sửa theo schema thật nếu cần -------------------------------------------
Q_KEYS = ["question", "query", "prompt", "scenario", "input", "instruction", "text", "context"]
A_KEYS = ["answer", "output", "label", "response", "reference", "gold", "target", "ground_truth", "expected"]
ID_KEYS = ["id", "qid", "question_id", "uid"]

# tên văn bản (regex) -> doc_id trong corpus của bạn. BỔ SUNG cho khớp registry.
DOC_MAP = [
    (re.compile(r"Bộ luật Lao động(?!.*2012)|45/2019/QH14", re.I), "labor_code_2019"),
    (re.compile(r"Nghị định\s+(số\s+)?145/2020", re.I), "nd_145_2020"),
    (re.compile(r"Nghị định\s+(số\s+)?135/2020", re.I), "nd_135_2020"),
    (re.compile(r"Nghị định\s+(số\s+)?152/2020", re.I), "nd_152_2020"),
    (re.compile(r"Nghị định\s+(số\s+)?12/2022", re.I), "nd_12_2022"),
    (re.compile(r"Thông tư\s+(số\s+)?11/2020", re.I), "tt_11_2020"),
]
OLD_LAW = re.compile(r"Bộ luật Lao động\s*(năm\s*)?2012|10/2012/QH13", re.I)
LABOR = re.compile(r"Bộ luật Lao động|Bộ luật lao động|45/2019/QH14", re.I)
# -----------------------------------------------------------------------------

_STOP = r"(?:Điều|khoản|điểm|và|hoặc|quy|thì|để|của|tại|theo|nên|cho)\b"
_WORD = r"(?!" + _STOP + r")[^\W\d_]+"
NAME = (
    r"(?:Nghị định|Thông tư)\s+(?:số\s+)?\d+/\d{4}(?:/[A-ZĐ\-]+)?"
    r"|(?:Bộ luật|Luật)\s+(?:" + _WORD + r"\s+){1,5}?(?:năm\s+)?\d{4}"
    r"|(?:Bộ luật|Luật)\s+" + _WORD + r"(?:\s+" + _WORD + r"){0,3}"
)
REF = re.compile(
    r"(?:điểm\s+([a-zđ])\s+)?(?:khoản\s+(\d+)\s+)?Điều\s+(\d+)\s*"
    r"(?:của\s+|,\s*)?(" + NAME + r")?",
    re.I,
)


def leaves(obj):
    """Mọi chuỗi trong bản ghi (kể cả lồng nhau)."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from leaves(v)


def pick(rec, keys):
    low = {k.lower(): k for k in rec}
    for k in keys:
        if k in low:
            v = rec[low[k]]
            return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return ""


def map_doc(name):
    if not name:
        return None
    for rx, did in DOC_MAP:
        if rx.search(name):
            return did
    return None


def extract_refs(text):
    mapped, unmapped, seen = [], [], set()
    for m in REF.finditer(text):
        diem, khoan, dieu, name = m.groups()
        name = (name or "").strip()
        key = (name, dieu, khoan, diem)
        if key in seen:
            continue
        seen.add(key)
        did = map_doc(name)
        item = {"doc_id": did, "dieu": int(dieu), "khoan": int(khoan) if khoan else None,
                "diem": diem, "law_name": name or None}
        (mapped if did else unmapped).append(item)
    return mapped, unmapped


def read_jsonl(p):
    out = []
    for i, line in enumerate(open(p, encoding="utf-8"), 1):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  [cảnh báo] {p.name} dòng {i}: JSON lỗi, bỏ qua")
    return out


def task_of(path, root):
    return path.relative_to(root).parts[0]  # '3.1', '5.3', ...


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/vlegal")
    ap.add_argument("--tasks", nargs="*", help="chỉ lấy các tác vụ này, ví dụ 3.1 3.5 (mặc định: tất cả)")
    ap.add_argument("--out", default="data/gold/vlegal_labor_gold_draft.jsonl")
    ap.add_argument("--inspect", action="store_true", help="chỉ in trường và 1 bản ghi mẫu mỗi tác vụ")
    ap.add_argument("--maxlen", type=int, default=300)
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(root.rglob("*.jsonl"))
    if not files:
        raise SystemExit(f"Không thấy .jsonl trong {root}")

    out, stats = [], {}
    for f in files:
        task = task_of(f, root)
        if args.tasks and task not in args.tasks:
            continue
        rows = read_jsonl(f)
        st = stats.setdefault(task, {"tong": 0, "lao_dong": 0, "co_ref": 0})
        for idx, r in enumerate(rows):
            st["tong"] += 1
            text = "\n".join(leaves(r))
            if not LABOR.search(text):
                continue
            st["lao_dong"] += 1

            if args.inspect:
                if st["lao_dong"] == 1:
                    print(f"\n=== {f.relative_to(root).as_posix()} | trường: {list(r.keys())}")
                    for k, v in r.items():
                        s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                        print(f"  {k}: {s[: args.maxlen]}")
                continue

            question = pick(r, Q_KEYS)
            answer = pick(r, A_KEYS)
            # ref lấy từ đáp án trước (căn cứ), nếu không có thì từ toàn bản ghi
            mapped, unmapped = extract_refs(answer)
            if not mapped:
                mapped, unmapped = extract_refs(text)
            flags = []
            if OLD_LAW.search(text):
                flags.append("nhac_luat_2012")
            if not mapped:
                flags.append("khong_co_ref_map_duoc")
            else:
                st["co_ref"] += 1
            if not question:
                flags.append("khong_doan_duoc_cau_hoi")
            rid = pick(r, ID_KEYS) or f"{f.stem}_{idx}"
            out.append({
                "qid": f"vlegal_{task}_{rid}",
                "task": task,
                "question": question,
                "gold_refs": mapped,
                "unmapped_refs": unmapped,
                "gold_answer": answer,
                "flags": flags,
                "source": "vlegal_bench",
                "reviewed": False,
                "raw": r,
            })

    if args.inspect:
        return

    print(f"{'tác vụ':8s} {'tổng':>6s} {'lao động':>9s} {'có ref map':>11s}")
    for t, s in sorted(stats.items()):
        print(f"{t:8s} {s['tong']:6d} {s['lao_dong']:9d} {s['co_ref']:11d}")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    good = [x for x in out if x["gold_refs"] and x["question"]]
    print(f"\nĐã ghi {len(out)} bản ghi -> {outp}")
    print(f"Dùng được ngay (có câu hỏi + ref map): {len(good)}")
    print("Cột 'flags' cho biết bản ghi cần xem tay. Nhớ review mẫu trước khi tin.")


if __name__ == "__main__":
    main()