"""One-off: splice PaddleOCR lines into the mojibake segment (p86) of VBHN-125.

Replaces text-layer lines from the corruption start to end-of-document with
OCR lines, then parses and saves the corrected corpus. Run:
    .venv/Scripts/python.exe scripts/splice_ocr_p85_86.py

Anchors verified against VBHN-125 (86 pages, 3106 text-layer lines):
- corruption starts at line index 3065 (page 86)
- "Di~u 220" heading at line index 3079 (page 86)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from legal_rag.corpus.extractor import extract_pdf_lines, save_corpus
from legal_rag.parsing.parser import parse_legal_structure

PDF_PATH = Path("data/raw/Văn-bản-hợp-nhất-125-VBHN-VPQH.pdf")
OCR_PATH = Path("data/ocr/vbhn125_p85_86_ocr.json")
OUT_DIR = Path("data/processed")
DOCUMENT_ID = "labor_code_2019"

# First corrupted text-layer line index (0-based). Verified 2026-10-16.
SPLICE_INDEX = 3065
EXPECTED_TOTAL_LINES = 3106


def main() -> None:
    lines = extract_pdf_lines(PDF_PATH)
    assert len(lines) == EXPECTED_TOTAL_LINES, f"line count changed: {len(lines)}"
    assert lines[SPLICE_INDEX - 1]["page"] == 85, "preceding page changed"
    assert lines[SPLICE_INDEX]["page"] == 86, "splice page changed"
    assert "Di~u 220" in lines[3079]["text"], "Diu220 anchor moved"

    ocr = json.loads(OCR_PATH.read_text(encoding="utf-8"))
    ocr_p86 = [l["text"].strip() for l in ocr["86"] if l["text"].strip()]
    # Drop leading page-number line ("86").
    if ocr_p86 and ocr_p86[0] == "86":
        ocr_p86 = ocr_p86[1:]
    assert ocr_p86[0].startswith("lao"), f"unexpected OCR start: {ocr_p86[0][:40]}"
    assert any("220" in t for t in ocr_p86[:20]), "no Dieu-220 heading in OCR"

    corrected = lines[:SPLICE_INDEX] + [
        {"text": t, "page": 86} for t in ocr_p86
    ]
    print(f"text-layer kept: {SPLICE_INDEX}, OCR lines spliced: {len(ocr_p86)}")

    chunks = parse_legal_structure(corrected)
    print(f"corpus chunks: {len(chunks)}")

    arts = sorted(
        {c["article"] for c in chunks if c.get("article")},
        key=lambda a: int(a.split()[1]),
    )
    print(f"articles: {len(arts)}, last: {arts[-1]}")
    d220 = [c for c in chunks if c.get("article") == "Điều 220"]
    print(f"Dieu-220 chunks: {len(d220)}")
    for c in d220:
        print(f"  [{c['chunk_type']}] clause={c.get('clause')} :: {c['content'][:70]}")

    jsonl_path, csv_path = save_corpus(chunks, OUT_DIR, DOCUMENT_ID + "_run02")
    print(f"saved: {jsonl_path}, {csv_path}")


if __name__ == "__main__":
    main()
