"""Splice the human transcription of page 86 into the VBHN-125 line stream.

Page 86 of the source PDF is a raster scan; its hidden OCR text layer
(font "HiddenHorzOCR") is lossy non-Vietnamese OCR noise and unusable for
diacritics. The corrected text comes from data/ocr/p86_transcription.json,
transcribed by reading a 600-DPI render of the scan, and is cross-checked
(diacritics stripped) against the PaddleOCR output in
data/ocr/vbhn125_p85_86_ocr.json.

Run:
    .venv/Scripts/python.exe scripts/splice_p86_transcription.py --run-id run03

Outputs (each run writes new files; never overwrites an older run):
    data/processed/<document_id>_<run_id>_corpus.jsonl (+ _preview.csv)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from legal_rag.corpus.extractor import extract_pdf_lines, save_corpus
from legal_rag.parsing.classifier import normalize_vi
from legal_rag.parsing.parser import parse_legal_structure

# Verified against VBHN-125 (86 pages, read-only): page 85 (clean text layer)
# ends at line index 3064; page 86 (scan, corrupted hidden layer) starts at
# line index 3065. Total extracted lines: 3106.
SPLICE_INDEX = 3065
EXPECTED_TOTAL_LINES = 3106
EXPECTED_PAGE_LINES = 27


def stripped_blob(texts) -> str:
    """Diacritics-stripped, letters-only blob for cross-checking."""
    return normalize_vi(" ".join(texts))


def lcs_len(a: str, b: str) -> int:
    """Length of the longest common subsequence of two strings."""
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b):
            if ca == cb:
                cur[j + 1] = prev[j] + 1
            else:
                cur[j + 1] = prev[j + 1] if prev[j + 1] >= cur[j] else cur[j]
        prev = cur
    return prev[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pdf", type=Path,
        default=Path("data/raw/Văn-bản-hợp-nhất-125-VBHN-VPQH.pdf"),
    )
    ap.add_argument(
        "--transcription", type=Path,
        default=Path("data/ocr/p86_transcription.json"),
    )
    ap.add_argument(
        "--ocr", type=Path,
        default=Path("data/ocr/vbhn125_p85_86_ocr.json"),
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--document-id", default="labor_code_2019")
    ap.add_argument("--run-id", required=True, help="new run id, e.g. run03; never reuse")
    ap.add_argument(
        "--min-ocr-coverage", type=float, default=0.82,
        help="abort if the OCR letter coverage of the transcription falls lower",
    )
    args = ap.parse_args()

    lines = extract_pdf_lines(args.pdf)
    assert len(lines) == EXPECTED_TOTAL_LINES, f"line count changed: {len(lines)}"
    assert lines[SPLICE_INDEX - 1]["page"] == 85, "preceding page changed"
    assert lines[SPLICE_INDEX]["page"] == 86, "splice page changed"
    assert any("Di~u 220" in l["text"] for l in lines[SPLICE_INDEX:]), \
        "hidden-layer corruption no longer present - re-review source"

    data = json.loads(args.transcription.read_text(encoding="utf-8"))
    page_lines = [l.strip() for l in data["lines"] if l.strip()]
    assert len(page_lines) == EXPECTED_PAGE_LINES, \
        f"unexpected transcription length: {len(page_lines)}"
    assert page_lines[0].startswith("lao động mà"), page_lines[0][:40]
    assert any(l.startswith("Điều 220.") for l in page_lines), "no Dieu-220 heading"
    assert page_lines[-1].endswith("trong Bộ luật này."), page_lines[-1][-40:]

    # Cross-check against the PaddleOCR run (letters only, diacritics
    # stripped). That OCR drops whole accented letters, so the meaningful
    # metric is subsequence containment: LCS(transcription, ocr)/len(ocr).
    ocr = json.loads(args.ocr.read_text(encoding="utf-8"))
    ocr_texts = [
        l["text"] for l in ocr["86"]
        if l["text"].strip() and l["text"].strip() != "86"
    ]
    a = stripped_blob(page_lines)
    b = stripped_blob(ocr_texts)
    lcs = lcs_len(a, b)
    coverage = lcs / len(b) if b else 0.0
    print(f"cross-check blobs: transcription={len(a)} chars, ocr={len(b)} chars, LCS={lcs}")
    print(f"OCR cross-check coverage (LCS/OCR): {coverage:.4f}")
    if coverage < args.min_ocr_coverage:
        raise SystemExit(
            f"coverage {coverage:.4f} < {args.min_ocr_coverage} - review transcription"
        )

    corrected = lines[:SPLICE_INDEX] + [
        {"text": t, "page": 86} for t in page_lines
    ]
    print(f"text-layer kept: {SPLICE_INDEX} lines; transcription spliced: {len(page_lines)} lines")

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

    jsonl_path, csv_path = save_corpus(
        chunks, args.out_dir, f"{args.document_id}_{args.run_id}"
    )
    print(f"saved: {jsonl_path}, {csv_path}")


if __name__ == "__main__":
    main()
