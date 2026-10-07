import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from extractor import main as run_extractor
from clean_chunk import main as run_clean
from validator import validate_chunks
from classifier import extract_pdf_lines
import json


def run_full_pipeline():
    print("=" * 70)
    print("LEGAL RAG PIPELINE - FULL RUN")
    print("=" * 70)

    print("\n[1/4] Running extractor...")
    run_extractor()

    chunks_path = Path(r"E:\LegalRAG\output\chunks.jsonl")
    if not chunks_path.exists():
        print("ERROR: chunks.jsonl not found after extractor")
        return

    print("\n[2/4] Loading chunks for validation...")
    with chunks_path.open("r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(chunks)} chunks")

    print("\n[3/4] Running validation...")
    report = validate_chunks(chunks)
    print(report.summary())

    report_path = Path(r"E:\LegalRAG\output\validation_report.txt")
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report.summary())
    print(f"Validation report saved to: {report_path}")

    print("\n[4/4] Running clean_chunk...")
    run_clean()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_full_pipeline()