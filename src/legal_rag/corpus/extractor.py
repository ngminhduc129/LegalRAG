"""
PDF extraction and legal structure parsing for corpus processing.
"""

import re
import json
import csv
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF

from legal_rag.parsing.parser import parse_legal_structure
from legal_rag.parsing.constants import DOCUMENT_NAME, DOCUMENT_NUMBER
from legal_rag.chunking.legal_aware import save_jsonl, save_csv


def clean_line(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\u00ad", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def is_page_number(text: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}", text.strip()))


def extract_pdf_lines(pdf_path: Path) -> List[Dict[str, Any]]:
    lines = []
    with fitz.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            page_num = page_index + 1
            page_text = page.get_text("text", sort=True)
            for raw_line in page_text.splitlines():
                line = clean_line(raw_line)
                if not line:
                    continue
                if is_page_number(line):
                    continue
                lines.append({
                    "text": line,
                    "page": page_num
                })
    return lines


def parse_corpus(pdf_path: Path) -> List[Dict[str, Any]]:
    """Parse PDF into structured chunks with legal hierarchy."""
    lines = extract_pdf_lines(pdf_path)
    chunks = parse_legal_structure(lines)
    return chunks


def save_corpus(chunks: List[Dict[str, Any]], output_dir: Path, document_id: str = "labor_code_2019"):
    """Save corpus chunks to JSONL and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{document_id}_corpus.jsonl"
    csv_path = output_dir / f"{document_id}_corpus_preview.csv"
    save_jsonl(chunks, jsonl_path)
    save_csv(chunks, csv_path)
    return jsonl_path, csv_path


def main():
    import sys
    pdf_path = Path(r"E:\LegalRAG\data\raw\Văn-bản-hợp-nhất-125-VBHN-VPQH.pdf")
    output_dir = Path(r"E:\LegalRAG\data\processed")

    if not pdf_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file PDF: {pdf_path}")

    print("Đang trích xuất PDF...")
    lines = extract_pdf_lines(pdf_path)
    print(f"Số dòng trích xuất: {len(lines)}")

    print("Đang phân tích cấu trúc pháp luật...")
    chunks = parse_legal_structure(lines)
    print(f"Tổng số chunk: {len(chunks)}")

    jsonl_path, csv_path = save_corpus(chunks, output_dir)

    print("\nĐã lưu kết quả:")
    print(f"JSONL: {jsonl_path}")
    print(f"CSV:   {csv_path}")

    print("\n===== 5 CHUNK ĐẦU TIÊN =====")
    for chunk in chunks[:5]:
        print("-" * 70)
        print(
            f"{chunk['chapter']} | "
            f"{chunk['section']} | "
            f"{chunk['article']} | "
            f"Khoản {chunk['clause']} | "
            f"Điểm {chunk['point']}"
        )
        print("Mục:", chunk["section_title"])
        print("Loại:", chunk["chunk_type"])
        print("Trang:", chunk["page_start"], "-", chunk["page_end"])
        print("Nội dung:", chunk["content"][:300])


if __name__ == "__main__":
    main()