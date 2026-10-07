import re
import json
import csv
from pathlib import Path

import fitz  # PyMuPDF

from parser import parse_legal_structure
from constants import DOCUMENT_NAME, DOCUMENT_NUMBER


PDF_PATH = Path(
    r"E:\LegalRAG\data\Văn-bản-hợp-nhất-125-VBHN-VPQH.pdf"
)

OUTPUT_DIR = Path(r"E:\LegalRAG\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JSONL_PATH = OUTPUT_DIR / "chunks.jsonl"
CSV_PATH = OUTPUT_DIR / "chunks_preview.csv"

DEBUG = True


def clean_line(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\u00ad", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def is_page_number(text: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}", text.strip()))


def extract_pdf_lines(pdf_path: Path):
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


def save_jsonl(chunks, output_path):
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def save_csv(chunks, output_path):
    if not chunks:
        print("Không có chunk để ghi CSV.")
        return

    fieldnames = list(chunks[0].keys())
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(chunks)


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file PDF: {PDF_PATH}")

    print("Đang trích xuất PDF...")
    lines = extract_pdf_lines(PDF_PATH)
    print(f"Số dòng trích xuất: {len(lines)}")

    print("Đang phân tích cấu trúc pháp luật...")
    chunks = parse_legal_structure(lines)
    print(f"Tổng số chunk: {len(chunks)}")

    save_jsonl(chunks, JSONL_PATH)
    save_csv(chunks, CSV_PATH)

    print("\nĐã lưu kết quả:")
    print(f"JSONL: {JSONL_PATH}")
    print(f"CSV:   {CSV_PATH}")

    if DEBUG:
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