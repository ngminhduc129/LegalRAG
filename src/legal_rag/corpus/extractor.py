"""PDF extraction and legal structure parsing for corpus processing."""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import fitz  # PyMuPDF

from legal_rag.chunking.legal_aware import save_csv, save_jsonl
from legal_rag.corpus.patches import apply_page_patches
from legal_rag.corpus.registry import DocumentMeta, resolve_document
from legal_rag.parsing.classifier import clean_line, is_page_number
from legal_rag.parsing.parser import ParseResult, parse_legal_structure


def _page_lines(text: str, page_num: int, source: str) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = clean_line(raw_line)
        if not line:
            continue
        if is_page_number(line):
            continue
        lines.append({"text": line, "page": page_num, "text_source": source})
    return lines


def extract_pdf_lines(
    pdf_path: Path,
    ocr_pages: Optional[Mapping[int, Sequence[str]]] = None,
) -> List[Dict[str, Any]]:
    """Extract page lines, preferring OCR text for pages selected for OCR.

    ``ocr_pages`` maps a 1-based page number to already-OCR'd lines.  A page is
    in that mapping only when the audit decided its text layer is missing or
    unusable (for example a scanned page that carries only a header stamp), so
    its OCR text wins over the fragmentary text layer.  Pages that are absent
    from the mapping keep their embedded text layer.
    """
    lines: List[Dict[str, Any]] = []
    ocr_pages = ocr_pages or {}
    with fitz.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            page_num = page_index + 1
            ocr_lines = ocr_pages.get(page_num)
            if ocr_lines:
                lines.extend(_page_lines("\n".join(ocr_lines), page_num, "ocr"))
                continue
            page_text = page.get_text("text", sort=True)
            if page_text.strip():
                lines.extend(_page_lines(page_text, page_num, "text_layer"))
    return lines


def parse_corpus(
    pdf_path: Path,
    document: DocumentMeta | str | None = None,
    patches: Optional[List[Dict[str, Any]]] = None,
    ocr_pages: Optional[Mapping[int, Sequence[str]]] = None,
) -> ParseResult:
    """Parse PDF into structured chunks with legal hierarchy.

    ``patches`` (see legal_rag.corpus.patches) are applied to the extracted
    lines before parsing, fixing pages whose text layer is broken.
    ``ocr_pages`` supplies OCR text for pages without an embedded text layer.
    """
    meta = resolve_document(document)
    lines = extract_pdf_lines(pdf_path, ocr_pages=ocr_pages)
    if patches:
        lines, _ = apply_page_patches(lines, patches)
    return parse_legal_structure(lines, meta)


def save_corpus(
    chunks: List[Dict[str, Any]],
    output_dir: Path,
    document_id: str = "unknown",
):
    """Save corpus chunks to JSONL and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{document_id}_corpus.jsonl"
    csv_path = output_dir / f"{document_id}_corpus_preview.csv"
    save_jsonl(chunks, jsonl_path)
    save_csv(chunks, csv_path)
    return jsonl_path, csv_path


def main():  # pragma: no cover - manual entry point
    import sys

    pdf_path = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/raw/Văn-bản-hợp-nhất-125-VBHN-VPQH.pdf"
    )
    output_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "data/processed")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print("Extracting PDF...")
    lines = extract_pdf_lines(pdf_path)
    print(f"Extracted lines: {len(lines)}")

    print("Parsing legal structure...")
    result = parse_legal_structure(lines)
    print(f"Total chunks: {len(result.chunks)}")

    jsonl_path, csv_path = save_corpus(result.chunks, output_dir, pdf_path.stem)
    print(f"JSONL: {jsonl_path}")
    print(f"CSV:   {csv_path}")


if __name__ == "__main__":
    main()
