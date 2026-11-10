"""Corpus audit: which PDFs/pages actually contain an extractable text layer.

The audit is deliberately read-only.  It never modifies ``data/raw`` and is the
input for deciding which pages need OCR.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import fitz  # PyMuPDF

from legal_rag.corpus.registry import DocumentMeta, DocumentRegistry, load_registry

DEFAULT_MIN_CHARS = 20
# A scanned page can carry a small text layer (a header stamp, a page number)
# while the actual content is only in the page image.  Such a page still needs
# OCR: a page whose image covers most of the page and whose text layer is
# shorter than this is treated as image-only.
IMAGE_COVERAGE_MIN = 0.5
IMAGE_TEXT_MIN_CHARS = 300
# A PDF whose streams were decoded as UTF-8 carries millions of U+FFFD
# replacements; a few coincidental byte matches in binary data are normal.
CORRUPTION_SEQUENCE = b"\xef\xbf\xbd"
CORRUPTION_MIN_HITS = 1000


def count_binary_corruption(pdf_path: Path | str, chunk_size: int = 1 << 20) -> int:
    """Count UTF-8 replacement-character sequences in a binary file.

    Downloads that pass binary data through a text decoder destroy PDF image
    streams: the file still opens, but pages render blank and OCR returns
    nothing.  Detecting it here prevents a silent ``empty`` document.
    """
    total = 0
    with Path(pdf_path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            total += block.count(CORRUPTION_SEQUENCE)
    return total


@dataclass
class PageAudit:
    page: int
    chars: int
    has_text: bool
    image_coverage: float = 0.0
    needs_ocr: bool = False


@dataclass
class PdfAudit:
    document_id: str
    document_name: str
    pdf: str
    exists: bool
    pages: int = 0
    pages_with_text: int = 0
    pages_without_text: List[int] = field(default_factory=list)
    pages_needing_ocr: List[int] = field(default_factory=list)
    total_chars: int = 0
    min_chars: int = DEFAULT_MIN_CHARS
    corruption_hits: int = 0
    status: str = "unknown"
    error: Optional[str] = None
    page_details: List[PageAudit] = field(default_factory=list)

    @property
    def text_coverage(self) -> float:
        if not self.pages:
            return 0.0
        return self.pages_with_text / self.pages

    def to_dict(self, include_pages: bool = False) -> Dict[str, Any]:
        data = asdict(self)
        if not include_pages:
            data.pop("page_details", None)
        data["text_coverage"] = round(self.text_coverage, 4)
        return data


def audit_pdf(
    pdf_path: Path | str,
    document: Optional[DocumentMeta] = None,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> PdfAudit:
    """Inspect a PDF's embedded text layer page by page."""
    path = Path(pdf_path)
    audit = PdfAudit(
        document_id=document.document_id if document else path.stem,
        document_name=document.document_name if document else path.stem,
        pdf=str(path),
        exists=path.exists(),
        min_chars=min_chars,
    )
    if not path.exists():
        audit.status = "missing_pdf"
        audit.error = f"PDF not found: {path}"
        return audit

    audit.corruption_hits = count_binary_corruption(path)
    if audit.corruption_hits >= CORRUPTION_MIN_HITS:
        audit.status = "source_corrupt"
        audit.error = (
            f"{audit.corruption_hits} U+FFFD replacement sequences in the PDF: "
            "image/text streams were corrupted by a lossy text decoding; "
            "re-download the original PDF (data/raw must not be edited)"
        )
        try:
            with fitz.open(path) as pdf:
                audit.pages = pdf.page_count
        except Exception:  # pragma: no cover - unreadable file
            pass
        return audit

    try:
        with fitz.open(path) as pdf:
            audit.pages = pdf.page_count
            for page_index, page in enumerate(pdf):
                text = page.get_text("text", sort=True).strip()
                chars = len(text)
                has_text = chars >= min_chars
                page_area = abs(page.rect.width * page.rect.height) or 1.0
                image_area = 0.0
                for info in page.get_image_info():
                    rect = fitz.Rect(info["bbox"])
                    image_area += abs(rect.width * rect.height)
                coverage = min(image_area / page_area, 1.0)
                needs_ocr = (not has_text) or (
                    coverage >= IMAGE_COVERAGE_MIN and chars < IMAGE_TEXT_MIN_CHARS
                )
                audit.page_details.append(
                    PageAudit(
                        page=page_index + 1,
                        chars=chars,
                        has_text=has_text,
                        image_coverage=round(coverage, 3),
                        needs_ocr=needs_ocr,
                    )
                )
                audit.total_chars += chars
                if has_text:
                    audit.pages_with_text += 1
                else:
                    audit.pages_without_text.append(page_index + 1)
                if needs_ocr:
                    audit.pages_needing_ocr.append(page_index + 1)
    except Exception as exc:  # pragma: no cover - corrupt PDF path
        audit.status = "parse_failed"
        audit.error = f"{type(exc).__name__}: {exc}"
        return audit

    if audit.pages and len(audit.pages_needing_ocr) == audit.pages:
        audit.status = "ocr_required"
    elif audit.pages_needing_ocr:
        audit.status = "partial_text"
    else:
        audit.status = "ok"
    return audit


def audit_registry(
    registry: Optional[DocumentRegistry] = None,
    min_chars: int = DEFAULT_MIN_CHARS,
    document_ids: Optional[Iterable[str]] = None,
) -> List[PdfAudit]:
    registry = registry or load_registry()
    audits = []
    for document in registry.select(document_ids=document_ids):
        pdf = document.pdf
        if pdf is None:
            audits.append(
                PdfAudit(
                    document_id=document.document_id,
                    document_name=document.document_name,
                    pdf="",
                    exists=False,
                    min_chars=min_chars,
                    status="missing_pdf",
                    error="registry entry has no pdf path",
                )
            )
            continue
        audits.append(audit_pdf(pdf, document, min_chars=min_chars))
    return audits


def write_audit_report(
    audits: List[PdfAudit], output_path: Path, include_pages: bool = True
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "documents": [audit.to_dict(include_pages=include_pages) for audit in audits],
        "totals": {
            "documents": len(audits),
            "pages": sum(a.pages for a in audits),
            "pages_with_text": sum(a.pages_with_text for a in audits),
            "pages_without_text": sum(len(a.pages_without_text) for a in audits),
            "pages_needing_ocr": sum(len(a.pages_needing_ocr) for a in audits),
        },
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path


def main() -> None:  # pragma: no cover - manual entry point
    import sys

    output = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed/audit.json")
    audits = audit_registry()
    write_audit_report(audits, output)
    for audit in audits:
        print(
            f"{audit.document_id:16s} {audit.status:14s} "
            f"pages={audit.pages:3d} text={audit.pages_with_text:3d} "
            f"ocr={len(audit.pages_needing_ocr):3d}"
        )
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
