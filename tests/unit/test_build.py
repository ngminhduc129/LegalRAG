"""Unit tests for the batch corpus build (no OCR engine required)."""

from pathlib import Path

import fitz
import pytest

import legal_rag.corpus.build as build_module
from legal_rag.corpus.audit import PdfAudit
from legal_rag.corpus.build import build_document, run_corpus_build
from legal_rag.corpus.extractor import extract_pdf_lines
from legal_rag.corpus.registry import DocumentMeta, DocumentRegistry


def _document(tmp_path, document_id="doc"):
    pdf = tmp_path / f"{document_id}.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    return DocumentMeta(
        document_id=document_id,
        document_name="Test document",
        document_number="1/2020/ND-CP",
        pdf=pdf,
        doc_type="Nghi dinh",
    )


def _audit(document, pages=1, with_text=1, without=None):
    return PdfAudit(
        document_id=document.document_id,
        document_name=document.document_name,
        pdf=str(document.pdf),
        exists=True,
        pages=pages,
        pages_with_text=with_text,
        pages_without_text=without or [],
        pages_needing_ocr=without or [],
    )


def _lines():
    return [
        {"text": "Điều 1. Phạm vi điều chỉnh", "page": 1},
        {"text": "1. Nội dung khoản một.", "page": 1},
    ]


def test_build_document_ok(tmp_path, monkeypatch):
    document = _document(tmp_path)
    monkeypatch.setattr(build_module, "audit_pdf", lambda *a, **k: _audit(document))
    monkeypatch.setattr(build_module, "extract_pdf_lines", lambda *a, **k: _lines())

    result, chunks = build_document(document, tmp_path / "by_document")

    assert result.status == "ok"
    assert result.articles_seen == 1
    assert chunks
    assert all(chunk["document_id"] == "doc" for chunk in chunks)
    assert all(chunk["retrieval_eligible"] is True for chunk in chunks)
    assert (tmp_path / "by_document" / "doc.jsonl").exists()


def test_build_document_ocr_required(tmp_path, monkeypatch):
    document = _document(tmp_path)
    monkeypatch.setattr(
        build_module,
        "audit_pdf",
        lambda *a, **k: _audit(document, pages=2, with_text=0, without=[1, 2]),
    )
    monkeypatch.setattr(build_module, "extract_pdf_lines", lambda *a, **k: [])

    result, chunks = build_document(document, tmp_path / "by_document")

    assert result.status == "ocr_required"
    assert result.pages_ocr_required == 2
    assert chunks == []


def test_build_document_missing_pdf(tmp_path):
    document = DocumentMeta(
        document_id="missing",
        document_name="Missing",
        document_number="0",
        pdf=tmp_path / "nope.pdf",
    )
    result, chunks = build_document(document, tmp_path / "by_document")
    assert result.status == "missing_pdf"
    assert chunks == []


def test_build_document_source_corrupt_skips_ocr(tmp_path, monkeypatch):
    document = _document(tmp_path)
    corrupt = _audit(document, pages=122, with_text=0, without=list(range(1, 123)))
    corrupt.status = "source_corrupt"
    corrupt.corruption_hits = 5_000_000
    corrupt.error = "corrupted streams"
    monkeypatch.setattr(build_module, "audit_pdf", lambda *a, **k: corrupt)

    def _fail(*args, **kwargs):
        raise AssertionError("OCR must not run for a corrupt PDF")

    monkeypatch.setattr(build_module, "_resolve_ocr_pages", _fail)

    result, chunks = build_document(document, tmp_path / "by_document")
    assert result.status == "source_corrupt"
    assert result.error == "corrupted streams"
    assert result.pages == 122
    assert chunks == []


def test_run_corpus_build_refuses_existing_run(tmp_path):
    document = _document(tmp_path)
    registry = DocumentRegistry([document])
    (tmp_path / "run01").mkdir()

    with pytest.raises(FileExistsError):
        run_corpus_build(
            "run01",
            registry=registry,
            chunks_dir=tmp_path,
            ocr_engine="tesseract",
        )


def _pdf_with_text(path, text="Van ban co text layer."):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((30, 40), text)
    doc.save(path)
    doc.close()
    return path


@pytest.mark.slow
def test_extract_prefers_ocr_for_pages_selected_for_ocr(tmp_path):
    pdf = _pdf_with_text(tmp_path / "scan.pdf", "Co quan phat hanh: Van phong")
    lines = extract_pdf_lines(
        pdf, ocr_pages={1: ["Điều 1. Phạm vi điều chỉnh", "1. Nội dung."]}
    )
    assert lines[0]["text"] == "Điều 1. Phạm vi điều chỉnh"
    assert lines[0]["text_source"] == "ocr"


@pytest.mark.slow
def test_extract_keeps_text_layer_when_page_not_ocr(tmp_path):
    pdf = _pdf_with_text(tmp_path / "digital.pdf")
    lines = extract_pdf_lines(pdf, ocr_pages={})
    assert lines
    assert lines[0]["text_source"] == "text_layer"


def _pdf_with_empty_page(path):
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((30, 40), "Điều 1. Phạm vi điều chỉnh.")
    page2 = doc.new_page()  # no text layer: scanned / blank page
    doc.save(path)
    doc.close()
    return path


@pytest.mark.slow
def test_extract_warns_and_records_empty_page(tmp_path, capsys):
    pdf = _pdf_with_empty_page(tmp_path / "mixed.pdf")
    empty: list[int] = []
    lines = extract_pdf_lines(
        pdf,
        on_empty_page=lambda p, n: empty.append(p),
        pdf_name="mixed.pdf",
    )
    # Only page 1 (with text) yields lines.
    assert all(line["page"] == 1 for line in lines)
    assert empty == [2]
    captured = capsys.readouterr()
    assert "[WARN] page 2 of mixed.pdf has no text layer" in captured.out
    assert "mixed.pdf" in captured.out


def _pdf_with_garbage_page(path):
    """PDF with page 1 clean, page 2 full of strange/garbage characters."""
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((30, 40), "Điều 1. Phạm vi điều chỉnh.")
    page2 = doc.new_page()
    page2.insert_text((30, 40), "lao d{>ng ma Ban tr9ng tai {}<>~£|^ khong duqc")
    doc.save(path)
    doc.close()
    return path


@pytest.mark.slow
def test_build_records_page_quality_warnings(tmp_path, monkeypatch):
    """build_document should record page quality warnings alongside empty_pages."""
    pdf = _pdf_with_garbage_page(tmp_path / "garbage.pdf")
    document = DocumentMeta(
        document_id="doc",
        document_name="Test document",
        document_number="1/2020/ND-CP",
        pdf=pdf,
        doc_type="Nghi dinh",
    )

    monkeypatch.setattr(build_module, "audit_pdf", lambda *a, **k: _audit(document, pages=2, with_text=2))
    monkeypatch.setattr(build_module, "extract_pdf_lines", lambda *a, **k: [
        {"text": "Điều 1. Phạm vi điều chỉnh.", "page": 1},
        {"text": "1. Nội dung.", "page": 1},
    ])

    result, chunks = build_document(document, tmp_path / "by_document")

    # Page 2 should be flagged for quality issues
    quality_warnings = [w for w in result.warnings if "PAGE_QUALITY" in w]
    assert len(quality_warnings) >= 1, f"Expected PAGE_QUALITY warnings, got: {result.warnings}"
    assert any("page 2" in w for w in quality_warnings)

    # Parse report should include page_quality_warnings
    report_path = tmp_path / "by_document" / ".." / "parse_reports" / "doc.json"
    if report_path.exists():
        import json
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert "page_quality_warnings" in report
        assert len(report["page_quality_warnings"]) >= 1


def test_build_fails_on_chunk_id_invariant_violation(tmp_path, monkeypatch):
    """build_document must fail when chunk_id↔metadata invariant is violated."""
    pdf = _pdf_with_garbage_page(tmp_path / "garbage.pdf")
    document = DocumentMeta(
        document_id="doc",
        document_name="Test document",
        document_number="1/2020/ND-CP",
        pdf=pdf,
        doc_type="Nghi dinh",
    )

    monkeypatch.setattr(build_module, "audit_pdf", lambda *a, **k: _audit(document, pages=1, with_text=1))
    monkeypatch.setattr(build_module, "extract_pdf_lines", lambda *a, **k: [
        {"text": "Điều 1. Phạm vi điều chỉnh.", "page": 1},
    ])

    # Mock parse_legal_structure to return corpus chunks with structure IDs
    from legal_rag.parsing.parser import ParseResult
    corpus_chunks = [
        {
            "document_name": "Test document",
            "document_number": "1/2020/ND-CP",
            "chapter": None,
            "chapter_title": None,
            "section": None,
            "section_title": None,
            "article": "Điều 1",
            "article_title": None,
            "clause": None,
            "point": None,
            "chunk_type": "article_intro",
            "content": "Phạm vi điều chỉnh.",
            "page_start": 1,
            "page_end": 1,
            "structure_instance_id": "si1",
            "parent_instance_id": None,
            "structure_role": "main",
            "operation_id": None,
        }
    ]
    monkeypatch.setattr(build_module, "parse_legal_structure", lambda lines, doc: ParseResult(chunks=corpus_chunks))

    from legal_rag.chunking.legal_aware import LegalAwareChunker
    original_chunk = LegalAwareChunker.chunk

    def bad_chunk(self, corpus_chunks):
        result = original_chunk(self, corpus_chunks)
        # Corrupt the first chunk's ID so it doesn't match its metadata
        if result:
            result[0]["chunk_id"] = "doc_điều_99_none_none_article_intro"
        return result

    monkeypatch.setattr(LegalAwareChunker, "chunk", bad_chunk)

    result, chunks = build_document(document, tmp_path / "by_document")

    assert result.status == "parse_failed"
    assert "chunk_id↔metadata" in result.error or "invariant" in result.error.lower()


def test_validate_chunk_id_invariant_skips_appendix():
    """validate_chunk_id_invariant must skip appendix chunks.

    Appendix chunks use their own ID scheme (document_id_appendix_<title>)
    that does not encode article/clause/point, so they must not be flagged
    as violations.
    """
    from legal_rag.corpus.build import validate_chunk_id_invariant

    chunks = [
        {
            "chunk_id": "doc_appendix_phụ_lục_01",
            "article": None,
            "clause": None,
            "point": None,
            "chunk_type": "appendix",
            "content": "Phụ lục 01",
        },
        {
            "chunk_id": "doc_appendix_phụ_lục_02",
            "article": None,
            "clause": None,
            "point": None,
            "chunk_type": "appendix",
            "content": "Phụ lục 02",
        },
        {
            "chunk_id": "doc_điều_1_none_none_article_intro",
            "article": "Điều 1",
            "clause": None,
            "point": None,
            "chunk_type": "article_intro",
            "content": "Nội dung Điều 1.",
        },
    ]

    violations = validate_chunk_id_invariant(chunks, "legal_aware", "doc")

    # Only the non-appendix chunk (article_intro) should be checked.
    # Its chunk_id matches metadata, so zero violations expected.
    assert len(violations) == 0, f"Expected 0 violations, got {len(violations)}: {violations}"
