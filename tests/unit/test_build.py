"""Unit tests for the batch corpus build (no OCR engine required)."""

from pathlib import Path

import pytest

import legal_rag.corpus.build as build_module
from legal_rag.corpus.audit import PdfAudit
from legal_rag.corpus.build import build_document, run_corpus_build
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
