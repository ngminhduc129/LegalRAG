"""Unit tests for OCR caching (no real OCR engine required)."""

from pathlib import Path

import fitz
import pytest

from legal_rag.corpus.ocr import (
    OcrUnavailable,
    get_engine,
    load_cached_pages,
    ocr_document,
    ocr_page,
    render_page,
)


class FakeEngine:
    name = "fake"
    lang = "vi"

    def __init__(self):
        self.calls = 0

    def ocr_image(self, png_bytes):
        self.calls += 1
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
        return ["Dòng một", "Dòng hai"]


@pytest.fixture()
def pdf_path(tmp_path):
    path = tmp_path / "sample.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.new_page()
    doc.save(path)
    doc.close()
    return path


def test_render_page(pdf_path):
    png = render_page(pdf_path, 1, dpi=72)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_ocr_page_uses_cache(pdf_path, tmp_path):
    engine = FakeEngine()
    cache_root = tmp_path / "ocr"
    first = ocr_page(
        pdf_path, 1, engine, cache_root=cache_root, document_id="doc", dpi=72
    )
    assert first["lines"] == ["Dòng một", "Dòng hai"]
    assert engine.calls == 1

    second = ocr_page(
        pdf_path, 1, engine, cache_root=cache_root, document_id="doc", dpi=72
    )
    assert second["lines"] == first["lines"]
    assert engine.calls == 1

    cached = load_cached_pages(cache_root, "doc", engine, dpi=72)
    assert cached[1] == ["Dòng một", "Dòng hai"]


def test_ocr_page_cache_key_includes_dpi(pdf_path, tmp_path):
    engine = FakeEngine()
    cache_root = tmp_path / "ocr"
    ocr_page(pdf_path, 1, engine, cache_root=cache_root, document_id="doc", dpi=72)
    ocr_page(pdf_path, 1, engine, cache_root=cache_root, document_id="doc", dpi=96)
    assert engine.calls == 2


def test_ocr_document_selected_pages(pdf_path, tmp_path):
    engine = FakeEngine()
    pages = ocr_document(
        pdf_path,
        engine,
        document_id="doc",
        pages=[2],
        cache_root=tmp_path / "ocr",
        dpi=72,
    )
    assert list(pages) == [2]


def test_get_engine_unknown():
    with pytest.raises(ValueError):
        get_engine("nope")


def test_get_engine_missing_dependency(monkeypatch):
    import legal_rag.corpus.ocr as ocr_module

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "pytesseract":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(OcrUnavailable):
        get_engine("tesseract")
