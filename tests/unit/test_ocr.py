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

    def __init__(self, psm=None):
        self.calls = 0
        if psm is not None:
            self.psm = psm

    def ocr_image(self, png_bytes):
        self.calls += 1
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
        return ["Dòng một", "Dòng hai"]


class FakePsmEngine(FakeEngine):
    """Engine that reports a psm setting in its cache params."""

    def __init__(self, psm=6):
        super().__init__()
        self.psm = psm

    def cache_params(self):
        return {"psm": self.psm}


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


def test_ocr_page_cache_key_includes_psm(pdf_path, tmp_path):
    import json

    cache_root = tmp_path / "ocr"
    engine = FakePsmEngine(psm=6)
    ocr_page(pdf_path, 1, engine, cache_root=cache_root, document_id="doc", dpi=72)
    ocr_page(pdf_path, 1, engine, cache_root=cache_root, document_id="doc", dpi=72)
    assert engine.calls == 1  # same psm -> cache hit

    engine.psm = 11
    ocr_page(pdf_path, 1, engine, cache_root=cache_root, document_id="doc", dpi=72)
    assert engine.calls == 2  # different psm -> re-OCR
    payload = json.loads(
        (cache_root / "doc" / "page-001.json").read_text(encoding="utf-8")
    )
    assert payload["psm"] == 11


def test_ocr_page_cache_key_includes_pdf_fingerprint(pdf_path, tmp_path):
    """Replacing the PDF must not reuse OCR from the old file."""
    engine = FakeEngine()
    cache_root = tmp_path / "ocr"
    ocr_page(
        pdf_path, 1, engine, cache_root=cache_root, document_id="doc",
        dpi=72, pdf_sha256="aaa",
    )
    ocr_page(
        pdf_path, 1, engine, cache_root=cache_root, document_id="doc",
        dpi=72, pdf_sha256="bbb",
    )
    assert engine.calls == 2


def test_legacy_cache_without_psm_is_accepted(pdf_path, tmp_path):
    """Cache entries written before psm was recorded stay reusable."""
    import json

    cache_root = tmp_path / "ocr"
    ocr_page(pdf_path, 1, FakeEngine(), cache_root=cache_root, document_id="doc", dpi=72)
    cache_file = cache_root / "doc" / "page-001.json"
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    payload.pop("psm", None)
    cache_file.write_text(json.dumps(payload), encoding="utf-8")

    engine = FakePsmEngine(psm=6)
    cached = load_cached_pages(cache_root, "doc", engine, dpi=72)
    assert cached[1] == ["Dòng một", "Dòng hai"]
    assert engine.calls == 0


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
