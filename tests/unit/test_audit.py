"""Unit tests for the read-only corpus audit."""

import fitz

from legal_rag.corpus.audit import (
    CORRUPTION_MIN_HITS,
    IMAGE_TEXT_MIN_CHARS,
    audit_pdf,
    count_binary_corruption,
)


def _blank_pdf(path, pages=2):
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page()
    doc.save(path)
    doc.close()
    return path


def _pdf_with_page_image_and_header(path):
    """One page: full-page image plus a short header text layer.

    This is the shape of a scanned page exported with a "Cơ quan phát hành"
    stamp: the audit must still treat it as image-only and OCR it.
    """
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100))
    pix.set_rect(pix.irect, (200, 200, 200))
    page.insert_image(page.rect, pixmap=pix)
    page.insert_text((20, 20), "Co quan phat hanh: Van phong Chinh phu")
    doc.save(path)
    doc.close()
    return path


def test_count_binary_corruption(tmp_path):
    clean = tmp_path / "clean.bin"
    clean.write_bytes(b"%PDF-1.4 normal binary \x00\x01\x02")
    assert count_binary_corruption(clean) == 0

    broken = tmp_path / "broken.bin"
    broken.write_bytes(b"%PDF" + b"\xef\xbf\xbd" * 5)
    assert count_binary_corruption(broken) == 5


def test_audit_flags_corrupt_pdf(tmp_path):
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"%PDF-1.4\n" + b"\xef\xbf\xbd" * CORRUPTION_MIN_HITS)
    audit = audit_pdf(path)
    assert audit.status == "source_corrupt"
    assert audit.corruption_hits >= CORRUPTION_MIN_HITS
    assert "re-download" in audit.error


def test_audit_reports_ocr_required_for_image_only_pdf(tmp_path):
    path = _blank_pdf(tmp_path / "blank.pdf", pages=2)
    audit = audit_pdf(path)
    assert audit.status == "ocr_required"
    assert audit.pages == 2
    assert audit.pages_without_text == [1, 2]
    assert audit.pages_needing_ocr == [1, 2]
    assert audit.corruption_hits == 0


def test_audit_ocrs_page_with_header_only_text_layer(tmp_path):
    path = _pdf_with_page_image_and_header(tmp_path / "scan.pdf")
    audit = audit_pdf(path)
    assert audit.pages_with_text == 1  # the header counts as text
    assert audit.pages_needing_ocr == [1]  # but the page image still needs OCR
    assert audit.status == "ocr_required"
    detail = audit.page_details[0]
    assert detail.image_coverage >= 0.5
    assert detail.chars < IMAGE_TEXT_MIN_CHARS


def test_audit_keeps_full_text_layer_page(tmp_path):
    path = tmp_path / "digital.pdf"
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100))
    pix.set_rect(pix.irect, (200, 200, 200))
    page.insert_image(page.rect, pixmap=pix)
    page.insert_textbox(page.rect, "Noi dung van ban rat dai. " * 40)
    doc.save(path)
    doc.close()

    audit = audit_pdf(path)
    assert audit.pages_needing_ocr == []
    assert audit.status == "ok"


def test_audit_missing_pdf(tmp_path):
    audit = audit_pdf(tmp_path / "nope.pdf")
    assert audit.status == "missing_pdf"
    assert audit.pages == 0
