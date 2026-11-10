"""Unit tests for parsing module."""

import pytest
from legal_rag.parsing.classifier import (
    LineClassifier,
    clean_line,
    is_page_number,
    extract_pdf_lines,
    CHAPTER_RE,
    SECTION_RE,
    ARTICLE_RE,
    CLAUSE_RE,
    POINT_RE,
)
from legal_rag.parsing.parser import LegalStructureParser, parse_legal_structure


def _parse(lines):
    """Parser now returns ParseResult; tests mostly need the chunk list."""
    return parse_legal_structure(lines).chunks


class TestClassifier:
    def test_chapter_regex(self):
        assert CHAPTER_RE.match("Chương I")
        assert CHAPTER_RE.match("CHƯƠNG 1")
        assert CHAPTER_RE.match("  Chương  II  ")
        assert not CHAPTER_RE.match("Mục 1")

    def test_section_regex(self):
        assert SECTION_RE.match("Mục 1")
        assert SECTION_RE.match("MỤC 2")
        assert not SECTION_RE.match("Chương 1")

    def test_article_regex(self):
        assert ARTICLE_RE.match("Điều 1. Title")
        assert ARTICLE_RE.match("ĐIỀU 15.  Title here  ")
        assert not ARTICLE_RE.match("Khoản 1.")

    def test_clause_regex(self):
        assert CLAUSE_RE.match("1. Clause content")
        assert CLAUSE_RE.match("  15.  Another clause  ")
        assert not CLAUSE_RE.match("a) Point content")

    def test_point_regex(self):
        assert POINT_RE.match("a) Point content")
        assert POINT_RE.match("  b)  Another point  ")
        assert not POINT_RE.match("1. Clause content")

    def test_clean_line(self):
        assert clean_line("  hello   world  ") == "hello world"
        assert clean_line("hello\u00a0world") == "hello world"
        assert clean_line("hello\u00adworld") == "helloworld"

    def test_is_page_number(self):
        assert is_page_number("1")
        assert is_page_number("10")
        assert is_page_number("123")
        assert not is_page_number("Khoản 1")
        assert not is_page_number("Điều 123")
        assert not is_page_number("30 ngày")

    def test_classify_line_chapter(self):
        classifier = LineClassifier()
        line = classifier.classify_line("Chương I", 1, 0)
        assert line.is_structural_candidate
        assert line.candidate_type == "chapter"

    def test_classify_line_article(self):
        classifier = LineClassifier()
        line = classifier.classify_line("Điều 1. Title", 1, 0)
        assert line.is_structural_candidate
        assert line.candidate_type == "article"

    def test_classify_line_clause(self):
        classifier = LineClassifier()
        line = classifier.classify_line("1. Clause content", 1, 0)
        assert line.is_structural_candidate
        assert line.candidate_type == "clause"


class TestParser:
    def test_parse_simple_article(self):
        lines = [
            {"text": "Chương I", "page": 1},
            {"text": "Mục 1", "page": 1},
            {"text": "Điều 1. Phạm vi điều chỉnh", "page": 1},
            {"text": "Bộ luật này quy định về quan hệ lao động.", "page": 1},
            {"text": "1. Bộ luật này quy định...", "page": 1},
            {"text": "2. Đối tượng áp dụng...", "page": 1},
        ]
        chunks = _parse(lines)
        assert len(chunks) >= 3  # article_intro + 2 clauses
        article_chunks = [c for c in chunks if c["article"] == "Điều 1"]
        assert len(article_chunks) >= 3

    def test_parse_with_points(self):
        lines = [
            {"text": "Điều 1. Title", "page": 1},
            {"text": "1. Clause intro", "page": 1},
            {"text": "a) Point 1", "page": 1},
            {"text": "b) Point 2", "page": 1},
        ]
        chunks = _parse(lines)
        point_chunks = [c for c in chunks if c["chunk_type"] == "point"]
        assert len(point_chunks) == 2
        assert point_chunks[0]["point"] == "a"
        assert point_chunks[1]["point"] == "b"

    def test_parse_duplicate_article_header(self):
        lines = [
            {"text": "Chương I", "page": 1},
            {"text": "Mục 1", "page": 1},
            {"text": "Điều 1. Title", "page": 1},
            {"text": "Bộ luật này quy định về quan hệ lao động.", "page": 1},
            {"text": "Điều 1. Title", "page": 2},  # Duplicate
            {"text": "Nội dung bổ sung về phạm vi điều chỉnh.", "page": 2},
        ]
        chunks = _parse(lines)
        article_chunks = [c for c in chunks if c["article"] == "Điều 1"]
        # Should not create duplicate article structure
        assert len(article_chunks) >= 1


class TestEdgeCases:
    def test_legal_reference_detection(self):
        lines = [
            {"text": "Điều 1. Title", "page": 1},
            {"text": "1. Theo quy định tại Điều 5", "page": 1},
        ]
        chunks = _parse(lines)
        # The reference should not be parsed as a new article
        articles = {c["article"] for c in chunks if c.get("article")}
        assert "Điều 5" not in articles

    def test_ref_like_title_keeps_real_titles(self):
        from legal_rag.parsing.parser import is_ref_like_title

        assert not is_ref_like_title("Quy định chuyển tiếp")
        assert not is_ref_like_title("Quy định chung")
        assert not is_ref_like_title("Quy định về xử phạt vi phạm hành chính")
        assert is_ref_like_title("Quy định tại Điều 5")
        assert is_ref_like_title("Quy định của Bộ luật này")
        assert is_ref_like_title("của Bộ luật này;")

    def test_article_title_starting_with_quy_dinh(self):
        lines = [
            {"text": "Điều 49. Sửa đổi, bổ sung", "page": 1},
            {"text": "1. Nội dung sửa đổi.", "page": 1},
            {"text": "Điều 50. Quy định chuyển tiếp", "page": 2},
            {"text": "1. Nội dung chuyển tiếp.", "page": 2},
        ]
        result = parse_legal_structure(lines)
        assert 49 in result.articles_seen
        assert 50 in result.articles_seen

    def test_strip_ocr_noise_before_heading(self):
        from legal_rag.parsing.classifier import clean_line

        assert clean_line("ặ Điều 33. Vi phạm") == "Điều 33. Vi phạm"
        assert clean_line("| Điều 11. Trình tự") == "Điều 11. Trình tự"
        assert clean_line("-. Điều 5. Nội dung") == "Điều 5. Nội dung"
        # Cross references and one-letter words must not be rewritten.
        assert clean_line("theo Điều 154 Bộ luật Lao động") == (
            "theo Điều 154 Bộ luật Lao động"
        )
        assert clean_line("ở Điều 5. Trường hợp") == "ở Điều 5. Trường hợp"

    def test_noisy_ocr_heading_is_parsed(self):
        from legal_rag.parsing.classifier import clean_line

        raw = [
            ("Điều 32. Vi phạm về người nước ngoài", 1),
            ("1. Nội dung khoản một.", 1),
            ("ặ Điều 33. Vi phạm của tổ chức", 2),
            ("1. Nội dung khoản một.", 2),
        ]
        lines = [{"text": clean_line(text), "page": page} for text, page in raw]
        result = parse_legal_structure(lines)
        assert 32 in result.articles_seen
        assert 33 in result.articles_seen

    def test_quoted_amendment_heading_is_not_a_new_article(self):
        lines = [
            {"text": "Điều 1. Sửa đổi, bổ sung", "page": 1},
            {"text": "1. Sửa đổi Điều 4 như sau:", "page": 1},
            {"text": "“Điều 4. Sử dụng người lao động nước ngoài", "page": 1},
            {"text": "1. Nội dung của khoản một.", "page": 1},
            {"text": "Điều 2. Bãi bỏ một số quy định", "page": 2},
            {"text": "1. Nội dung của khoản một.", "page": 2},
        ]
        result = parse_legal_structure(lines)
        assert result.articles_seen == [1, 2]

    def test_empty_lines_skipped(self):
        lines = [
            {"text": "Điều 1. Title", "page": 1},
            {"text": "", "page": 1},  # Empty line
            {"text": "  ", "page": 1},  # Whitespace only
            {"text": "1. Clause", "page": 1},
        ]
        chunks = _parse(lines)
        assert len(chunks) >= 1


class TestSignatureAndAppendix:
    def test_quoted_noi_nhan_does_not_truncate(self):
        lines = [
            {"text": "Điều 1. Title", "page": 1},
            {"text": "Nội dung có cụm “Nơi nhận:” trong ngoặc kép.", "page": 1},
            {"text": "1. Clause one", "page": 1},
            {"text": "Nơi nhận:", "page": 2},
            {"text": "- Văn phòng Quốc hội;", "page": 2},
            {"text": "TM. BỘ TRƯỞNG", "page": 2},
            {"text": "Nguyễn Văn A", "page": 2},
        ]
        result = parse_legal_structure(lines)
        assert result.signature_lines > 0
        assert any(c["chunk_type"] == "clause" for c in result.chunks)
        assert "Nội dung có cụm" in "\n".join(c["content"] for c in result.chunks)

    def test_distribution_list_line_is_not_signature_start(self):
        from legal_rag.parsing.classifier import is_signature_start

        assert not is_signature_start("- Văn phòng Quốc hội;")
        assert not is_signature_start("“Nơi nhận:”")
        assert is_signature_start("Nơi nhận:")

    def test_appendix_is_separated(self):
        lines = [
            {"text": "Điều 1. Title", "page": 1},
            {"text": "1. Clause", "page": 1},
            {"text": "Nơi nhận:", "page": 1},
            {"text": "TM. BỘ TRƯỞNG", "page": 1},
            {"text": "- Lưu: VT, PC.", "page": 1},
            {"text": "PHỤ LỤC I", "page": 2},
            {"text": "Mẫu số 01", "page": 2},
            {"text": "Nội dung biểu mẫu.", "page": 2},
        ]
        result = parse_legal_structure(lines)
        assert result.appendix_chunks >= 1
        appendix = [c for c in result.chunks if c["chunk_type"] == "appendix"]
        assert appendix
        assert appendix[0]["article"] is None
        assert "Nội dung biểu mẫu." in appendix[0]["content"]

    def test_distribution_list_detects_signature_before_long_appendix(self):
        """A scanned decree: no ``./.`` before Nơi nhận and a long appendix."""
        lines = [
            {"text": "Điều 5. Hiệu lực và trách nhiệm thi hành", "page": 1},
            {"text": "1. Nghị định này có hiệu lực từ ngày ký.", "page": 1},
            {"text": "Người sử dụng lao động chịu trách nhiệm thi hành.", "page": 1},
            {"text": "Nơi nhận: TM. CHÍNH PHỦ", "page": 1},
            {"text": "- Ban Bí thư Trung ương Đảng;", "page": 1},
            {"text": "- Thủ tướng, các Phó Thủ tướng Chính phủ;", "page": 1},
            {"text": "- Các bộ, cơ quan ngang bộ;", "page": 1},
            {"text": "- Lưu: VT, KTTH.", "page": 1},
            {"text": "DANH MỤC ĐỊA BÀN ÁP DỤNG", "page": 2},
            {"text": "(Kèm theo Nghị định số 74/2024/NĐ-CP)", "page": 2},
            {"text": "1. Vùng I, gồm các địa bàn:", "page": 2},
            {"text": "- Các quận thuộc thành phố Hà Nội;", "page": 2},
        ]
        result = parse_legal_structure(lines)
        assert result.appendix_chunks >= 1
        appendix = [c for c in result.chunks if c["chunk_type"] == "appendix"]
        assert any("Vùng I" in c["content"] for c in appendix)
        # The appendix table must not become a clause of the last article.
        assert not any(
            c["chunk_type"] == "clause" and "Vùng I" in c["content"]
            for c in result.chunks
        )

    def test_signature_end_splits_appendix_without_marker(self):
        lines = [
            {"text": "Điều 1. Title", "page": 1},
            {"text": "1. Clause", "page": 1},
            {"text": "Nội dung kết thúc./.", "page": 1},
            {"text": "Nơi nhận: BỘ TRƯỞNG", "page": 1},
            {"text": "- Bộ trưởng (để báo cáo);", "page": 1},
            {"text": "(để biết); Nguyễn Văn A - Lưu: VT, PC.", "page": 1},
            {"text": "BẢNG DỮ LIỆU", "page": 2},
            {"text": "Nội dung bảng.", "page": 2},
        ]
        result = parse_legal_structure(lines)
        assert result.appendix_chunks >= 1
        assert any(c["chunk_type"] == "appendix" for c in result.chunks)

    def test_document_metadata_propagated(self):
        from legal_rag.corpus.registry import DocumentMeta

        document = DocumentMeta(
            document_id="nd_test",
            document_name="Nghị định test",
            document_number="99/2020/NĐ-CP",
            doc_type="Nghị định",
        )
        lines = [
            {"text": "Điều 1. Title", "page": 1},
            {"text": "1. Clause", "page": 1},
        ]
        result = parse_legal_structure(lines, document)
        for chunk in result.chunks:
            assert chunk["document_id"] == "nd_test"
            assert chunk["document_name"] == "Nghị định test"
            assert chunk["document_number"] == "99/2020/NĐ-CP"

    def test_nfc_normalization(self):
        from legal_rag.parsing.classifier import clean_line

        decomposed = "Điều 1. Pha\u0323m vi"
        assert clean_line(decomposed) == "Điều 1. Phạm vi"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
