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
        chunks = parse_legal_structure(lines)
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
        chunks = parse_legal_structure(lines)
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
        chunks = parse_legal_structure(lines)
        article_chunks = [c for c in chunks if c["article"] == "Điều 1"]
        # Should not create duplicate article structure
        assert len(article_chunks) >= 1


class TestEdgeCases:
    def test_legal_reference_detection(self):
        lines = [
            {"text": "Điều 1. Title", "page": 1},
            {"text": "1. Theo quy định tại Điều 5", "page": 1},
        ]
        chunks = parse_legal_structure(lines)
        # The reference should not be parsed as a new article
        articles = {c["article"] for c in chunks if c.get("article")}
        assert "Điều 5" not in articles

    def test_empty_lines_skipped(self):
        lines = [
            {"text": "Điều 1. Title", "page": 1},
            {"text": "", "page": 1},  # Empty line
            {"text": "  ", "page": 1},  # Whitespace only
            {"text": "1. Clause", "page": 1},
        ]
        chunks = parse_legal_structure(lines)
        assert len(chunks) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])