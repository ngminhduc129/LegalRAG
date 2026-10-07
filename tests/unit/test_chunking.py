"""Unit tests for chunking module."""

import pytest
from legal_rag.chunking.legal_aware import (
    LegalAwareChunker,
    FixedSizeChunker,
    ParagraphChunker,
    ArticleChunker,
    ClauseChunker,
    make_chunk,
)
from legal_rag.chunking.cleaner import clean_chunk, clean_content, is_page_number
from legal_rag.chunking.validator import validate_chunks, ValidationIssue


class TestLegalAwareChunker:
    def setup_method(self):
        self.chunker = LegalAwareChunker(document_id="test_doc")

    def test_basic_article_chunking(self):
        lines = [
            {"text": "Điều 1. Phạm vi điều chỉnh", "page": 1},
            {"text": "Bộ luật này quy định về quan hệ lao động.", "page": 1},
            {"text": "1. Bộ luật này quy định...", "page": 1},
            {"text": "2. Đối tượng áp dụng...", "page": 1},
        ]
        chunks = self.chunker.chunk(lines)
        assert len(chunks) == 3  # article_intro + 2 clauses
        assert chunks[0]["chunk_type"] == "article_intro"
        assert chunks[1]["chunk_type"] == "clause"
        assert chunks[2]["chunk_type"] == "clause"

    def test_clause_with_points(self):
        lines = [
            {"text": "Điều 1. Title", "page": 1},
            {"text": "1. Clause intro", "page": 1},
            {"text": "a) Point A", "page": 1},
            {"text": "b) Point B", "page": 1},
        ]
        chunks = self.chunker.chunk(lines)
        chunk_types = [c["chunk_type"] for c in chunks]
        assert "clause_intro" in chunk_types
        assert chunk_types.count("point") == 2

    def test_chapter_section_hierarchy(self):
        lines = [
            {"text": "Chương I", "page": 1},
            {"text": "Mục 1", "page": 1},
            {"text": "Điều 1. Title", "page": 1},
            {"text": "1. Clause", "page": 1},
        ]
        chunks = self.chunker.chunk(lines)
        for chunk in chunks:
            assert chunk["chapter"] == "Chương I"
            assert chunk["section"] == "Mục 1"
            assert chunk["article"] == "Điều 1"

    def test_duplicate_article_header(self):
        lines = [
            {"text": "Điều 1. Title", "page": 1},
            {"text": "Content", "page": 1},
            {"text": "Điều 1. Title", "page": 2},  # Duplicate
            {"text": "More content", "page": 2},
        ]
        chunks = self.chunker.chunk(lines)
        articles = {c["article"] for c in chunks if c.get("article")}
        assert len(articles) == 1  # Only one article


class TestFixedSizeChunker:
    def test_fixed_size_chunking(self):
        chunker = FixedSizeChunker(chunk_size=10, overlap=2, document_id="test")
        lines = [{"text": "word", "page": 1} for _ in range(25)]
        chunks = chunker.chunk(lines)
        # 25 words, chunk_size=10, overlap=2 -> step=8
        # chunks at indices: 0, 8, 16, 24 -> 4 chunks
        assert len(chunks) == 4
        for chunk in chunks:
            assert chunk["chunk_type"] == "fixed_size"
            assert chunk["chunk_size"] == 10
            assert chunk["overlap"] == 2


class TestParagraphChunker:
    def test_paragraph_chunking(self):
        chunker = ParagraphChunker(document_id="test")
        text = "Para 1.\n\nPara 2.\n\nPara 3."
        lines = [{"text": line, "page": 1} for line in text.split("\n")]
        chunks = chunker.chunk(lines)
        assert len(chunks) == 3
        for chunk in chunks:
            assert chunk["chunk_type"] == "paragraph"


class TestArticleChunker:
    def test_article_chunking(self):
        chunker = ArticleChunker(document_id="test")
        lines = [
            {"text": "Điều 1. Title 1", "page": 1},
            {"text": "Content 1", "page": 1},
            {"text": "Điều 2. Title 2", "page": 2},
            {"text": "Content 2", "page": 2},
        ]
        chunks = chunker.chunk(lines)
        assert len(chunks) == 2
        assert chunks[0]["article"] == "Điều 1"
        assert chunks[1]["article"] == "Điều 2"


class TestClauseChunker:
    def test_clause_chunking(self):
        chunker = ClauseChunker(document_id="test")
        lines = [
            {"text": "Điều 1. Title", "page": 1},
            {"text": "1. Clause 1", "page": 1},
            {"text": "Content 1", "page": 1},
            {"text": "2. Clause 2", "page": 1},
            {"text": "Content 2", "page": 1},
        ]
        chunks = chunker.chunk(lines)
        assert len(chunks) == 2
        assert chunks[0]["clause"] == "1"
        assert chunks[1]["clause"] == "2"


class TestCleaner:
    def test_is_page_number(self):
        assert is_page_number("1")
        assert is_page_number("10")
        assert is_page_number("123")
        assert not is_page_number("Khoản 1")
        assert not is_page_number("Điều 123")
        assert not is_page_number("30 ngày")

    def test_clean_content_removes_page_numbers(self):
        content = "Line 1\n5\nLine 2"
        cleaned, actions, suspicious = clean_content(content)
        assert "5" not in cleaned
        assert any("REMOVE_PAGE_NUMBER" in a for a in actions)

    def test_clean_content_superscript(self):
        content = "Text³"
        cleaned, actions, _ = clean_content(content)
        assert "³" not in cleaned
        assert any("REMOVE_SUPERSCRIPT_FOOTNOTE" in a for a in actions)

    def test_clean_chunk(self):
        chunk = {
            "chunk_id": "test_1",
            "content": "Line 1\n5\nLine 2",
            "article_title": "Title³",
        }
        cleaned = clean_chunk(chunk)
        assert cleaned["cleaned"]
        assert "5" not in cleaned["content"]
        assert "³" not in cleaned["article_title"]


class TestValidator:
    def test_validate_basic_chunks(self):
        chunks = [
            {
                "chunk_id": "test_1",
                "document_id": "test",
                "document_name": "Test",
                "document_number": "1",
                "article": "Điều 1",
                "chunk_type": "clause",
                "content": "Content.",
                "page_start": 1,
                "page_end": 1,
            }
        ]
        report = validate_chunks(chunks)
        assert report.errors == 0

    def test_validate_missing_metadata(self):
        chunks = [
            {
                "chunk_id": "test_1",
                "content": "Content.",
                "page_start": 1,
                "page_end": 1,
                # Missing required fields
            }
        ]
        report = validate_chunks(chunks)
        assert report.errors > 0
        missing_fields = [i for i in report.issues if i.code == "MISSING_METADATA"]
        assert len(missing_fields) > 0

    def test_validate_invalid_page_range(self):
        chunks = [
            {
                "chunk_id": "test_1",
                "document_id": "test",
                "document_name": "Test",
                "document_number": "1",
                "article": "Điều 1",
                "chunk_type": "clause",
                "content": "Content.",
                "page_start": 5,
                "page_end": 3,  # Invalid: start > end
            }
        ]
        report = validate_chunks(chunks)
        assert report.errors > 0
        invalid_range = [i for i in report.issues if i.code == "INVALID_PAGE_RANGE"]
        assert len(invalid_range) > 0

    def test_duplicate_structure_key(self):
        chunks = [
            {
                "chunk_id": "test_1",
                "document_id": "test",
                "document_name": "Test",
                "document_number": "1",
                "article": "Điều 1",
                "clause": "1",
                "point": None,
                "chunk_type": "clause",
                "content": "Content.",
                "page_start": 1,
                "page_end": 1,
            },
            {
                "chunk_id": "test_2",
                "document_id": "test",
                "document_name": "Test",
                "document_number": "1",
                "article": "Điều 1",
                "clause": "1",
                "point": None,
                "chunk_type": "clause",
                "content": "Content duplicate.",
                "page_start": 1,
                "page_end": 1,
            }
        ]
        report = validate_chunks(chunks)
        duplicates = [i for i in report.issues if i.code == "DUPLICATE_STRUCTURE_KEY"]
        assert len(duplicates) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])