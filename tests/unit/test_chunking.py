"""Unit tests for chunking module."""

import pytest
from legal_rag.chunking.legal_aware import (
    LegalAwareChunker,
    FixedSizeChunker,
    ParagraphChunker,
    ArticleChunker,
    ClauseChunker,
    make_chunk,
    _safe_id_part,
)
from legal_rag.chunking.cleaner import clean_chunk, clean_content, is_page_number
from legal_rag.chunking.validator import validate_chunks, ValidationIssue


def _parser_chunk(
    article=None,
    article_title=None,
    clause=None,
    point=None,
    chunk_type="clause",
    content="Content",
    page_start=1,
    page_end=1,
    chapter="Chương I",
    chapter_title=None,
    section="Mục 1",
    section_title=None,
):
    """Helper to create parser-output-style chunk."""
    return {
        "document_name": "Bộ luật Lao động",
        "document_number": "125/VBHN-VPQH",
        "chapter": chapter,
        "chapter_title": chapter_title,
        "section": section,
        "section_title": section_title,
        "article": article,
        "article_title": article_title,
        "clause": clause,
        "point": point,
        "chunk_type": chunk_type,
        "content": content,
        "page_start": page_start,
        "page_end": page_end,
    }


class TestLegalAwareChunker:
    def setup_method(self):
        self.chunker = LegalAwareChunker(document_id="test_doc")

    def test_basic_article_chunking(self):
        """Transform parser chunks: article_intro + 2 clauses."""
        corpus_chunks = [
            _parser_chunk(article="Điều 1", article_title="Title", chunk_type="article_intro", content="Intro text"),
            _parser_chunk(article="Điều 1", article_title="Title", clause="1", chunk_type="clause", content="Clause 1"),
            _parser_chunk(article="Điều 1", article_title="Title", clause="2", chunk_type="clause", content="Clause 2"),
        ]
        chunks = self.chunker.chunk(corpus_chunks)
        assert len(chunks) == 3
        assert chunks[0]["chunk_type"] == "article_intro"
        assert chunks[1]["chunk_type"] == "clause"
        assert chunks[2]["chunk_type"] == "clause"
        # Check chunk_ids are generated
        for c in chunks:
            assert c["chunk_id"].startswith("test_doc_")

    def test_clause_with_points(self):
        """Transform parser chunks: clause_intro + 2 points."""
        corpus_chunks = [
            _parser_chunk(article="Điều 1", article_title="Title", chunk_type="article_intro", content="Article intro"),
            _parser_chunk(article="Điều 1", article_title="Title", clause="1", chunk_type="clause_intro", content="Clause intro"),
            _parser_chunk(article="Điều 1", article_title="Title", clause="1", point="a", chunk_type="point", content="Point A"),
            _parser_chunk(article="Điều 1", article_title="Title", clause="1", point="b", chunk_type="point", content="Point B"),
        ]
        chunks = self.chunker.chunk(corpus_chunks)
        chunk_types = [c["chunk_type"] for c in chunks]
        assert "clause_intro" in chunk_types
        assert chunk_types.count("point") == 2
        # Points should have parent_id linking to clause_intro
        point_chunks = [c for c in chunks if c["chunk_type"] == "point"]
        clause_intro_chunk = [c for c in chunks if c["chunk_type"] == "clause_intro"][0]
        for pc in point_chunks:
            assert pc["parent_id"] == clause_intro_chunk["chunk_id"]

    def test_chapter_section_hierarchy(self):
        """Hierarchy metadata preserved from parser."""
        corpus_chunks = [
            _parser_chunk(article="Điều 1", article_title="Title", clause="1", chunk_type="clause", content="Clause",
                         chapter="Chương I", chapter_title="QUY ĐỊNH CHUNG", section="Mục 1", section_title="Phạm vi"),
        ]
        chunks = self.chunker.chunk(corpus_chunks)
        for chunk in chunks:
            assert chunk["chapter"] == "Chương I"
            assert chunk["chapter_title"] == "QUY ĐỊNH CHUNG"
            assert chunk["section"] == "Mục 1"
            assert chunk["section_title"] == "Phạm vi"
            assert chunk["article"] == "Điều 1"

    def test_preamble_extraction(self):
        """Content before first article becomes preamble chunk."""
        corpus_chunks = [
            _parser_chunk(article=None, chunk_type="article_intro", content="Preamble line 1"),
            _parser_chunk(article=None, chunk_type="article_intro", content="Preamble line 2"),
            _parser_chunk(article="Điều 1", article_title="Title", clause="1", chunk_type="clause", content="Clause"),
        ]
        chunks = self.chunker.chunk(corpus_chunks)
        preamble_chunks = [c for c in chunks if c["chunk_type"] == "preamble"]
        assert len(preamble_chunks) == 1
        assert "Preamble line 1" in preamble_chunks[0]["content"]
        assert "Preamble line 2" in preamble_chunks[0]["content"]
        # Preamble should have no article/clause/point
        assert preamble_chunks[0]["article"] is None
        assert preamble_chunks[0]["clause"] is None

    def test_no_duplicate_parsing(self):
        """LegalAwareChunker no longer re-runs regex on content."""
        # Parser output has joined content; chunker should not re-split
        corpus_chunks = [
            _parser_chunk(article="Điều 1", article_title="Title", chunk_type="article_intro", content="Line 1\nLine 2"),
            _parser_chunk(article="Điều 1", article_title="Title", clause="1", chunk_type="clause", content="Clause content"),
        ]
        chunks = self.chunker.chunk(corpus_chunks)
        assert len(chunks) == 2
        # Content preserved as-is from parser
        assert chunks[0]["content"] == "Line 1\nLine 2"

    def test_consistent_chunk_ids(self):
        """Running chunker twice on same corpus produces same chunk_ids."""
        corpus_chunks = [
            _parser_chunk(article="Điều 1", article_title="Title", chunk_type="article_intro", content="Intro"),
            _parser_chunk(article="Điều 1", article_title="Title", clause="1", chunk_type="clause", content="Clause"),
            _parser_chunk(article="Điều 1", article_title="Title", clause="1", point="a", chunk_type="point", content="Point"),
        ]
        chunks1 = self.chunker.chunk(corpus_chunks)
        chunks2 = self.chunker.chunk(corpus_chunks)
        ids1 = [c["chunk_id"] for c in chunks1]
        ids2 = [c["chunk_id"] for c in chunks2]
        assert ids1 == ids2

    def test_parent_id_for_points(self):
        """Points link to clause_intro via parent_id."""
        corpus_chunks = [
            _parser_chunk(article="Điều 1", article_title="Title", clause="1", chunk_type="clause_intro", content="Clause intro"),
            _parser_chunk(article="Điều 1", article_title="Title", clause="1", point="a", chunk_type="point", content="Point A"),
            _parser_chunk(article="Điều 1", article_title="Title", clause="1", point="b", chunk_type="point", content="Point B"),
        ]
        chunks = self.chunker.chunk(corpus_chunks)
        clause_intro = [c for c in chunks if c["chunk_type"] == "clause_intro"][0]
        points = [c for c in chunks if c["chunk_type"] == "point"]
        for p in points:
            assert p["parent_id"] == clause_intro["chunk_id"]

    def test_document_id_propagated(self):
        """document_id added to all chunks."""
        corpus_chunks = [
            _parser_chunk(article="Điều 1", article_title="Title", clause="1", chunk_type="clause", content="Clause"),
        ]
        chunks = self.chunker.chunk(corpus_chunks)
        assert chunks[0]["document_id"] == "test_doc"

    def test_empty_corpus(self):
        """Empty input returns empty list."""
        chunks = self.chunker.chunk([])
        assert chunks == []

    def test_only_preamble_no_articles(self):
        """Only preamble chunks - no articles."""
        corpus_chunks = [
            _parser_chunk(article=None, chunk_type="article_intro", content="Only preamble"),
        ]
        chunks = self.chunker.chunk(corpus_chunks)
        assert len(chunks) == 1
        assert chunks[0]["chunk_type"] == "preamble"


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
            assert chunk["unit"] == "words"

    def test_fixed_size_overlap_content(self):
        """Verify overlap actually works - adjacent chunks share words."""
        chunker = FixedSizeChunker(chunk_size=10, overlap=4, document_id="test")
        words = [f"w{i}" for i in range(20)]
        lines = [{"text": w, "page": 1} for w in words]
        chunks = chunker.chunk(lines)
        # Step = 6, chunks at 0, 6, 12, 18
        assert len(chunks) == 4
        # Chunk 0: w0-w9, Chunk 1: w6-w15 -> overlap w6-w9
        chunk0_words = chunks[0]["content"].split()
        chunk1_words = chunks[1]["content"].split()
        overlap_words = set(chunk0_words) & set(chunk1_words)
        assert len(overlap_words) >= 4  # At least 4 words overlap

    def test_fixed_size_preserve_metadata_option(self):
        """Test preserve_metadata parameter."""
        chunker = FixedSizeChunker(chunk_size=10, overlap=2, document_id="test", preserve_metadata=True)
        lines = [{"text": f"word{i}", "page": i % 3 + 1} for i in range(15)]
        chunks = chunker.chunk(lines)
        for chunk in chunks:
            assert "page_start" in chunk
            assert "page_end" in chunk


class TestParagraphChunker:
    def test_paragraph_chunking(self):
        chunker = ParagraphChunker(document_id="test")
        text = "Para 1.\n\nPara 2.\n\nPara 3."
        lines = [{"text": line, "page": 1} for line in text.split("\n")]
        chunks = chunker.chunk(lines)
        assert len(chunks) == 3
        for chunk in chunks:
            assert chunk["chunk_type"] == "paragraph"

    def test_paragraph_metadata(self):
        """Paragraph chunks should have page info."""
        chunker = ParagraphChunker(document_id="test")
        lines = [
            {"text": "Para 1 line 1", "page": 1},
            {"text": "", "page": 1},
            {"text": "Para 2 line 1", "page": 2},
        ]
        chunks = chunker.chunk(lines)
        for chunk in chunks:
            assert "page_start" in chunk
            assert "page_end" in chunk


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

    def test_article_chapter_section_metadata(self):
        """Article chunks should preserve chapter and section."""
        chunker = ArticleChunker(document_id="test")
        lines = [
            {"text": "Chương I", "page": 1},
            {"text": "Mục 1", "page": 1},
            {"text": "Điều 1. Title", "page": 1},
            {"text": "Content", "page": 1},
        ]
        chunks = chunker.chunk(lines)
        assert chunks[0]["chapter"] == "Chương I"
        assert chunks[0]["section"] == "Mục 1"


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

    def test_clause_chapter_section_metadata(self):
        """Clause chunks should preserve chapter and section."""
        chunker = ClauseChunker(document_id="test")
        lines = [
            {"text": "Chương I", "page": 1},
            {"text": "Mục 1", "page": 1},
            {"text": "Điều 1. Title", "page": 1},
            {"text": "1. Clause", "page": 1},
            {"text": "Content", "page": 1},
        ]
        chunks = chunker.chunk(lines)
        assert chunks[0]["chapter"] == "Chương I"
        assert chunks[0]["section"] == "Mục 1"

    def test_clause_includes_points(self):
        """Clause chunker includes points in clause content (doesn't separate)."""
        chunker = ClauseChunker(document_id="test")
        lines = [
            {"text": "Điều 1. Title", "page": 1},
            {"text": "1. Clause intro", "page": 1},
            {"text": "a) Point A", "page": 1},
            {"text": "b) Point B", "page": 1},
        ]
        chunks = chunker.chunk(lines)
        assert len(chunks) == 1
        assert "Clause intro" in chunks[0]["content"]
        assert "Point A" in chunks[0]["content"]
        assert "Point B" in chunks[0]["content"]


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


class TestSafeIdPart:
    def test_safe_id_part_none(self):
        assert _safe_id_part(None) == "none"

    def test_safe_id_part_string(self):
        assert _safe_id_part("Điều 1") == "điều_1"
        assert _safe_id_part("Chương I") == "chương_i"
        assert _safe_id_part("a/b") == "a_b"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])