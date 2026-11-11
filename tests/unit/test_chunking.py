"""Unit tests for chunking module."""

import pytest
import json
from pathlib import Path
from typing import Any, Dict, Optional

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
from legal_rag.chunking.identity import (
    StructureRegistry,
    IdentityResolver,
    OperationDetector,
)
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

    def test_repeated_quoted_numbering_gets_occurrence(self):
        """Quoted amendments repeat article/clause/point numbering."""
        corpus_chunks = [
            _parser_chunk(article="Điều 219", article_title="Sửa đổi", chunk_type="article_intro", content="Intro"),
            _parser_chunk(article="Điều 219", article_title="Sửa đổi", clause="1", point="a", chunk_type="point", content="First law point a"),
            _parser_chunk(article="Điều 219", article_title="Sửa đổi", clause="1", point="a", chunk_type="point", content="Second law point a"),
        ]
        chunks = self.chunker.chunk(corpus_chunks)
        points = [c for c in chunks if c["chunk_type"] == "point"]
        assert len({c["chunk_id"] for c in points}) == 2
        assert [c["structure_occurrence"] for c in points] == [1, 2]

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

    def test_appendix_only_not_merged_into_preamble(self):
        """Appendix-only documents must keep appendix chunks separate."""
        corpus_chunks = [
            _parser_chunk(article=None, chunk_type="appendix", content="Appendix table A"),
            _parser_chunk(article=None, chunk_type="appendix", content="Appendix table B"),
        ]
        chunks = self.chunker.chunk(corpus_chunks)
        assert [c["chunk_type"] for c in chunks] == ["appendix", "appendix"]
        assert chunks[0]["chunk_id"].startswith("test_doc_appendix_")
        assert chunks[0]["parent_id"] is None

    def test_registry_metadata_fills_missing_fields(self):
        """Chunker fills missing document metadata from the registry entry."""
        from legal_rag.corpus.registry import DocumentMeta

        document = DocumentMeta(
            document_id="nd_x",
            document_name="Nghị định X",
            document_number="1/2020/NĐ-CP",
            doc_type="Nghị định",
        )
        chunker = LegalAwareChunker(document_id=document)
        corpus_chunks = [
            {
                "article": "Điều 1",
                "clause": "1",
                "point": None,
                "chunk_type": "clause",
                "content": "Clause",
                "page_start": 1,
                "page_end": 1,
            }
        ]
        chunk = chunker.chunk(corpus_chunks)[0]
        assert chunk["document_id"] == "nd_x"
        assert chunk["document_name"] == "Nghị định X"
        assert chunk["document_number"] == "1/2020/NĐ-CP"
        assert chunk["document_type"] == "Nghị định"


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


class TestBaselineChunkerCorpusMetadata:
    def test_fixed_size_preserves_all_source_structures(self):
        corpus_chunks = [
            _parser_chunk(
                article="Điều 1",
                article_title="Title 1",
                clause="1",
                content="one two three",
                page_start=2,
                page_end=2,
            ),
            _parser_chunk(
                article="Điều 2",
                article_title="Title 2",
                clause="1",
                content="four five",
                page_start=3,
                page_end=3,
            ),
        ]
        chunks = FixedSizeChunker(
            chunk_size=4,
            overlap=0,
            document_id="test",
        ).chunk(corpus_chunks)

        assert chunks[0]["articles"] == ["Điều 1", "Điều 2"]
        assert chunks[0]["article"] is None
        assert chunks[0]["page_start"] == 2
        assert chunks[0]["page_end"] == 3
        assert chunks[0]["document_name"] == "Bộ luật Lao động"
        assert chunks[0]["document_number"] == "125/VBHN-VPQH"
        assert [ref["article"] for ref in chunks[0]["source_refs"]] == [
            "Điều 1",
            "Điều 2",
        ]
        assert validate_chunks(chunks).errors == 0

    def test_paragraph_preserves_source_metadata(self):
        corpus_chunks = [
            _parser_chunk(
                article="Điều 1",
                article_title="Title 1",
                content="First sentence.\nContinuation",
                page_start=4,
                page_end=4,
            ),
            _parser_chunk(
                article="Điều 2",
                article_title="Title 2",
                content="Second sentence.",
                page_start=5,
                page_end=5,
            ),
        ]
        chunks = ParagraphChunker(document_id="test").chunk(corpus_chunks)

        assert chunks[0]["article"] == "Điều 1"
        assert chunks[0]["articles"] == ["Điều 1"]
        assert chunks[0]["page_start"] == 4
        assert chunks[0]["page_end"] == 4
        assert chunks[1]["articles"] == ["Điều 1", "Điều 2"]
        assert chunks[1]["page_start"] == 4
        assert chunks[1]["page_end"] == 5
        assert all(chunk["source_refs"] for chunk in chunks)

    def test_article_and_clause_include_document_metadata_and_sources(self):
        corpus_chunks = [
            _parser_chunk(
                article="Điều 1",
                article_title="Title 1",
                clause="1",
                point="a",
                chunk_type="point",
                content="Point A.",
                page_start=6,
                page_end=6,
            ),
            _parser_chunk(
                article="Điều 1",
                article_title="Title 1",
                clause="1",
                point="b",
                chunk_type="point",
                content="Point B.",
                page_start=7,
                page_end=7,
            ),
        ]

        article = ArticleChunker(document_id="test").chunk(corpus_chunks)[0]
        clause = ClauseChunker(document_id="test").chunk(corpus_chunks)[0]

        for chunk in (article, clause):
            assert chunk["document_name"] == "Bộ luật Lao động"
            assert chunk["document_number"] == "125/VBHN-VPQH"
            assert chunk["page_start"] == 6
            assert chunk["page_end"] == 7
            assert len(chunk["source_refs"]) == 2
        assert clause["points"] == ["a", "b"]

    def test_clause_chunker_keeps_article_intro_content(self):
        corpus_chunks = [
            _parser_chunk(
                article="Điều 1",
                article_title="Title 1",
                chunk_type="article_intro",
                content="Article introduction.",
                page_start=1,
                page_end=1,
            ),
            _parser_chunk(
                article="Điều 1",
                article_title="Title 1",
                clause="1",
                chunk_type="clause",
                content="Clause content.",
                page_start=1,
                page_end=1,
            ),
        ]

        chunks = ClauseChunker(document_id="test").chunk(corpus_chunks)

        assert [chunk["chunk_type"] for chunk in chunks] == [
            "article_intro",
            "clause",
        ]
        assert [chunk["content"] for chunk in chunks] == [
            "Article introduction.",
            "Clause content.",
        ]
        assert all(chunk["source_refs"] for chunk in chunks)


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

    def test_appendix_without_article_is_valid(self):
        """Appendices have no article by design, so no article is required."""
        chunks = [
            {
                "chunk_id": "test_appendix_1",
                "document_id": "test",
                "document_name": "Test",
                "document_number": "1",
                "chunk_type": "appendix",
                "content": "Appendix table.",
                "page_start": 2,
                "page_end": 2,
            }
        ]
        report = validate_chunks(chunks)
        assert report.errors == 0
        assert not [
            issue for issue in report.issues if issue.code == "MISSING_METADATA"
        ]

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
                "content": "Content.",
                "page_start": 1,
                "page_end": 1,
            }
        ]
        report = validate_chunks(chunks)
        duplicates = [i for i in report.issues if i.code == "DUPLICATE_STRUCTURE_KEY"]
        assert len(duplicates) > 0

    def test_shared_structure_key_different_content(self):
        # Quoted amendments (e.g. Dieu 219) reuse structure keys with
        # different content: informational, not a warning.
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
                "content": "Different quoted content.",
                "page_start": 1,
                "page_end": 1,
            }
        ]
        report = validate_chunks(chunks)
        duplicates = [i for i in report.issues if i.code == "DUPLICATE_STRUCTURE_KEY"]
        assert len(duplicates) == 0
        shared = [i for i in report.issues if i.code == "SHARED_STRUCTURE_KEY"]
        assert len(shared) == 1
        assert shared[0].severity == "INFO"


class TestSafeIdPart:
    def test_safe_id_part_none(self):
        assert _safe_id_part(None) == "none"

    def test_safe_id_part_string(self):
        assert _safe_id_part("Điều 1") == "điều_1"
        assert _safe_id_part("Chương I") == "chương_i"
        assert _safe_id_part("a/b") == "a_b"


class TestStructureIdentity:
    """Tests A–J for structure-identity-aware chunking."""

    def _identity_chunk(
        self,
        instance_id: str,
        parent_instance_id: Optional[str],
        structure_role: str,
        operation_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        base = _parser_chunk(**kwargs)
        base.update({
            "structure_instance_id": instance_id,
            "parent_instance_id": parent_instance_id,
            "structure_role": structure_role,
        })
        if operation_id is not None:
            base["operation_id"] = operation_id
        return base

    # ── Test A — Identity propagation ──────────────────────────────────

    def test_identity_propagation(self):
        """Every output chunk preserves its parser instance IDs and role."""
        corpus_chunks = [
            self._identity_chunk("si1", None, "main",
                article="Điều 1", article_title="Title", chunk_type="article_intro", content="Intro"),
            self._identity_chunk("si2", "si1", "main",
                article="Điều 1", clause="1", chunk_type="clause_intro", content="Clause intro"),
            self._identity_chunk("si3", "si2", "main",
                article="Điều 1", clause="1", point="a", chunk_type="point", content="Point A"),
        ]
        chunks = LegalAwareChunker(document_id="test").chunk(corpus_chunks)
        by_id = {
            c["structure_instance_id"]: c
            for c in chunks if c.get("structure_instance_id")
        }
        assert by_id["si1"]["structure_instance_id"] == "si1"
        assert by_id["si1"]["parent_instance_id"] is None
        assert by_id["si1"]["structure_role"] == "main"
        assert by_id["si2"]["structure_instance_id"] == "si2"
        assert by_id["si2"]["parent_instance_id"] == "si1"
        assert by_id["si2"]["structure_role"] == "main"
        assert by_id["si3"]["structure_instance_id"] == "si3"
        assert by_id["si3"]["parent_instance_id"] == "si2"
        assert by_id["si3"]["structure_role"] == "main"

    # ── Test B — Duplicate article numbers ─────────────────────────────

    def test_duplicate_article_numbers(self):
        """Same article number, different instance IDs stay separate."""
        corpus_chunks = [
            self._identity_chunk("si1", None, "main",
                article="Điều 35", article_title="Title", chunk_type="article_intro", content="Intro A"),
            self._identity_chunk("si2", "si1", "main",
                article="Điều 35", clause="1", chunk_type="clause", content="Clause A"),
            self._identity_chunk("si3", None, "main",
                article="Điều 35", article_title="Quoted", chunk_type="article_intro", content="Intro B"),
            self._identity_chunk("si4", "si3", "main",
                article="Điều 35", clause="1", chunk_type="clause", content="Clause B"),
        ]
        chunks = LegalAwareChunker(document_id="test").chunk(corpus_chunks)
        article_intros = [c for c in chunks if c.get("chunk_type") == "article_intro"]
        assert len(article_intros) == 2
        ids = {c["structure_instance_id"] for c in article_intros}
        assert ids == {"si1", "si3"}
        clauses = [c for c in chunks if c.get("chunk_type") == "clause"]
        si2_parent = [c for c in clauses if c.get("structure_instance_id") == "si2"][0]
        si4_parent = [c for c in clauses if c.get("structure_instance_id") == "si4"][0]
        assert si2_parent["parent_instance_id"] == "si1"
        assert si4_parent["parent_instance_id"] == "si3"
        assert si2_parent["parent_instance_id"] != si4_parent["parent_instance_id"]

    # ── Test C — Quoted replacement ────────────────────────────────────

    def test_quoted_replacement(self):
        """Quoted replacement has distinct role and identity from outer article."""
        corpus_chunks = [
            self._identity_chunk("si1", None, "main",
                article="Điều 1", article_title="Sửa đổi", chunk_type="article_intro", content="Outer amendment"),
            self._identity_chunk("si2", "si1", "quoted_replacement",
                article="Điều 4", article_title="Quoted replacement", chunk_type="article_intro", content="Quoted Art 4"),
            self._identity_chunk("si3", "si2", "quoted_replacement",
                article="Điều 4", clause="1", chunk_type="clause", content="Quoted clause"),
            self._identity_chunk("si4", "si3", "quoted_replacement",
                article="Điều 4", clause="1", point="a", chunk_type="point", content="Quoted point a"),
        ]
        chunks = LegalAwareChunker(document_id="test").chunk(corpus_chunks)
        quoted = [c for c in chunks if c.get("structure_role") == "quoted_replacement"]
        assert len(quoted) == 3  # article_intro, clause, point
        # Quoted replacement article_intro points to outer article (si1)
        quoted_intro = [c for c in quoted if c.get("chunk_type") == "article_intro"][0]
        assert quoted_intro["parent_instance_id"] == "si1"
        # Point's parent_instance_id is the quoted clause (si3)
        quoted_point = [c for c in quoted if c.get("chunk_type") == "point"][0]
        assert quoted_point["parent_instance_id"] == "si3"
        # No cross-role merge: main article_intro stays separate
        main_intros = [c for c in chunks if c.get("structure_role") == "main" and c.get("chunk_type") == "article_intro"]
        assert len(main_intros) == 1
        assert main_intros[0]["structure_instance_id"] == "si1"

    # ── Test D — Multiple amendment operations ─────────────────────────

    def test_multiple_amendment_operations(self):
        """Two operations modifying different articles get different op IDs."""
        corpus_chunks = [
            self._identity_chunk("si1", None, "main",
                article="Điều 1", article_title="Sửa đổi", chunk_type="article_intro", content="Outer",
                operation_id="op1"),
            self._identity_chunk("si2", "si1", "quoted_replacement",
                article="Điều 4", article_title="Quoted 4", chunk_type="article_intro", content="Q4",
                operation_id="op1"),
            self._identity_chunk("si3", "si2", "quoted_replacement",
                article="Điều 4", clause="1", chunk_type="clause", content="Q4 c1",
                operation_id="op1"),
            self._identity_chunk("si4", None, "main",
                article="Điều 2", article_title="Sửa đổi", chunk_type="article_intro", content="Outer 2",
                operation_id="op2"),
            self._identity_chunk("si5", "si4", "quoted_replacement",
                article="Điều 5", article_title="Quoted 5", chunk_type="article_intro", content="Q5",
                operation_id="op2"),
        ]
        chunks = LegalAwareChunker(document_id="test").chunk(corpus_chunks)
        op1_chunks = [c for c in chunks if c.get("operation_id") == "op1"]
        op2_chunks = [c for c in chunks if c.get("operation_id") == "op2"]
        assert len(op1_chunks) == 3  # si1, si2, si3
        assert len(op2_chunks) == 2  # si4, si5
        # Chunks within the same operation share its ID
        assert all(c["operation_id"] == "op1" for c in op1_chunks)
        assert all(c["operation_id"] == "op2" for c in op2_chunks)
        # Different operations never merge
        assert not any(c.get("operation_id") == "op1" for c in op2_chunks)

    def test_unsupported_operation_null_id(self):
        """Unsupported operations retain null operation_id."""
        corpus_chunks = [
            self._identity_chunk("si1", None, "main",
                article="Điều 1", article_title="Title", chunk_type="article_intro", content="Normal"),
        ]
        chunks = LegalAwareChunker(document_id="test").chunk(corpus_chunks)
        assert chunks[0].get("operation_id") is None

    # ── Test E — Parent resolution ─────────────────────────────────────

    def test_parent_resolution_uses_instance_id(self):
        """Parent resolution uses parent_instance_id, not article/clause tuples."""
        corpus_chunks = [
            self._identity_chunk("si1", None, "main",
                article="Điều 1", article_title="Title", chunk_type="article_intro", content="Intro"),
            self._identity_chunk("si2", "si1", "main",
                article="Điều 1", clause="1", chunk_type="clause_intro", content="Clause intro"),
            self._identity_chunk("si3", "si2", "main",
                article="Điều 1", clause="1", point="a", chunk_type="point", content="Point A"),
        ]
        chunks = LegalAwareChunker(document_id="test").chunk(corpus_chunks)
        point = [c for c in chunks if c.get("chunk_type") == "point"][0]
        clause_intro = [c for c in chunks if c.get("chunk_type") == "clause_intro"][0]
        # parent_id resolves to the clause_intro's chunk_id
        assert point["parent_id"] == clause_intro["chunk_id"]
        # parent_instance_id preserved as legal structure metadata
        assert point["parent_instance_id"] == "si2"
        # parent_id is distinct from parent_instance_id
        assert point["parent_id"] != point["parent_instance_id"]

    def test_invalid_parent_reference_detected(self):
        """Invalid parent references are detected (warning), not silently fixed."""
        corpus_chunks = [
            self._identity_chunk("si1", "si_missing", "main",
                article="Điều 1", article_title="Title", chunk_type="article_intro", content="Intro"),
        ]
        chunks = LegalAwareChunker(document_id="test").chunk(corpus_chunks)
        # Chunk is preserved even though parent is missing
        assert chunks[0]["structure_instance_id"] == "si1"
        assert chunks[0]["parent_instance_id"] == "si_missing"
        # parent_id cannot be resolved (parent not in registry)
        assert chunks[0]["parent_id"] is None

    def test_missing_parent_metadata_no_data_loss(self):
        """Missing parent metadata does not cause data loss."""
        corpus_chunks = [
            {
                "article": "Điều 1",
                "article_title": "Title",
                "clause": "1",
                "point": "a",
                "chunk_type": "point",
                "content": "Point content",
                "page_start": 1,
                "page_end": 1,
                # No instance IDs at all
            },
        ]
        chunks = LegalAwareChunker(document_id="test").chunk(corpus_chunks)
        assert len(chunks) == 1
        assert chunks[0]["content"] == "Point content"
        assert chunks[0]["parent_id"] is None
        assert chunks[0].get("structure_instance_id") is None

    # ── Test F — Multiple chunks per structure ─────────────────────────

    def test_multiple_chunks_per_structure(self):
        """One structure can produce multiple chunks; all share instance ID."""
        corpus_chunks = [
            self._identity_chunk("si1", None, "main",
                article="Điều 1", article_title="Title", chunk_type="article_intro", content="Intro part 1"),
            self._identity_chunk("si1", None, "main",
                article="Điều 1", article_title="Title", chunk_type="article_intro", content="Intro part 2"),
        ]
        chunks = LegalAwareChunker(document_id="test").chunk(corpus_chunks)
        intros = [c for c in chunks if c.get("chunk_type") == "article_intro"]
        assert len(intros) == 2
        assert all(c["structure_instance_id"] == "si1" for c in intros)
        assert len({c["chunk_id"] for c in intros}) == 2  # chunk IDs remain unique

    # ── Test G — Intro metadata ────────────────────────────────────────

    def test_intro_metadata_preserved(self):
        """Preamble, chapter/section/article/clause intro metadata preserved."""
        corpus_chunks = [
            {
                "article": None, "clause": None, "point": None,
                "chunk_type": "preamble", "content": "Preamble text",
                "page_start": 1, "page_end": 1,
                "structure_role": "main",
                "document_id": "test",
            },
            self._identity_chunk("si_a1", None, "main",
                article="Điều 1", article_title="Title", chunk_type="article_intro", content="Article intro"),
            self._identity_chunk("si_s1", "si_a1", "main",
                article=None, clause=None, point=None,
                chunk_type="section_intro", content="Section intro", page_start=1, page_end=1),
            self._identity_chunk("si_c1", "si_a1", "main",
                article="Điều 1", clause="1", chunk_type="clause_intro", content="Clause intro"),
        ]
        chunks = LegalAwareChunker(document_id="test").chunk(corpus_chunks)
        types = [c["chunk_type"] for c in chunks]
        assert "preamble" in types
        assert "article_intro" in types
        assert "section_intro" in types
        assert "clause_intro" in types
        # Source order maintained: preamble before article_intro
        assert types.index("preamble") < types.index("article_intro")
        # Identity metadata preserved
        article_intro = [c for c in chunks if c.get("chunk_type") == "article_intro"][0]
        assert article_intro["structure_instance_id"] == "si_a1"
        assert article_intro["structure_role"] == "main"
        section_intro = [c for c in chunks if c.get("chunk_type") == "section_intro"][0]
        assert section_intro["structure_instance_id"] == "si_s1"
        assert section_intro["parent_instance_id"] == "si_a1"

    def test_cross_role_isolation(self):
        """main, amendment, quoted_replacement, appendix do not merge."""
        corpus_chunks = [
            self._identity_chunk("si1", None, "main",
                article="Điều 1", article_title="Main", chunk_type="article_intro", content="Main content"),
            self._identity_chunk("si2", "si1", "quoted_replacement",
                article="Điều 4", article_title="Quoted", chunk_type="article_intro", content="Quoted content"),
            {
                "article": None, "clause": None, "point": None,
                "chunk_type": "appendix", "content": "Appendix table",
                "page_start": 9, "page_end": 9,
                "structure_role": "appendix",
            },
        ]
        chunks = LegalAwareChunker(document_id="test").chunk(corpus_chunks)
        roles = {c.get("structure_role") for c in chunks if c.get("structure_role")}
        assert "main" in roles
        assert "quoted_replacement" in roles
        assert "appendix" in roles
        # No cross-role merge: each chunk keeps its own role
        main = [c for c in chunks if c.get("structure_role") == "main"]
        quoted = [c for c in chunks if c.get("structure_role") == "quoted_replacement"]
        appendix = [c for c in chunks if c.get("chunk_type") == "appendix"]
        assert len(main) == 1
        assert len(quoted) == 1
        assert len(appendix) == 1
        assert main[0]["content"] == "Main content"
        assert quoted[0]["content"] == "Quoted content"
        assert appendix[0]["content"] == "Appendix table"

    # ── Test I — Backward compatibility ────────────────────────────────

    def test_legacy_chunks_without_instance_ids(self):
        """Legacy parser chunks without instance IDs do not crash."""
        corpus_chunks = [
            _parser_chunk(article="Điều 1", article_title="Title", chunk_type="article_intro", content="Intro"),
            _parser_chunk(article="Điều 1", clause="1", chunk_type="clause_intro", content="Clause intro"),
            _parser_chunk(article="Điều 1", clause="1", point="a", chunk_type="point", content="Point A"),
        ]
        chunks = LegalAwareChunker(document_id="test").chunk(corpus_chunks)
        assert len(chunks) == 3
        # Content remains available
        assert chunks[0]["content"] == "Intro"
        assert chunks[2]["content"] == "Point A"
        # Missing identity is explicitly absent (not fabricated)
        assert chunks[0].get("structure_instance_id") is None
        assert chunks[0].get("parent_instance_id") is None
        # parent_id for points resolves via clause_intro string-key fallback
        point = chunks[2]
        clause_intro = chunks[1]
        assert point["parent_id"] == clause_intro["chunk_id"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestLaborCode2019Benchmarks:
    """Fixed benchmarks on rebuilt labor_code_2019 chunks.

    These verify that the chunk_id↔metadata invariant holds for specific
    known chunks after the off-by-one fix (legal_aware.py:357) and the
    page-86 patch are applied.
    """

    JSONL_PATH = Path(__file__).resolve().parents[2] / "data" / "chunks" / "test_reg_labor" / "all_legal_aware.jsonl"

    def _load_chunk(self, chunk_id: str) -> Dict[str, Any]:
        if not self.JSONL_PATH.exists():
            pytest.skip(f"Chunks file not found: {self.JSONL_PATH}")
        with open(self.JSONL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                c = json.loads(line)
                if c.get("chunk_id") == chunk_id:
                    return c
        pytest.fail(f"Chunk not found: {chunk_id}")

    def test_dieu_124_3_starts_with_cach_chuc(self):
        c = self._load_chunk("labor_code_2019_điều_124_3_none_clause")
        assert c["article"] == "Điều 124"
        assert c["clause"] == "3"
        assert c["point"] is None
        assert c["content"].strip().startswith("Cách chức")

    def test_dieu_124_4_starts_with_sa_thai(self):
        c = self._load_chunk("labor_code_2019_điều_124_4_none_clause")
        assert c["article"] == "Điều 124"
        assert c["clause"] == "4"
        assert c["point"] is None
        assert c["content"].strip().startswith("Sa thải")

    def test_dieu_35_2_d_point_starts_with_quay_roi(self):
        c = self._load_chunk("labor_code_2019_điều_35_2_d_point")
        assert c["article"] == "Điều 35"
        assert c["clause"] == "2"
        assert c["point"] == "d"
        assert c["content"].strip().startswith("Bị quấy rối tình dục")

    def test_dieu_115_1_a_point_starts_with_ket_hon(self):
        c = self._load_chunk("labor_code_2019_điều_115_1_a_point")
        assert c["article"] == "Điều 115"
        assert c["clause"] == "1"
        assert c["point"] == "a"
        assert c["content"].strip().startswith("Kết hôn: nghỉ 03 ngày")