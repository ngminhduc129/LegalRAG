"""
Legal-aware chunking for Vietnamese Labor Law.

Implements chunking strategies that preserve legal hierarchy:
Chapter -> Section -> Article -> Clause -> Point
"""

import re
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF

from legal_rag.parsing.classifier import (
    CHAPTER_RE, SECTION_RE, ARTICLE_RE, CLAUSE_RE, POINT_RE,
    clean_line, is_page_number, extract_pdf_lines,
)
from legal_rag.parsing.constants import DOCUMENT_NAME, DOCUMENT_NUMBER
from legal_rag.chunking.base import BaseChunker


# =========================================================
# CHUNK CREATION
# =========================================================

def join_text(parts: List[str]) -> str:
    """Join text parts into chunk content."""
    return "\n".join(p.strip() for p in parts if p and p.strip()).strip()


def make_chunk(
    chapter: Optional[str],
    chapter_title: Optional[str],
    section: Optional[str],
    section_title: Optional[str],
    article: Optional[str],
    article_title: Optional[str],
    clause: Optional[str],
    point: Optional[str],
    content: str,
    page_start: int,
    page_end: int,
    chunk_type: str,
    document_id: str = "labor_code_2019",
) -> Dict[str, Any]:
    """Create a standardized chunk record."""
    return {
        "chunk_id": f"{document_id}_{article}_{clause}_{point}_{chunk_type}".replace(" ", "_").replace("/", "_").lower(),
        "document_id": document_id,
        "document_name": DOCUMENT_NAME,
        "document_number": DOCUMENT_NUMBER,
        "chapter": chapter,
        "chapter_title": chapter_title,
        "section": section,
        "section_title": section_title,
        "article": article,
        "article_title": article_title,
        "clause": clause,
        "point": point,
        "chunk_type": chunk_type,
        "content": content.strip(),
        "page_start": page_start,
        "page_end": page_end,
    }


# =========================================================
# LEGAL STRUCTURE PARSER (Legal-aware Chunker)
# =========================================================

class LegalAwareChunker(BaseChunker):
    """
    Legal-aware chunker that preserves Vietnamese legal hierarchy.

    Chunk types:
    - article_intro: Article preamble/introductory text
    - clause: Full clause (no sub-points)
    - clause_intro: Clause preamble before points
    - point: Individual point within a clause
    """

    def __init__(self, document_id: str = "labor_code_2019"):
        self.document_id = document_id
        self.chunks = []

        # State
        self.chapter: Optional[str] = None
        self.chapter_title: Optional[str] = None
        self.section: Optional[str] = None
        self.section_title: Optional[str] = None
        self.article: Optional[str] = None
        self.article_title: Optional[str] = None
        self.article_page: Optional[int] = None

        self.article_intro: List[str] = []
        self.article_intro_pages: List[int] = []

        self.clause: Optional[str] = None
        self.clause_parts: List[str] = []
        self.clause_pages: List[int] = []
        self.clause_intro: List[str] = []
        self.clause_intro_pages: List[int] = []
        self.clause_has_points: bool = False

        self.point: Optional[str] = None
        self.point_parts: List[str] = []
        self.point_pages: List[int] = []

    def _emit_article_intro(self):
        content = join_text(self.article_intro)
        if content and self.article is not None:
            pages = self.article_intro_pages
            self.chunks.append(make_chunk(
                chapter=self.chapter,
                chapter_title=self.chapter_title,
                section=self.section,
                section_title=self.section_title,
                article=self.article,
                article_title=self.article_title,
                clause=None,
                point=None,
                content=content,
                page_start=min(pages),
                page_end=max(pages),
                chunk_type="article_intro",
                document_id=self.document_id,
            ))
        self.article_intro = []
        self.article_intro_pages = []

    def _emit_clause_intro(self):
        content = join_text(self.clause_intro)
        if content and self.article is not None:
            pages = self.clause_intro_pages
            self.chunks.append(make_chunk(
                chapter=self.chapter,
                chapter_title=self.chapter_title,
                section=self.section,
                section_title=self.section_title,
                article=self.article,
                article_title=self.article_title,
                clause=self.clause,
                point=None,
                content=content,
                page_start=min(pages),
                page_end=max(pages),
                chunk_type="clause_intro",
                document_id=self.document_id,
            ))
        self.clause_intro = []
        self.clause_intro_pages = []

    def _emit_clause(self):
        if self.clause is not None and not self.clause_has_points:
            content = join_text(self.clause_parts)
            if content:
                pages = self.clause_pages
                self.chunks.append(make_chunk(
                    chapter=self.chapter,
                    chapter_title=self.chapter_title,
                    section=self.section,
                    section_title=self.section_title,
                    article=self.article,
                    article_title=self.article_title,
                    clause=self.clause,
                    point=None,
                    content=content,
                    page_start=min(pages),
                    page_end=max(pages),
                    chunk_type="clause",
                    document_id=self.document_id,
                ))
        elif self.clause is not None and self.clause_has_points:
            self._emit_clause_intro()

        self.clause = None
        self.clause_parts = []
        self.clause_pages = []
        self.clause_intro = []
        self.clause_intro_pages = []
        self.clause_has_points = False
        self.point = None
        self.point_parts = []
        self.point_pages = []

    def _emit_point(self):
        if self.point is not None:
            content = join_text(self.point_parts)
            if content and self.article is not None:
                pages = self.point_pages
                self.chunks.append(make_chunk(
                    chapter=self.chapter,
                    chapter_title=self.chapter_title,
                    section=self.section,
                    section_title=self.section_title,
                    article=self.article,
                    article_title=self.article_title,
                    clause=self.clause,
                    point=self.point,
                    content=content,
                    page_start=min(pages),
                    page_end=max(pages),
                    chunk_type="point",
                    document_id=self.document_id,
                ))
        self.point = None
        self.point_parts = []
        self.point_pages = []

    def _close_article(self):
        self._emit_point()
        self._emit_clause()
        self._emit_article_intro()
        self.article = None
        self.article_title = None
        self.article_page = None

    def _is_same_article(self, candidate_article: str, candidate_title: str) -> bool:
        """Check if this is a duplicate article header (e.g., at page break)."""
        return (
            self.article is not None
            and self.article == candidate_article
            and (
                self.article_title is None
                or not candidate_title
                or self.article_title.strip().casefold() == candidate_title.strip().casefold()
            )
        )

    def chunk(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse lines into legal-aware chunks."""
        self.chunks = []
        self.chapter = None
        self.chapter_title = None
        self.section = None
        self.section_title = None
        self.article = None
        self.article_title = None
        self.article_page = None
        self.article_intro = []
        self.article_intro_pages = []
        self.clause = None
        self.clause_parts = []
        self.clause_pages = []
        self.clause_intro = []
        self.clause_intro_pages = []
        self.clause_has_points = False
        self.point = None
        self.point_parts = []
        self.point_pages = []

        i = 0
        while i < len(lines):
            item = lines[i]
            text = item["text"]
            page = item["page"]

            # CHAPTER
            chapter_match = CHAPTER_RE.match(text)
            if chapter_match:
                self._close_article()
                self.chapter = f"Chương {chapter_match.group(1)}"
                self.chapter_title = None
                if i + 1 < len(lines):
                    next_text = lines[i + 1]["text"]
                    if not any(p.match(next_text) for p in [CHAPTER_RE, SECTION_RE, ARTICLE_RE, CLAUSE_RE]):
                        self.chapter_title = next_text
                        i += 1
                i += 1
                continue

            # SECTION
            section_match = SECTION_RE.match(text)
            if section_match:
                self._close_article()
                self.section = f"Mục {section_match.group(1)}"
                self.section_title = None
                if i + 1 < len(lines):
                    next_text = lines[i + 1]["text"]
                    if not any(p.match(next_text) for p in [CHAPTER_RE, SECTION_RE, ARTICLE_RE, CLAUSE_RE]):
                        self.section_title = next_text
                        i += 1
                i += 1
                continue

            # ARTICLE
            article_match = ARTICLE_RE.match(text)
            if article_match:
                article_number = article_match.group(1)
                article_title_text = article_match.group(2).strip()
                candidate_article = f"Điều {article_number}"

                if self._is_same_article(candidate_article, article_title_text):
                    i += 1
                    continue

                self._close_article()
                self.article = candidate_article
                self.article_title = article_title_text or None
                self.article_page = page
                i += 1
                continue

            # Skip content before first article
            if self.article is None:
                i += 1
                continue

            # CLAUSE
            clause_match = CLAUSE_RE.match(text)
            if clause_match:
                # Emit article_intro if this is the first clause of the article
                if self.article_intro and self.clause is None:
                    self._emit_article_intro()

                self._emit_point()
                self._emit_clause()
                self.clause = clause_match.group(1)
                self.clause_parts = [clause_match.group(2)]
                self.clause_pages = [page]
                self.clause_has_points = False
                i += 1
                continue

            # POINT
            point_match = POINT_RE.match(text)
            if point_match and self.clause is not None:
                if not self.clause_has_points:
                    self.clause_intro = list(self.clause_parts)
                    self.clause_intro_pages = list(self.clause_pages)
                    self.clause_parts = []
                    self.clause_pages = []
                    self.clause_has_points = True
                self._emit_point()
                self.point = point_match.group(1).lower()
                self.point_parts = [point_match.group(2)]
                self.point_pages = [page]
                i += 1
                continue

            # CONTENT
            if self.point is not None:
                self.point_parts.append(text)
                self.point_pages.append(page)
            elif self.clause is not None:
                self.clause_parts.append(text)
                self.clause_pages.append(page)
            else:
                self.article_intro.append(text)
                self.article_intro_pages.append(page)

            i += 1

        self._close_article()
        return self.chunks

    def chunk_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Parse PDF directly into chunks."""
        lines = extract_pdf_lines(pdf_path)
        return self.chunk(lines)


# =========================================================
# BASELINE CHUNKERS (for comparison)
# =========================================================

class FixedSizeChunker(BaseChunker):
    """Fixed-size chunking with overlap (baseline CH01)."""

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        document_id: str = "labor_code_2019",
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.document_id = document_id

    def chunk(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk from lines (joins all text first)."""
        text = "\n".join(line["text"] for line in lines)
        return self._chunk_text(text)

    def chunk_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Parse PDF directly into chunks."""
        lines = extract_pdf_lines(pdf_path)
        text = "\n".join(line["text"] for line in lines)
        return self._chunk_text(text)

    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            if not chunk_words:
                break
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "chunk_id": f"{self.document_id}_fixed_{i // (self.chunk_size - self.overlap)}",
                "document_id": self.document_id,
                "content": chunk_text,
                "chunk_type": "fixed_size",
                "chunk_index": i // (self.chunk_size - self.overlap),
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
            })
        return chunks


class ParagraphChunker(BaseChunker):
    """Paragraph-based chunking (baseline CH02)."""

    def __init__(self, document_id: str = "labor_code_2019"):
        self.document_id = document_id

    def chunk(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        text = "\n".join(line["text"] for line in lines)
        return self._chunk_text(text)

    def chunk_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        lines = extract_pdf_lines(pdf_path)
        text = "\n".join(line["text"] for line in lines)
        return self._chunk_text(text)

    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        for i, para in enumerate(paragraphs):
            chunks.append({
                "chunk_id": f"{self.document_id}_para_{i}",
                "document_id": self.document_id,
                "content": para,
                "chunk_type": "paragraph",
                "paragraph_index": i,
            })
        return chunks


class ArticleChunker(BaseChunker):
    """Article-level chunking - each article as one chunk (baseline CH03)."""

    def __init__(self, document_id: str = "labor_code_2019"):
        self.document_id = document_id

    def chunk(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Group lines by article
        articles = {}
        current_article = None
        for line in lines:
            match = ARTICLE_RE.match(line["text"])
            if match:
                current_article = f"Điều {match.group(1)}"
                if current_article not in articles:
                    articles[current_article] = {"lines": [], "title": match.group(2).strip(), "pages": set()}
            if current_article:
                articles[current_article]["lines"].append(line["text"])
                articles[current_article]["pages"].add(line["page"])

        chunks = []
        for i, (article, data) in enumerate(articles.items()):
            content = "\n".join(data["lines"])
            pages = sorted(data["pages"])
            chunks.append({
                "chunk_id": f"{self.document_id}_article_{i}",
                "document_id": self.document_id,
                "article": article,
                "article_title": data["title"],
                "content": content,
                "chunk_type": "article",
                "page_start": min(pages),
                "page_end": max(pages),
            })
        return chunks

    def chunk_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        lines = extract_pdf_lines(pdf_path)
        return self.chunk(lines)


class ClauseChunker(BaseChunker):
    """Clause-level chunking - each clause as one chunk (baseline CH04)."""

    def __init__(self, document_id: str = "labor_code_2019"):
        self.document_id = document_id

    def chunk(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        current_article = None
        current_clause = None
        clause_parts = []
        clause_pages = []
        clause_idx = 0

        for line in lines:
            text = line["text"]
            page = line["page"]

            article_match = ARTICLE_RE.match(text)
            if article_match:
                current_article = f"Điều {article_match.group(1)}"
                current_clause = None
                if clause_parts:
                    chunks.append({
                        "chunk_id": f"{self.document_id}_clause_{clause_idx}",
                        "document_id": self.document_id,
                        "article": current_article,
                        "clause": current_clause,
                        "content": "\n".join(clause_parts),
                        "chunk_type": "clause",
                        "page_start": min(clause_pages),
                        "page_end": max(clause_pages),
                    })
                    clause_idx += 1
                    clause_parts = []
                    clause_pages = []
                continue

            clause_match = CLAUSE_RE.match(text)
            if clause_match and current_article:
                if clause_parts:
                    chunks.append({
                        "chunk_id": f"{self.document_id}_clause_{clause_idx}",
                        "document_id": self.document_id,
                        "article": current_article,
                        "clause": current_clause,
                        "content": "\n".join(clause_parts),
                        "chunk_type": "clause",
                        "page_start": min(clause_pages),
                        "page_end": max(clause_pages),
                    })
                    clause_idx += 1
                current_clause = clause_match.group(1)
                clause_parts = [clause_match.group(2)]
                clause_pages = [page]
                continue

            if current_clause:
                clause_parts.append(text)
                clause_pages.append(page)

        # Last clause
        if clause_parts:
            chunks.append({
                "chunk_id": f"{self.document_id}_clause_{clause_idx}",
                "document_id": self.document_id,
                "article": current_article,
                "clause": current_clause,
                "content": "\n".join(clause_parts),
                "chunk_type": "clause",
                "page_start": min(clause_pages),
                "page_end": max(clause_pages),
            })

        return chunks

    def chunk_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        lines = extract_pdf_lines(pdf_path)
        return self.chunk(lines)


# =========================================================
# UTILS
# =========================================================

def save_jsonl(chunks: List[Dict[str, Any]], output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def save_csv(chunks: List[Dict[str, Any]], output_path: Path):
    if not chunks:
        print("No chunks to write.")
        return
    fieldnames = list(chunks[0].keys())
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(chunks)