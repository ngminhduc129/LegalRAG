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


def _safe_id_part(value: Optional[str]) -> str:
    """Convert a value to a safe ID part, handling None."""
    if value is None:
        return "none"
    return value.replace(" ", "_").replace("/", "_").lower()


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
    parent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a standardized chunk record."""
    chunk_id = f"{document_id}_{_safe_id_part(article)}_{_safe_id_part(clause)}_{_safe_id_part(point)}_{chunk_type}"
    return {
        "chunk_id": chunk_id,
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
        "parent_id": parent_id,
    }


# =========================================================
# LEGAL STRUCTURE PARSER (Legal-aware Chunker)
# =========================================================

class LegalAwareChunker(BaseChunker):
    """
    Legal-aware chunker that transforms parser output into final chunks.

    Consumes corpus chunks from LegalStructureParser (already parsed with hierarchy)
    and adds: chunk_id, document_id, parent_id, preamble extraction.

    Chunk types (output):
    - preamble: Content before first article
    - article_intro: Article preamble/introductory text
    - clause: Full clause (no sub-points)
    - clause_intro: Clause preamble before points
    - point: Individual point within a clause (includes clause_intro context)
    """

    def __init__(self, document_id: str = "labor_code_2019"):
        self.document_id = document_id

    def chunk(self, corpus_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform parser output chunks into final chunks with IDs and parent links.

        Args:
            corpus_chunks: List of chunks from LegalStructureParser with fields:
                document_name, document_number, chapter, chapter_title,
                section, section_title, article, article_title, clause, point,
                chunk_type, content, page_start, page_end

        Returns:
            List of transformed chunks with added chunk_id, document_id, parent_id
        """
        if not corpus_chunks:
            return []

        # Find first article chunk to identify preamble
        first_article_idx = None
        for i, chunk in enumerate(corpus_chunks):
            if chunk.get("article") is not None:
                first_article_idx = i
                break

        # Extract preamble: all chunks before first article
        preamble_chunks = corpus_chunks[:first_article_idx] if first_article_idx is not None else corpus_chunks
        article_chunks = corpus_chunks[first_article_idx:] if first_article_idx is not None else []

        result_chunks = []

        # 1. Emit preamble (merge all pre-article chunks into one)
        if preamble_chunks:
            preamble_content = "\n".join(c["content"] for c in preamble_chunks if c.get("content"))
            if preamble_content.strip():
                pages = [c["page_start"] for c in preamble_chunks] + [c["page_end"] for c in preamble_chunks]
                result_chunks.append(make_chunk(
                    chapter=None,
                    chapter_title=None,
                    section=None,
                    section_title=None,
                    article=None,
                    article_title=None,
                    clause=None,
                    point=None,
                    content=preamble_content,
                    page_start=min(pages),
                    page_end=max(pages),
                    chunk_type="preamble",
                    document_id=self.document_id,
                ))

        # 2. Process article chunks - add IDs and parent links
        # Build clause_intro parent_id map for point linking
        clause_intro_map = {}  # (article, clause) -> chunk_id of clause_intro

        for chunk in article_chunks:
            # Generate stable chunk_id with document_id
            chunk_id = f"{self.document_id}_{_safe_id_part(chunk.get('article'))}_{_safe_id_part(chunk.get('clause'))}_{_safe_id_part(chunk.get('point'))}_{chunk['chunk_type']}"

            # Determine parent_id for points
            parent_id = None
            if chunk["chunk_type"] == "point" and chunk.get("clause") and chunk.get("article"):
                parent_key = (chunk["article"], chunk["clause"])
                parent_id = clause_intro_map.get(parent_key)

            # Store clause_intro chunk_id for point linking
            if chunk["chunk_type"] == "clause_intro" and chunk.get("clause") and chunk.get("article"):
                clause_intro_map[(chunk["article"], chunk["clause"])] = chunk_id

            # Build transformed chunk
            transformed = {
                **chunk,
                "chunk_id": chunk_id,
                "document_id": self.document_id,
                "parent_id": parent_id,
            }
            result_chunks.append(transformed)

        return result_chunks

    def chunk_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Parse PDF directly into chunks (uses full parse pipeline)."""
        from legal_rag.corpus.extractor import parse_corpus
        corpus_chunks = parse_corpus(pdf_path)
        return self.chunk(corpus_chunks)


# =========================================================
# BASELINE CHUNKERS (for comparison)
# =========================================================

class FixedSizeChunker(BaseChunker):
    """Fixed-size chunking with overlap (baseline CH01).

    Uses word-based chunking (split by whitespace), not token-based.
    chunk_size and overlap are in words.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        document_id: str = "labor_code_2019",
        preserve_metadata: bool = False,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.document_id = document_id
        self.preserve_metadata = preserve_metadata

    def chunk(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk from lines (joins all text first)."""
        text = "\n".join(line["text"] for line in lines)
        return self._chunk_text(text, lines if self.preserve_metadata else None)

    def chunk_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Parse PDF directly into chunks."""
        lines = extract_pdf_lines(pdf_path)
        text = "\n".join(line["text"] for line in lines)
        return self._chunk_text(text, lines if self.preserve_metadata else None)

    def _chunk_text(self, text: str, lines: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        words = text.split()
        chunks = []
        step = self.chunk_size - self.overlap
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            if not chunk_words:
                break
            chunk_text = " ".join(chunk_words)
            chunk_data = {
                "chunk_id": f"{self.document_id}_fixed_{i // step}",
                "document_id": self.document_id,
                "content": chunk_text,
                "chunk_type": "fixed_size",
                "chunk_index": i // step,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "unit": "words",
            }
            # Optionally add metadata from first line in chunk range
            if lines and self.preserve_metadata:
                # Find approximate page range for this chunk
                char_pos = len(" ".join(words[:i]))
                chunk_char_len = len(chunk_text)
                # Simple heuristic: assign page from first line
                chunk_data["page_start"] = lines[0]["page"] if lines else 1
                chunk_data["page_end"] = lines[-1]["page"] if lines else 1
            chunks.append(chunk_data)
        return chunks


class ParagraphChunker(BaseChunker):
    """Paragraph-based chunking (baseline CH02).

    Splits text by double newline. Note: input lines from PDF extraction
    are already split by single newlines, so paragraphs are reconstructed
    by joining with single newline then splitting on double newline.
    """

    def __init__(self, document_id: str = "labor_code_2019"):
        self.document_id = document_id

    def chunk(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        text = "\n".join(line["text"] for line in lines)
        return self._chunk_text(text, lines)

    def chunk_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        lines = extract_pdf_lines(pdf_path)
        text = "\n".join(line["text"] for line in lines)
        return self._chunk_text(text, lines)

    def _chunk_text(self, text: str, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        for i, para in enumerate(paragraphs):
            chunks.append({
                "chunk_id": f"{self.document_id}_para_{i}",
                "document_id": self.document_id,
                "content": para,
                "chunk_type": "paragraph",
                "paragraph_index": i,
                "page_start": lines[0]["page"] if lines else 1,
                "page_end": lines[-1]["page"] if lines else 1,
            })
        return chunks


class ArticleChunker(BaseChunker):
    """Article-level chunking - each article as one chunk (baseline CH03).

    Accepts both lines format (from PDF) and corpus format (from parser).
    """

    def __init__(self, document_id: str = "labor_code_2019"):
        self.document_id = document_id

    def chunk(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Detect format: corpus chunks have 'content' and 'article' fields
        if lines and "content" in lines[0] and "article" in lines[0]:
            return self._chunk_corpus(lines)
        return self._chunk_lines(lines)

    def _chunk_corpus(self, corpus_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group corpus chunks by article."""
        articles = {}
        for chunk in corpus_chunks:
            article = chunk.get("article")
            if not article:
                continue
            if article not in articles:
                articles[article] = {
                    "chunks": [],
                    "title": chunk.get("article_title", ""),
                    "pages": set(),
                    "chapter": chunk.get("chapter"),
                    "section": chunk.get("section"),
                }
            articles[article]["chunks"].append(chunk)
            articles[article]["pages"].add(chunk.get("page_start", 1))
            articles[article]["pages"].add(chunk.get("page_end", 1))

        chunks = []
        for i, (article, data) in enumerate(articles.items()):
            content = "\n".join(c["content"] for c in data["chunks"] if c.get("content"))
            pages = sorted(data["pages"])
            chunks.append({
                "chunk_id": f"{self.document_id}_article_{i}",
                "document_id": self.document_id,
                "chapter": data.get("chapter"),
                "section": data.get("section"),
                "article": article,
                "article_title": data["title"],
                "content": content,
                "chunk_type": "article",
                "page_start": min(pages) if pages else 1,
                "page_end": max(pages) if pages else 1,
            })
        return chunks

    def _chunk_lines(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Original lines-based implementation
        articles = {}
        current_article = None
        current_chapter = None
        current_section = None
        for line in lines:
            text = line["text"]
            page = line["page"]
            
            chapter_match = CHAPTER_RE.match(text)
            if chapter_match:
                current_chapter = f"Chương {chapter_match.group(1)}"
                continue
            
            section_match = SECTION_RE.match(text)
            if section_match:
                current_section = f"Mục {section_match.group(1)}"
                continue
            
            article_match = ARTICLE_RE.match(text)
            if article_match:
                current_article = f"Điều {article_match.group(1)}"
                if current_article not in articles:
                    articles[current_article] = {
                        "lines": [], 
                        "title": article_match.group(2).strip(), 
                        "pages": set(),
                        "chapter": current_chapter,
                        "section": current_section,
                    }
            if current_article:
                articles[current_article]["lines"].append(text)
                articles[current_article]["pages"].add(page)

        chunks = []
        for i, (article, data) in enumerate(articles.items()):
            content = "\n".join(data["lines"])
            pages = sorted(data["pages"])
            chunks.append({
                "chunk_id": f"{self.document_id}_article_{i}",
                "document_id": self.document_id,
                "chapter": data.get("chapter"),
                "section": data.get("section"),
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
    """Clause-level chunking - each clause as one chunk (baseline CH04).

    Accepts both lines format (from PDF) and corpus format (from parser).
    """

    def __init__(self, document_id: str = "labor_code_2019"):
        self.document_id = document_id

    def chunk(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Detect format: corpus chunks have 'content' and 'clause' fields
        if lines and "content" in lines[0] and "clause" in lines[0]:
            return self._chunk_corpus(lines)
        return self._chunk_lines(lines)

    def _chunk_corpus(self, corpus_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group corpus chunks by (article, clause)."""
        clauses = {}
        for chunk in corpus_chunks:
            article = chunk.get("article")
            clause = chunk.get("clause")
            if not article or clause is None:
                continue
            key = (article, clause)
            if key not in clauses:
                clauses[key] = {
                    "chunks": [],
                    "pages": set(),
                    "chapter": chunk.get("chapter"),
                    "section": chunk.get("section"),
                }
            clauses[key]["chunks"].append(chunk)
            clauses[key]["pages"].add(chunk.get("page_start", 1))
            clauses[key]["pages"].add(chunk.get("page_end", 1))

        chunks = []
        for i, ((article, clause), data) in enumerate(clauses.items()):
            content = "\n".join(c["content"] for c in data["chunks"] if c.get("content"))
            pages = sorted(data["pages"])
            chunks.append({
                "chunk_id": f"{self.document_id}_clause_{i}",
                "document_id": self.document_id,
                "chapter": data.get("chapter"),
                "section": data.get("section"),
                "article": article,
                "clause": clause,
                "content": content,
                "chunk_type": "clause",
                "page_start": min(pages) if pages else 1,
                "page_end": max(pages) if pages else 1,
            })
        return chunks

    def _chunk_lines(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        current_article = None
        current_clause = None
        current_chapter = None
        current_section = None
        clause_parts = []
        clause_pages = []
        clause_idx = 0

        for line in lines:
            text = line["text"]
            page = line["page"]

            chapter_match = CHAPTER_RE.match(text)
            if chapter_match:
                current_chapter = f"Chương {chapter_match.group(1)}"
                continue

            section_match = SECTION_RE.match(text)
            if section_match:
                current_section = f"Mục {section_match.group(1)}"
                continue

            article_match = ARTICLE_RE.match(text)
            if article_match:
                new_article = f"Điều {article_match.group(1)}"
                # Save current clause and article before resetting, to emit pending clause with correct article
                pending_clause = current_clause
                pending_article = current_article
                current_article = new_article
                current_clause = None
                if clause_parts and pending_article:
                    chunks.append({
                        "chunk_id": f"{self.document_id}_clause_{clause_idx}",
                        "document_id": self.document_id,
                        "chapter": current_chapter,
                        "section": current_section,
                        "article": pending_article,
                        "clause": pending_clause,
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
                        "chapter": current_chapter,
                        "section": current_section,
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
        if clause_parts and current_article:
            chunks.append({
                "chunk_id": f"{self.document_id}_clause_{clause_idx}",
                "document_id": self.document_id,
                "chapter": current_chapter,
                "section": current_section,
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