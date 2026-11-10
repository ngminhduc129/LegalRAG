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
    clean_line, is_page_number,
)
from legal_rag.corpus.extractor import extract_pdf_lines
from legal_rag.corpus.registry import DocumentMeta, resolve_document
from legal_rag.chunking.base import BaseChunker
from legal_rag.chunking.identity import (
    StructureRegistry,
    IdentityResolver,
)


STRUCTURE_FIELDS = (
    "chapter",
    "chapter_title",
    "section",
    "section_title",
    "article",
    "article_title",
    "clause",
    "point",
)


def _source_reference(source: Dict[str, Any], document: DocumentMeta) -> Dict[str, Any]:
    reference = {
        "document_id": source.get("document_id", document.document_id),
        "document_name": source.get("document_name", document.document_name),
        "document_number": source.get("document_number", document.document_number),
        "document_type": source.get("document_type", document.doc_type),
        "status": source.get("status", document.status),
        **{field: source.get(field) for field in STRUCTURE_FIELDS},
        "chunk_type": source.get("chunk_type"),
        "page_start": source.get("page_start", source.get("page", 1)),
        "page_end": source.get("page_end", source.get("page", 1)),
    }
    if source.get("chunk_id"):
        reference["source_chunk_id"] = source["chunk_id"]
    return reference


def _aggregate_source_metadata(
    sources: List[Dict[str, Any]], document: DocumentMeta
) -> Dict[str, Any]:
    """Keep source hierarchy and page locations for chunks spanning parser records."""
    references = []
    seen = set()
    for source in sources:
        reference = _source_reference(source, document)
        key = tuple(reference.items())
        if key not in seen:
            seen.add(key)
            references.append(reference)

    metadata = {
        **document.as_metadata(),
        "document_id": document.document_id,
        "document_name": next(
            (reference["document_name"] for reference in references if reference["document_name"]),
            document.document_name,
        ),
        "document_number": next(
            (reference["document_number"] for reference in references if reference["document_number"]),
            document.document_number,
        ),
        "source_refs": references,
    }
    for field in STRUCTURE_FIELDS:
        values = list(dict.fromkeys(
            reference[field]
            for reference in references
            if reference.get(field) is not None
        ))
        metadata[field] = values[0] if len(values) == 1 else None
        if field in {"chapter", "section", "article", "article_title", "clause", "point"}:
            plural_field = {
                "chapter": "chapters",
                "section": "sections",
                "article": "articles",
                "article_title": "article_titles",
                "clause": "clauses",
                "point": "points",
            }[field]
            metadata[plural_field] = values

    pages = [
        page
        for reference in references
        for page in (reference["page_start"], reference["page_end"])
        if isinstance(page, int) and page > 0
    ]
    metadata["page_start"] = min(pages) if pages else 1
    metadata["page_end"] = max(pages) if pages else metadata["page_start"]
    return metadata


# =========================================================
# CHUNK CREATION
# =========================================================

def join_text(parts: List[str]) -> str:
    """Join text parts into chunk content."""
    return "\n".join(p.strip() for p in parts if p and p.strip()).strip()


def _safe_id_part(value) -> str:
    """Convert a value to a safe ID part, handling None."""
    if value is None:
        return "none"
    return str(value).replace(" ", "_").replace("/", "_").lower()


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
    document: Optional[DocumentMeta] = None,
    document_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    structure_instance_id: Optional[str] = None,
    parent_instance_id: Optional[str] = None,
    structure_role: Optional[str] = None,
    operation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a standardized chunk record."""
    doc = resolve_document(document or document_id)
    chunk_id = f"{doc.document_id}_{_safe_id_part(article)}_{_safe_id_part(clause)}_{_safe_id_part(point)}_{chunk_type}"
    record: Dict[str, Any] = {
        **doc.as_metadata(),
        "chunk_id": chunk_id,
        "document_id": doc.document_id,
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
        "structure_occurrence": 1,
    }
    if structure_instance_id is not None:
        record["structure_instance_id"] = structure_instance_id
    if parent_instance_id is not None:
        record["parent_instance_id"] = parent_instance_id
    if structure_role is not None:
        record["structure_role"] = structure_role
    if operation_id is not None:
        record["operation_id"] = operation_id
    return record


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

    def __init__(self, document_id: DocumentMeta | str = "unknown"):
        self.document = resolve_document(document_id)
        self.document_id = self.document.document_id

    def chunk(self, corpus_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform parser output chunks into final chunks with IDs and parent links.

        Parser-generated structure identity
        (``structure_instance_id``, ``parent_instance_id``, ``structure_role``,
        ``operation_id``) is the authoritative source for legal hierarchy.
        Chunk-level ``parent_id`` is resolved separately from
        ``parent_instance_id`` and is never conflated with it.

        Args:
            corpus_chunks: List of chunks from LegalStructureParser with fields:
                document_name, document_number, chapter, chapter_title,
                section, section_title, article, article_title, clause, point,
                chunk_type, content, page_start, page_end,
                structure_instance_id, parent_instance_id, structure_role,
                operation_id

        Returns:
            List of transformed chunks with added chunk_id, document_id,
            parent_id and preserved structure identity metadata.
        """
        if not corpus_chunks:
            return []

        # Appendix chunks are standalone: they must never be merged into the
        # preamble, and an appendix-only document (no article at all) must
        # still produce output.
        appendix_chunks = [
            chunk for chunk in corpus_chunks
            if chunk.get("chunk_type") == "appendix"
        ]
        body_chunks = [
            chunk for chunk in corpus_chunks
            if chunk.get("chunk_type") != "appendix"
        ]

        # Pass 1: register every structure in the registry and generate stable
        # chunk_ids.  Registering all structures first allows parent resolution
        # even when a parent structure appears later in the input.
        registry = StructureRegistry()
        chunk_ids_by_index: Dict[int, str] = {}
        seen_ids: set[str] = set()
        # Legacy fallback: chunks without parser instance IDs (legacy parser
        # output) use (article, clause) string keys for parent resolution.
        # This is explicitly a fallback, not authoritative parser identity.
        legacy_clause_intro_map: Dict[tuple[str, str], str] = {}
        for index, chunk in enumerate(body_chunks):
            # Ensure document_id is set for parent resolution.
            if not chunk.get("document_id"):
                chunk["document_id"] = self.document_id
            chunk_type = chunk.get("chunk_type", "unknown")
            instance_id = chunk.get("structure_instance_id")
            article = chunk.get("article")
            clause = chunk.get("clause")

            chunk_id = (
                f"{self.document_id}_"
                f"{_safe_id_part(article)}_"
                f"{_safe_id_part(clause)}_"
                f"{_safe_id_part(chunk.get('point'))}_"
                f"{chunk_type}"
            )
            occurrence = 1
            unique_id = chunk_id
            while unique_id in seen_ids:
                occurrence += 1
                unique_id = f"{chunk_id}__occ{occurrence}"
            seen_ids.add(unique_id)
            chunk_ids_by_index[index] = unique_id
            chunk["chunk_id"] = unique_id
            chunk["structure_occurrence"] = occurrence

            if instance_id:
                registry.register(
                    instance_id=instance_id,
                    parent_instance_id=chunk.get("parent_instance_id"),
                    structure_role=chunk.get("structure_role", "main"),
                    operation_id=chunk.get("operation_id"),
                )
                registry.add_chunk_id(instance_id, unique_id)
            elif chunk_type == "clause_intro" and article and clause:
                legacy_clause_intro_map.setdefault((article, clause), unique_id)

        # Find first article chunk to identify preamble
        first_article_idx = None
        for i, chunk in enumerate(body_chunks):
            if chunk.get("article") is not None:
                first_article_idx = i
                break

        # Extract preamble: all non-appendix chunks before the first article.
        # If there is no article at all, everything non-appendix is preamble.
        preamble_chunks = (
            body_chunks[:first_article_idx]
            if first_article_idx is not None
            else body_chunks
        )
        article_chunks = (
            body_chunks[first_article_idx:] if first_article_idx is not None else []
        )

        result_chunks: List[Dict[str, Any]] = []

        # 1. Emit preamble (merge all pre-article chunks into one).
        # A merged preamble has no single structure_instance_id; preserve
        # structure_role only when all preamble chunks agree on it.
        if preamble_chunks:
            preamble_content = "\n".join(
                c["content"] for c in preamble_chunks if c.get("content")
            )
            if preamble_content.strip():
                pages = [
                    c["page_start"] for c in preamble_chunks
                ] + [c["page_end"] for c in preamble_chunks]
                roles = {
                    c.get("structure_role") for c in preamble_chunks
                    if c.get("structure_role") is not None
                }
                preamble_role = roles.pop() if len(roles) == 1 else None
                preamble_chunk = make_chunk(
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
                    document=self.document,
                    structure_role=preamble_role,
                )
                # If the merged preamble's chunk_id collides with one already
                # seen in pass 1 (e.g. a preamble-type input chunk), suffix it.
                if preamble_chunk["chunk_id"] in seen_ids:
                    occurrence = 1
                    base_id = preamble_chunk["chunk_id"]
                    while preamble_chunk["chunk_id"] in seen_ids:
                        occurrence += 1
                        preamble_chunk["chunk_id"] = f"{base_id}__occ{occurrence}"
                seen_ids.add(preamble_chunk["chunk_id"])
                result_chunks.append(preamble_chunk)

        # 2. Process article chunks - resolve parent links from parser identity.
        resolver = IdentityResolver(registry)
        for index, chunk in enumerate(article_chunks):
            chunk_type = chunk.get("chunk_type", "unknown")
            chunk_id = chunk_ids_by_index[index]
            instance_id = chunk.get("structure_instance_id")

            # Resolve chunk-level parent_id from parent_instance_id, never from
            # (article, clause) tuples.  parent_instance_id itself is preserved
            # unchanged as legal structure metadata.
            parent_id: Optional[str] = None
            if instance_id and instance_id in registry.records:
                parent_id = resolver.resolve_parent_chunk_id(
                    instance_id,
                    chunk_type,
                    self.document_id,
                    body_chunks,
                )
            elif chunk_type == "point" and article and clause:
                # Legacy fallback: resolve point's parent via (article, clause)
                # string key.  Marked explicitly as fallback, not parser identity.
                parent_id = legacy_clause_intro_map.get((article, clause))

            # Build transformed chunk, filling any missing document metadata
            # from the registry entry instead of hard-coded constants.
            transformed = {**chunk}
            for key, value in self.document.as_metadata().items():
                if not transformed.get(key):
                    transformed[key] = value
            transformed.update({
                "chunk_id": chunk_id,
                "document_id": self.document_id,
                "parent_id": parent_id,
                "structure_occurrence": chunk.get("structure_occurrence", 1),
                # structure_instance_id, parent_instance_id, structure_role and
                # operation_id are preserved from the parser input via {**chunk}.
            })
            result_chunks.append(transformed)

        # 3. Emit appendix chunks unchanged (no parent links, own chunk ids).
        appendix_seen_ids = set()
        for i, chunk in enumerate(appendix_chunks):
            appendix_id = (
                f"{self.document_id}_appendix_"
                f"{_safe_id_part(chunk.get('appendix_title') or i + 1)}"
            )
            occurrence = 1
            unique_id = appendix_id
            while unique_id in appendix_seen_ids:
                occurrence += 1
                unique_id = f"{appendix_id}__occ{occurrence}"
            appendix_seen_ids.add(unique_id)
            transformed = {**chunk}
            for key, value in self.document.as_metadata().items():
                if not transformed.get(key):
                    transformed[key] = value
            transformed.update({
                "chunk_id": unique_id,
                "document_id": self.document_id,
                "parent_id": None,
                "structure_occurrence": occurrence,
            })
            result_chunks.append(transformed)

        return result_chunks

    def chunk_from_pdf(
        self, pdf_path: Path, document: DocumentMeta | str | None = None
    ) -> List[Dict[str, Any]]:
        """Parse PDF directly into chunks (uses full parse pipeline)."""
        from legal_rag.corpus.extractor import parse_corpus
        corpus_chunks = parse_corpus(pdf_path, document or self.document)
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
        document_id: DocumentMeta | str = "unknown",
        preserve_metadata: bool = False,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.document = resolve_document(document_id)
        self.document_id = self.document.document_id
        self.preserve_metadata = preserve_metadata

    def chunk(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk parser records or lines while retaining their source metadata."""
        if lines and "content" in lines[0]:
            return self._chunk_corpus(lines)
        return self._chunk_lines(lines)

    def chunk_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Parse PDF directly into chunks."""
        lines = extract_pdf_lines(pdf_path)
        return self._chunk_lines(lines)

    def _chunk_corpus(self, corpus_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        words = [
            (word, source)
            for source in corpus_chunks
            for word in source.get("content", "").split()
        ]
        return self._chunk_words(words)

    def _chunk_lines(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        words = [
            (word, line)
            for line in lines
            for word in line.get("text", "").split()
        ]
        return self._chunk_words(words)

    def _chunk_words(self, words: List[tuple]) -> List[Dict[str, Any]]:
        chunks = []
        step = self.chunk_size - self.overlap
        for i in range(0, len(words), step):
            window = words[i:i + self.chunk_size]
            if not window:
                break
            chunk_text = " ".join(word for word, _ in window)
            metadata = _aggregate_source_metadata(
                [source for _, source in window],
                self.document,
            )
            chunk_data = {
                **metadata,
                "chunk_id": f"{self.document_id}_fixed_{self.chunk_size}_{i // step}",
                "content": chunk_text,
                "chunk_type": "fixed_size",
                "chunk_index": i // step,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "unit": "words",
            }
            chunks.append(chunk_data)
        return chunks


class ParagraphChunker(BaseChunker):
    """Paragraph-based chunking (baseline CH02).

    Groups consecutive PDF lines into paragraphs: a paragraph boundary is
    placed after any line ending with sentence-terminal punctuation
    (., ;, :, ?, !, possibly followed by a closing quote). PDF extraction
    provides no blank lines, so splitting on double newline (the previous
    implementation) always yielded a single chunk for the whole document.
    """

    # A line ending with one of these (after stripping closing quotes/
    # brackets) terminates the current paragraph.
    TERMINAL_ENDINGS = (".", ";", ":", "?", "!", "…")
    TRAILING_CLOSERS = "\"'”’»)]}"

    def __init__(self, document_id: DocumentMeta | str = "unknown"):
        self.document = resolve_document(document_id)
        self.document_id = self.document.document_id

    def chunk(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if lines and "content" in lines[0]:
            text_lines = [
                (text, source.get("page_start", 1), source)
                for source in lines
                for text in source.get("content", "").splitlines()
                if text.strip()
            ]
            return self._chunk_lines(text_lines)

        text_lines = [
            (line["text"], line.get("page", 1), line)
            for line in lines
            if line.get("text") and line["text"].strip()
        ]
        return self._chunk_lines(text_lines)

    def chunk_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        lines = extract_pdf_lines(pdf_path)
        text_lines = [
            (line["text"], line.get("page", 1), line)
            for line in lines
            if line.get("text") and line["text"].strip()
        ]
        return self._chunk_lines(text_lines)

    def _ends_paragraph(self, line: str) -> bool:
        stripped = line.strip().rstrip(self.TRAILING_CLOSERS).strip()
        return bool(stripped) and stripped[-1] in self.TERMINAL_ENDINGS

    def _chunk_lines(self, text_lines: List[tuple]) -> List[Dict[str, Any]]:
        chunks = []
        buf: List[str] = []
        pages: List[int] = []
        sources = []
        for text, page, source in text_lines:
            buf.append(text.strip())
            pages.append(page)
            sources.append(source)
            if self._ends_paragraph(text):
                chunks.append(self._make_chunk(chunks, buf, pages, sources))
                buf = []
                pages = []
                sources = []
        if buf:
            chunks.append(self._make_chunk(chunks, buf, pages, sources))
        # Re-number sequentially
        for i, chunk in enumerate(chunks):
            chunk["chunk_id"] = f"{self.document_id}_para_{i}"
            chunk["paragraph_index"] = i
        return chunks

    def _make_chunk(
        self,
        chunks: List[Dict[str, Any]],
        buf: List[str],
        pages: List[int],
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            **_aggregate_source_metadata(sources, self.document),
            "chunk_id": "",
            "content": "\n".join(buf),
            "chunk_type": "paragraph",
            "paragraph_index": len(chunks),
            "page_start": min(pages) if pages else 1,
            "page_end": max(pages) if pages else 1,
        }

    def _chunk_text(self, text: str, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Kept for backward compatibility; delegates to line-based chunking.
        text_lines = [
            (line["text"], line.get("page", 1), line)
            for line in (lines or [])
            if line.get("text") and line["text"].strip()
        ]
        return self._chunk_lines(text_lines)


class ArticleChunker(BaseChunker):
    """Article-level chunking - each article as one chunk (baseline CH03).

    Accepts both lines format (from PDF) and corpus format (from parser).
    """

    def __init__(self, document_id: DocumentMeta | str = "unknown"):
        self.document = resolve_document(document_id)
        self.document_id = self.document.document_id

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
                }
            articles[article]["chunks"].append(chunk)

        chunks = []
        for i, (article, data) in enumerate(articles.items()):
            content = "\n".join(c["content"] for c in data["chunks"] if c.get("content"))
            chunks.append({
                **_aggregate_source_metadata(data["chunks"], self.document),
                "chunk_id": f"{self.document_id}_article_{i}",
                "content": content,
                "chunk_type": "article",
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
                        "sources": [],
                        "chapter": current_chapter,
                        "section": current_section,
                    }
            if current_article:
                articles[current_article]["lines"].append(text)
                articles[current_article]["sources"].append({
                    **line,
                    "page_start": page,
                    "page_end": page,
                    "chapter": articles[current_article]["chapter"],
                    "section": articles[current_article]["section"],
                    "article": current_article,
                    "article_title": articles[current_article]["title"],
                })

        chunks = []
        for i, (article, data) in enumerate(articles.items()):
            content = "\n".join(data["lines"])
            chunks.append({
                **_aggregate_source_metadata(data["sources"], self.document),
                "chunk_id": f"{self.document_id}_article_{i}",
                "content": content,
                "chunk_type": "article",
            })
        return chunks

    def chunk_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        lines = extract_pdf_lines(pdf_path)
        return self.chunk(lines)


class ClauseChunker(BaseChunker):
    """Clause-level chunking - each clause as one chunk (baseline CH04).

    Accepts both lines format (from PDF) and corpus format (from parser).
    """

    def __init__(self, document_id: DocumentMeta | str = "unknown"):
        self.document = resolve_document(document_id)
        self.document_id = self.document.document_id

    def chunk(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Detect format: corpus chunks have 'content' and 'clause' fields
        if lines and "content" in lines[0] and "clause" in lines[0]:
            return self._chunk_corpus(lines)
        return self._chunk_lines(lines)

    def _chunk_corpus(self, corpus_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group corpus chunks by (article, clause)."""
        clauses = {}
        article_intros = []
        for chunk in corpus_chunks:
            article = chunk.get("article")
            clause = chunk.get("clause")
            if article and chunk.get("chunk_type") == "article_intro":
                article_intros.append(chunk)
            if not article or clause is None:
                continue
            key = (article, clause)
            if key not in clauses:
                clauses[key] = {
                    "chunks": [],
                }
            clauses[key]["chunks"].append(chunk)

        chunks = [
            {
                **_aggregate_source_metadata([source], self.document),
                "chunk_id": f"{self.document_id}_article_intro_{i}",
                "content": source.get("content", ""),
                "chunk_type": "article_intro",
            }
            for i, source in enumerate(article_intros)
        ]
        for i, ((article, clause), data) in enumerate(clauses.items()):
            content = "\n".join(c["content"] for c in data["chunks"] if c.get("content"))
            chunks.append({
                **_aggregate_source_metadata(data["chunks"], self.document),
                "chunk_id": f"{self.document_id}_clause_{i}",
                "content": content,
                "chunk_type": "clause",
            })
        return chunks

    def _chunk_lines(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        current_article = None
        current_clause = None
        current_chapter = None
        current_section = None
        clause_parts = []
        clause_sources = []
        clause_idx = 0
        current_point = None

        def finish_clause(article: str, chapter: Optional[str], section: Optional[str]):
            nonlocal clause_idx, clause_parts, clause_sources
            metadata = _aggregate_source_metadata(clause_sources, self.document)
            metadata.update({
                "chapter": chapter,
                "section": section,
                "article": article,
                "clause": current_clause,
            })
            chunks.append({
                **metadata,
                "chunk_id": f"{self.document_id}_clause_{clause_idx}",
                "content": "\n".join(clause_parts),
                "chunk_type": "clause",
            })
            clause_idx += 1
            clause_parts = []
            clause_sources = []

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
                pending_chapter = current_chapter
                pending_section = current_section
                current_article = new_article
                current_clause = None
                current_point = None
                if clause_parts and pending_article:
                    current_clause = pending_clause
                    finish_clause(pending_article, pending_chapter, pending_section)
                    current_clause = None
                continue

            clause_match = CLAUSE_RE.match(text)
            if clause_match and current_article:
                if clause_parts:
                    finish_clause(current_article, current_chapter, current_section)
                current_clause = clause_match.group(1)
                clause_parts = [clause_match.group(2)]
                current_point = None
                clause_sources = [{
                    **line,
                    "page_start": page,
                    "page_end": page,
                    "chapter": current_chapter,
                    "section": current_section,
                    "article": current_article,
                    "clause": current_clause,
                }]
                continue

            if current_clause:
                clause_parts.append(text)
                point_match = POINT_RE.match(text)
                if point_match:
                    current_point = point_match.group(1)
                clause_sources.append({
                    **line,
                    "page_start": page,
                    "page_end": page,
                    "chapter": current_chapter,
                    "section": current_section,
                    "article": current_article,
                    "clause": current_clause,
                    "point": current_point,
                })

        # Last clause
        if clause_parts and current_article:
            finish_clause(current_article, current_chapter, current_section)

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