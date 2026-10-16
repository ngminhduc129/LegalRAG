import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

from legal_rag.parsing.classifier import (
    LineInfo, LineClassifier, CHAPTER_RE, SECTION_RE,
    ARTICLE_RE, CLAUSE_RE, POINT_RE,
    clean_line, is_page_number, extract_pdf_lines,
    is_legal_reference_continuation, is_article_intro_start,
    extract_article_number, match_article_heading,
    is_noise_line, is_signature_start,
)

from legal_rag.parsing.constants import DOCUMENT_NAME, DOCUMENT_NUMBER


def is_ref_like_title(title: str) -> bool:
    """True if a heading title is actually a cross-reference fragment.

    Compares whole words (not prefixes): "Làm thêm giờ" must NOT match
    the starter "là".
    """
    cleaned = title.strip().lstrip("\"'“”‘’([{").lower()
    words = cleaned.split()
    if not words:
        return False
    if words[0] in ("của", "tại", "theo", "và", "hoặc", "trong", "như", "là"):
        return True
    return " ".join(words[:2]) in ("quy định", "đối với", "căn cứ")


@dataclass
class ParserState:
    chapter: Optional[str] = None
    chapter_title: Optional[str] = None
    section: Optional[str] = None
    section_title: Optional[str] = None
    article: Optional[str] = None
    article_title: Optional[str] = None
    article_page: Optional[int] = None

    clause: Optional[str] = None
    clause_parts: List[str] = field(default_factory=list)
    clause_pages: List[int] = field(default_factory=list)
    clause_intro: List[str] = field(default_factory=list)
    clause_intro_pages: List[int] = field(default_factory=list)
    clause_has_points: bool = False

    point: Optional[str] = None
    point_parts: List[str] = field(default_factory=list)
    point_pages: List[int] = field(default_factory=list)

    article_intro: List[str] = field(default_factory=list)
    article_intro_pages: List[int] = field(default_factory=list)

    def reset_article(self):
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

    def reset_clause(self):
        self.clause = None
        self.clause_parts = []
        self.clause_pages = []
        self.clause_intro = []
        self.clause_intro_pages = []
        self.clause_has_points = False
        self.point = None
        self.point_parts = []
        self.point_pages = []

    def reset_point(self):
        self.point = None
        self.point_parts = []
        self.point_pages = []


def join_text(parts: List[str]) -> str:
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
    chunk_type: str
) -> Dict[str, Any]:
    return {
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


class LegalStructureParser:
    def __init__(self):
        self.classifier = LineClassifier()
        self.state = ParserState()
        self.chunks = []
        self.warnings = []
        self.seen_articles = {}
        self.dropped_lines = 0

    def parse(self, lines: List[dict]) -> List[Dict[str, Any]]:
        self.state = ParserState()
        self.chunks = []
        self.warnings = []
        self.dropped_lines = 0
        self.seen_articles = {}

        kept = []
        signature_seen = False
        for line_data in lines:
            text = line_data.get("text", "")
            if is_noise_line(text):
                self.dropped_lines += 1
                continue
            if signature_seen or is_signature_start(text):
                signature_seen = True
                self.dropped_lines += 1
                continue
            kept.append(line_data)
        self.warnings.append(f"Dropped {self.dropped_lines} noise/signature lines")

        classified = self.classifier.classify_all(kept)

        i = 0
        while i < len(classified):
            line_info = classified[i]
            text = line_info.text
            page = line_info.page

            new_i = i

            if line_info.is_structural_candidate:
                if line_info.candidate_type == "chapter":
                    new_i = self._handle_chapter(line_info, i, classified)
                elif line_info.candidate_type == "section":
                    new_i = self._handle_section(line_info, i, classified)
                elif line_info.candidate_type == "article":
                    new_i = self._handle_article(line_info, i, classified)
                elif line_info.candidate_type == "clause":
                    new_i = self._handle_clause(line_info, i, classified)
                elif line_info.candidate_type == "point":
                    new_i = self._handle_point(line_info, i, classified)

            if new_i == i:
                self._handle_content(line_info)
                new_i = i + 1
            else:
                # Skip the lines that were consumed by the handler
                pass

            i = new_i

        self._close_all()
        return self.chunks

    def _handle_chapter(self, line_info: LineInfo, i: int, classified: List[LineInfo]) -> int:
        match = CHAPTER_RE.match(line_info.text)
        if not match:
            return i

        self._close_all()

        self.state.chapter = f"Chương {match.group(1)}"
        self.state.chapter_title = None

        new_i = i + 1
        if new_i < len(classified):
            next_text = classified[new_i].text
            if not any(p.match(next_text) for p in [CHAPTER_RE, SECTION_RE, ARTICLE_RE, CLAUSE_RE]):
                self.state.chapter_title = next_text
                new_i += 1

        return new_i

    def _handle_section(self, line_info: LineInfo, i: int, classified: List[LineInfo]) -> int:
        match = SECTION_RE.match(line_info.text)
        if not match:
            return i

        self._close_article()

        self.state.section = f"Mục {match.group(1)}"
        self.state.section_title = None

        new_i = i + 1
        if new_i < len(classified):
            next_text = classified[new_i].text
            if not any(p.match(next_text) for p in [CHAPTER_RE, SECTION_RE, ARTICLE_RE, CLAUSE_RE]):
                self.state.section_title = next_text
                new_i += 1

        return new_i

    def _should_start_new_article(self, line_info: LineInfo, i: int, classified: List[LineInfo]) -> bool:
        if not line_info.is_structural_candidate or line_info.candidate_type != "article":
            return False

        features = line_info.features

        # Only reject if there's STRONG evidence it's a legal reference
        # Legal refs typically: continue previous sentence, or follow "theo quy định tại"
        if features.get("preceded_by_legal_ref"):
            return False

        # If previous line doesn't end with period, it might be a continuation
        # But only reject if previous line looks like a legal ref prefix
        if features.get("prev_line_incomplete"):
            # Additional check: is previous line a clause/point? If so, this is likely a ref
            if i > 0:
                prev = classified[i - 1]
                if prev.is_structural_candidate and prev.candidate_type in ("clause", "point"):
                    return False

        # Strong positive signals
        if features.get("next_line_is_clause"):
            return True

        if features.get("has_title") and features.get("title_length", 0) > 3:
            return True

        if features.get("followed_by_paragraph"):
            return True

        # Default: if it matches article pattern, treat as article
        return True

    def _handle_article(self, line_info: LineInfo, i: int, classified: List[LineInfo]) -> int:
        parsed = match_article_heading(line_info.text)
        if parsed is None:
            return i
        article_number, article_title_text = parsed[0], parsed[1]

        # Cross-reference fragments (e.g. "của Bộ luật này;") are content,
        # not new articles. Returning i routes the line to _handle_content.
        if is_ref_like_title(article_title_text):
            return i

        # Article numbers are unique: any repeat is a page-header reprint
        # (drop the line) or an in-text reference (keep as content).
        if article_number in self.seen_articles:
            if self.state.article == f"Điều {article_number}":
                return i + 1
            return i

        if not self._should_start_new_article(line_info, i, classified):
            return i

        candidate_article = f"Điều {article_number}"

        same_article = (
            self.state.article is not None
            and self.state.article == candidate_article
            and (
                self.state.article_title is None
                or not article_title_text
                or (
                    self.state.article_title.strip().casefold()
                    == article_title_text.strip().casefold()
                )
            )
        )

        if same_article:
            return i + 1

        self._close_article()

        self.state.article = candidate_article
        self.state.article_title = article_title_text or None
        self.state.article_page = line_info.page
        self.seen_articles[article_number] = article_title_text

        new_i = self._collect_multiline_article_title(i, classified)

        return new_i

    def _is_title_continuation(self, text: str) -> bool:
        """True if a line right after a heading continues the title.

        Titles wrap mid-phrase, so a continuation starts lowercase or with
        a digit (e.g. "15 tuổi đến chưa đủ 18 tuổi" completing "...từ đủ").
        A genuine intro/content sentence starts uppercase or with a quote —
        never lowercase. (Digit-start structural lines like "1. ..." are
        filtered by stop_patterns before this check runs.)
        """
        stripped = text.strip().lstrip("\"'“”‘’([{")
        if not stripped:
            return False
        if len(stripped) > 120:
            return False
        return stripped[0].islower() or stripped[0].isdigit()

    def _collect_multiline_article_title(self, i: int, classified: List[LineInfo]) -> int:
        if not self.state.article_title:
            return i + 1

        stop_patterns = [CHAPTER_RE, SECTION_RE, ARTICLE_RE, CLAUSE_RE, POINT_RE]
        max_title_len = 200
        max_title_lines = 5

        j = i + 1
        title_lines = 1
        while j < len(classified) and title_lines < max_title_lines:
            next_text = classified[j].text

            if any(p.match(next_text) for p in stop_patterns):
                break

            if is_article_intro_start(next_text):
                break

            if not self._is_title_continuation(next_text):
                break

            if len(self.state.article_title) > max_title_len:
                break

            self.state.article_title += " " + next_text
            j += 1
            title_lines += 1

        return j

    def _should_start_new_clause(self, line_info: LineInfo) -> bool:
        if not line_info.is_structural_candidate or line_info.candidate_type != "clause":
            return False

        features = line_info.features

        if features.get("is_number_only") or features.get("looks_like_money_date"):
            return False

        if features.get("preceded_by_article_intro"):
            return True

        return True

    def _handle_clause(self, line_info: LineInfo, i: int, classified: List[LineInfo]) -> int:
        match = CLAUSE_RE.match(line_info.text)
        if not match:
            return i

        if not self._should_start_new_clause(line_info):
            return i

        if self.state.article is None:
            return i

        # Emit article_intro if this is the first clause of the article
        if self.state.article_intro and self.state.clause is None:
            self._emit_article_intro()

        self._emit_point()
        self._emit_clause()

        self.state.clause = match.group(1)
        self.state.clause_parts = [match.group(2)]
        self.state.clause_pages = [line_info.page]
        self.state.clause_has_points = False

        return i + 1

    def _handle_point(self, line_info: LineInfo, i: int, classified: List[LineInfo]) -> int:
        match = POINT_RE.match(line_info.text)
        if not match:
            return i

        if self.state.clause is None:
            return i

        if not self.state.clause_has_points:
            self.state.clause_intro = list(self.state.clause_parts)
            self.state.clause_intro_pages = list(self.state.clause_pages)
            self.state.clause_parts = []
            self.state.clause_pages = []
            self.state.clause_has_points = True

        self._emit_point()

        self.state.point = match.group(1).lower()
        self.state.point_parts = [match.group(2)]
        self.state.point_pages = [line_info.page]

        return i + 1

    def _handle_content(self, line_info: LineInfo):
        text = line_info.text
        page = line_info.page

        if self.state.point is not None:
            self.state.point_parts.append(text)
            self.state.point_pages.append(page)
        elif self.state.clause is not None:
            self.state.clause_parts.append(text)
            self.state.clause_pages.append(page)
        else:
            self.state.article_intro.append(text)
            self.state.article_intro_pages.append(page)

    def _emit_article_intro(self):
        content = join_text(self.state.article_intro)
        if content and self.state.article is not None:
            pages = self.state.article_intro_pages
            self.chunks.append(make_chunk(
                chapter=self.state.chapter,
                chapter_title=self.state.chapter_title,
                section=self.state.section,
                section_title=self.state.section_title,
                article=self.state.article,
                article_title=self.state.article_title,
                clause=None,
                point=None,
                content=content,
                page_start=min(pages),
                page_end=max(pages),
                chunk_type="article_intro"
            ))
        self.state.article_intro = []
        self.state.article_intro_pages = []

    def _emit_clause_intro(self):
        content = join_text(self.state.clause_intro)
        if content and self.state.article is not None:
            pages = self.state.clause_intro_pages
            self.chunks.append(make_chunk(
                chapter=self.state.chapter,
                chapter_title=self.state.chapter_title,
                section=self.state.section,
                section_title=self.state.section_title,
                article=self.state.article,
                article_title=self.state.article_title,
                clause=self.state.clause,
                point=None,
                content=content,
                page_start=min(pages),
                page_end=max(pages),
                chunk_type="clause_intro"
            ))
        self.state.clause_intro = []
        self.state.clause_intro_pages = []

    def _emit_clause(self):
        if self.state.clause is not None and not self.state.clause_has_points:
            content = join_text(self.state.clause_parts)
            if content:
                pages = self.state.clause_pages
                self.chunks.append(make_chunk(
                    chapter=self.state.chapter,
                    chapter_title=self.state.chapter_title,
                    section=self.state.section,
                    section_title=self.state.section_title,
                    article=self.state.article,
                    article_title=self.state.article_title,
                    clause=self.state.clause,
                    point=None,
                    content=content,
                    page_start=min(pages),
                    page_end=max(pages),
                    chunk_type="clause"
                ))
        elif self.state.clause is not None and self.state.clause_has_points:
            self._emit_clause_intro()

        self.state.reset_clause()

    def _emit_point(self):
        if self.state.point is not None:
            content = join_text(self.state.point_parts)
            if content and self.state.article is not None:
                pages = self.state.point_pages
                self.chunks.append(make_chunk(
                    chapter=self.state.chapter,
                    chapter_title=self.state.chapter_title,
                    section=self.state.section,
                    section_title=self.state.section_title,
                    article=self.state.article,
                    article_title=self.state.article_title,
                    clause=self.state.clause,
                    point=self.state.point,
                    content=content,
                    page_start=min(pages),
                    page_end=max(pages),
                    chunk_type="point"
                ))
        self.state.reset_point()

    def _close_article(self):
        self._emit_point()
        self._emit_clause()
        self._emit_article_intro()
        self.state.reset_article()

    def _close_all(self):
        self._close_article()


def parse_legal_structure(lines: List[dict]) -> List[Dict[str, Any]]:
    parser = LegalStructureParser()
    return parser.parse(lines)