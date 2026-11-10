import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from legal_rag.corpus.extractor import extract_pdf_lines
from legal_rag.corpus.registry import DocumentMeta, resolve_document
from legal_rag.parsing.classifier import (
    LineInfo, LineClassifier, CHAPTER_RE, SECTION_RE,
    ARTICLE_RE, CLAUSE_RE, POINT_RE,
    clean_line, is_page_number,
    is_legal_reference_continuation, is_article_intro_start,
    extract_article_number, match_article_heading,
    is_noise_line, is_signature_start, is_signature_title,
    has_signature_title_anywhere,
    is_appendix_start, is_appendix_heading, is_signature_end, VBHN_SIGNATURE_RE,
    is_likely_legal_reference_in_content,
)


def is_ref_like_title(title: str) -> bool:
    """True if a heading title is actually a cross-reference fragment.

    Compares whole words (not prefixes): "Làm thêm giờ" must NOT match
    the starter "là".  "Quy định chuyển tiếp" is a real article title, so
    "quy định" only counts as a reference when it points somewhere
    ("Quy định tại Điều 5", "Quy định của Bộ luật này").
    """
    cleaned = title.strip().lstrip("\"'“”‘’([{").lower()
    words = cleaned.split()
    if not words:
        return False
    if words[0] in ("của", "tại", "theo", "và", "hoặc", "trong", "như", "là"):
        return True
    first_two = " ".join(words[:2])
    if first_two in ("quy định", "đối với", "căn cứ"):
        if first_two != "quy định":
            return True
        rest = words[2:4]
        return any(
            word in ("tại", "của", "này", "theo", "điều", "khoản", "mục", "chương")
            for word in rest
        )
    return False


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

    preamble: List[str] = field(default_factory=list)
    preamble_pages: List[int] = field(default_factory=list)

    # Structure instance tracking (requirement 4.2)
    _structure_counter: int = field(default_factory=int)
    _structure_stack: List[Dict[str, Any]] = field(default_factory=list)
    # Role of the current structure block: main | amendment | quoted_replacement | appendix
    _structure_role: str = "main"
    # Whether we are inside a quoted replacement block (e.g. "Điều 4. ..." inside amendment)
    _in_quoted_replacement: bool = False
    # Identity of the current amendment operation (op_N), only set when a
    # confidently-supported pattern ("Sửa đổi Điều X như sau") is detected.
    operation_id: Optional[str] = None
    _operation_counter: int = field(default_factory=int)
    # Direct instance id references (avoid stack lookup complexity)
    article_instance_id: Optional[str] = None
    clause_instance_id: Optional[str] = None
    point_instance_id: Optional[str] = None

    def reset_article(self):
        self.article = None
        self.article_title = None
        self.article_page = None
        self.article_instance_id = None
        self.operation_id = None
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
        self.clause_instance_id = None
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
        self.point_instance_id = None

    def _next_instance_id(self) -> str:
        self._structure_counter += 1
        return f"si{self._structure_counter}"

    def _current_instance(self) -> Optional[Dict[str, Any]]:
        return self._structure_stack[-1] if self._structure_stack else None

    def push_structure(self, kind: str, number: str, role: str = "main") -> str:
        """Push a new structure instance onto the stack. Returns its instance id."""
        instance_id = self._next_instance_id()
        parent = self._current_instance()
        self._structure_stack.append({
            "instance_id": instance_id,
            "kind": kind,  # chapter | section | article | clause | point
            "number": number,
            "role": role,
            "parent_id": parent["instance_id"] if parent else None,
        })
        # Store direct reference for emit methods
        if kind == "article":
            self.article_instance_id = instance_id
        elif kind == "clause":
            self.clause_instance_id = instance_id
        elif kind == "point":
            self.point_instance_id = instance_id
        return instance_id

    def pop_structure(self) -> Optional[Dict[str, Any]]:
        if self._structure_stack:
            return self._structure_stack.pop()
        return None

    def set_structure_role(self, role: str) -> None:
        self._structure_role = role

    def set_quoted_replacement(self, active: bool) -> None:
        self._in_quoted_replacement = active


@dataclass
class ParseResult:
    """Structured parser output plus audit information."""

    chunks: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    dropped_lines: int = 0
    articles_seen: List[int] = field(default_factory=list)
    missing_articles: List[int] = field(default_factory=list)
    duplicate_article_numbers: List[int] = field(default_factory=list)
    signature_lines: int = 0
    appendix_chunks: int = 0
    review_lines: List[Dict[str, Any]] = field(default_factory=list)


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
    chunk_type: str,
    document: Optional[DocumentMeta] = None,
    **extra: Any,
) -> Dict[str, Any]:
    meta = (document or resolve_document(None)).as_metadata()
    return {
        **meta,
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
        **extra,
    }


class LegalStructureParser:
    def __init__(self, document: Optional[DocumentMeta] = None):
        self.document = resolve_document(document)
        self.classifier = LineClassifier()
        self.state = ParserState()
        self.chunks = []
        self.warnings = []
        self.seen_articles = {}
        self._article_occurrences: Dict[int, int] = {}
        self.dropped_lines = 0
        self.signature_lines = 0
        self.appendix_chunks = 0
        self.review_lines: List[Dict[str, Any]] = []

    def parse(self, lines: List[dict]) -> ParseResult:
        self.state = ParserState()
        self.chunks = []
        self.warnings = []
        self.dropped_lines = 0
        self.seen_articles = {}
        self._article_occurrences = {}
        self.signature_lines = 0
        self.appendix_chunks = 0
        self.review_lines = []

        main_lines, signature_lines, appendix_lines = self._split_document(lines)

        kept = []
        for line_data in main_lines:
            text = line_data.get("text", "")
            if is_noise_line(text):
                self.dropped_lines += 1
                continue
            kept.append(line_data)
        self.warnings.append(f"Dropped {self.dropped_lines} noise lines")

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

            i = new_i

        self._close_all()
        self._emit_appendix(appendix_lines)

        articles_seen = sorted(self.seen_articles)
        missing: List[int] = []
        expected = self.document.expected_articles
        if expected:
            seen = set(articles_seen)
            missing = [n for n in range(1, expected + 1) if n not in seen]
            if missing:
                self.warnings.append(f"MISSING_ARTICLES: {missing}")

        duplicate_numbers = sorted(
            number
            for number, count in self._article_occurrences.items()
            if count > 1
        )

        return ParseResult(
            chunks=self.chunks,
            warnings=self.warnings,
            dropped_lines=self.dropped_lines,
            articles_seen=articles_seen,
            missing_articles=missing,
            duplicate_article_numbers=duplicate_numbers,
            signature_lines=self.signature_lines,
            appendix_chunks=self.appendix_chunks,
            review_lines=self.review_lines,
        )

    @staticmethod
    def _has_distribution_list(lines: List[dict], index: int, window: int = 8) -> bool:
        """True when the lines after ``index`` look like a ``Nơi nhận`` list.

        A scanned decree can end a normal sentence right before ``Nơi nhận:``
        (OCR loses the ``./.`` marker), and a long appendix can push the
        signature block before the halfway point of the extracted lines.  The
        bullet list that follows the marker is then the reliable signal.
        """
        following = [
            item.get("text", "").strip() for item in lines[index + 1:index + 1 + window]
        ]
        return sum(1 for text in following if text.startswith("-")) >= 2

    def _find_signature_index(self, lines: List[dict]) -> Optional[int]:
        """Locate the trailing signature block without matching quoted text.

        A real block is opened by ``Nơi nhận:`` after a line ending in ``./.``
        and contains a signer title within the next lines.  The VBHN marker is
        accepted only near the end of the document.
        """
        for i, line in enumerate(lines):
            text = line.get("text", "").strip()
            if is_signature_start(text):
                prev = lines[i - 1].get("text", "").rstrip() if i else ""
                window = [item.get("text", "") for item in lines[i:i + 40]]
                near_end = i >= int(0.5 * len(lines))
                has_signer = any(
                    is_signature_title(item) or has_signature_title_anywhere(item)
                    for item in window
                )
                if has_signer and (
                    prev.endswith(("./.", "./", "/"))
                    or near_end
                    or self._has_distribution_list(lines, i)
                ):
                    return i
            if VBHN_SIGNATURE_RE.match(text) and i >= int(0.8 * len(lines)):
                return i
        return None

    def _split_document(
        self, lines: List[dict]
    ) -> Tuple[List[dict], List[dict], List[dict]]:
        """Split into main text, signature block, and appendix lines."""
        signature_idx = self._find_signature_index(lines)
        if signature_idx is not None:
            main = lines[:signature_idx]
            tail = lines[signature_idx:]
            appendix_idx = next(
                (i for i, line in enumerate(tail) if is_appendix_start(line.get("text", ""))),
                None,
            )
            if appendix_idx is None:
                # No explicit appendix heading: the signature block ends at the
                # distribution-list closing line ("Luu: VT, ...").  Anything
                # after it is appendix/table material and must not be dropped.
                signature_end = next(
                    (
                        i + 1
                        for i, line in enumerate(tail)
                        if is_signature_end(line.get("text", ""))
                    ),
                    None,
                )
                if signature_end is not None and signature_end < len(tail):
                    signature = tail[:signature_end]
                    appendix = tail[signature_end:]
                else:
                    signature = tail
                    appendix = []
            else:
                signature = tail[:appendix_idx]
                appendix = tail[appendix_idx:]
            self.signature_lines = len(signature)
            if signature:
                self.warnings.append(
                    f"Signature block: {len(signature)} lines starting on page "
                    f"{signature[0].get('page')}"
                )
            if appendix:
                self.warnings.append(
                    f"Appendix: {len(appendix)} lines starting on page "
                    f"{appendix[0].get('page')}"
                )
            return main, signature, appendix

        appendix_idx = next(
            (
                i
                for i, line in enumerate(lines)
                if is_appendix_start(line.get("text", ""))
                and (i == 0 or i >= int(0.5 * len(lines)))
            ),
            None,
        )
        if appendix_idx is not None:
            self.warnings.append(
                f"Appendix without signature: {len(lines) - appendix_idx} lines "
                f"starting on page {lines[appendix_idx].get('page')}"
            )
            return lines[:appendix_idx], [], lines[appendix_idx:]
        return lines, [], []

    def _emit_appendix(self, lines: List[dict]) -> None:
        if not lines:
            return
        cleaned = [
            line for line in lines if line.get("text", "").strip() and not is_noise_line(line["text"])
        ]
        if not cleaned:
            return

        groups: List[List[dict]] = []
        current: List[dict] = []
        for line in cleaned:
            if is_appendix_heading(line["text"]) and current:
                groups.append(current)
                current = [line]
            else:
                current.append(line)
        if current:
            groups.append(current)

        max_words = 400
        for group in groups:
            title = group[0]["text"].strip()
            pieces: List[List[dict]] = []
            current_piece: List[dict] = []
            current_words = 0
            for line in group:
                words = len(line["text"].split())
                if current_piece and current_words + words > max_words:
                    pieces.append(current_piece)
                    current_piece = []
                    current_words = 0
                current_piece.append(line)
                current_words += words
            if current_piece:
                pieces.append(current_piece)

            for piece in pieces:
                content = join_text([line["text"] for line in piece])
                if not content:
                    continue
                pages = [
                    line["page"] for line in piece if isinstance(line.get("page"), int)
                ]
                self.chunks.append(
                    make_chunk(
                        chapter=None,
                        chapter_title=None,
                        section=None,
                        section_title=None,
                        article=None,
                        article_title=None,
                        clause=None,
                        point=None,
                        content=content,
                        page_start=min(pages) if pages else 1,
                        page_end=max(pages) if pages else 1,
                        chunk_type="appendix",
                        document=self.document,
                        section_type="appendix",
                        appendix_title=title,
                    )
                )
                self.appendix_chunks += 1


    @staticmethod
    def _clean_structure_title(title: str, kind: str) -> str:
        """Strip a page folio accidentally glued to a header title.

        Page running headers merge the title and folio on one visual line
        (e.g. "ĐIỀU KHOẢN THI HÀNH3"). Chapter/section titles never
        legitimately end with bare digits, so a trailing 1-3 digit run is
        removed. Callers should log the change for audit.
        """
        cleaned = re.sub(r"\d{1,3}$", "", title).strip()
        return cleaned

    def _handle_chapter(self, line_info: LineInfo, i: int, classified: List[LineInfo]) -> int:
        match = CHAPTER_RE.match(line_info.text)
        if not match:
            return i

        self._close_all()

        self.state.chapter = f"Chương {match.group(1)}"
        self.state.chapter_title = None
        self.state.section = None
        self.state.section_title = None
        # New chapter resets structure role to main.
        self.state.set_structure_role("main")
        self.state.set_quoted_replacement(False)

        instance_id = self.state.push_structure("chapter", match.group(1), "main")
        chapter_instance_id = instance_id

        title, new_i = self._collect_structure_title(i, classified, "chapter")
        if title:
            self.state.chapter_title = title
            return new_i
        return i + 1

    def _handle_section(self, line_info: LineInfo, i: int, classified: List[LineInfo]) -> int:
        match = SECTION_RE.match(line_info.text)
        if not match:
            return i

        self._close_article()

        self.state.section = f"Mục {match.group(1)}"
        self.state.section_title = None

        instance_id = self.state.push_structure("section", match.group(1), "main")
        section_instance_id = instance_id

        title, new_i = self._collect_structure_title(i, classified, "section")
        if title:
            self.state.section_title = title
            return new_i
        return i + 1

    def _should_start_new_article(self, line_info: LineInfo, i: int, classified: List[LineInfo]) -> bool:
        if not line_info.is_structural_candidate or line_info.candidate_type != "article":
            return False

        features = line_info.features

        article_number = features.get("article_number", 0)
        # Article numbers > 999 are unreasonable for Vietnamese legal docs;
        # reject as false positive (e.g. OCR money amount "110.000.000").
        if article_number > 999:
            return False

        # Only reject if there's STRONG evidence it's a legal reference.
        if features.get("preceded_by_legal_ref"):
            return False

        # If previous line doesn't end with period, it might be a continuation.
        if features.get("prev_line_incomplete"):
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

    def _is_quoted_replacement_start(self, line_info: LineInfo, i: int, classified: List[LineInfo]) -> bool:
        """Detect if this article heading is inside a quoted replacement block.

        Pattern: "1. Sửa đổi Điều 4 như sau:" followed by "Điều 4. ..."
        The second "Điều 4" belongs to the quoted replacement, not the main doc.
        """
        text = line_info.text.strip()
        text_upper = text.upper()

        # If previous line looks like an amendment intro ("Sửa đổi Điều X như sau")
        # and current line starts with "Điều <same number>", it's a quoted replacement.
        if i > 0:
            prev = classified[i - 1]
            prev_text = prev.text.upper().strip()
            if re.search(r"SỬA\s+ĐỔI\s+ĐIỀU\s+\d+\s+NHƯ\s+SAU", prev_text):
                # Current article number matches the one mentioned in prev line
                prev_num_match = re.search(r"ĐIỀU\s+(\d+)", prev_text)
                if prev_num_match:
                    curr_match = ARTICLE_RE.match(line_info.text)
                    if curr_match and int(curr_match.group(1)) == int(prev_num_match.group(1)):
                        return True

        # If we are already inside a quoted replacement, subsequent headings
        # with lower/equal article numbers likely belong to it.
        if self.state._in_quoted_replacement:
            return True

        return False

    def _handle_article(self, line_info: LineInfo, i: int, classified: List[LineInfo]) -> int:
        parsed = match_article_heading(line_info.text)
        if parsed is None:
            return i
        article_number, article_title_text = parsed[0], parsed[1]

        # Reject titles that are just punctuation (e.g. ";", ".", ",").
        if article_title_text and not re.search(r"[\wÀ-ỹà-ỹđĐ]", article_title_text):
            return i

        # Cross-reference fragments (e.g. "của Bộ luật này;") are content.
        if is_ref_like_title(article_title_text):
            return i

        # Determine structure role: main vs quoted_replacement
        is_quoted = self._is_quoted_replacement_start(line_info, i, classified)
        role = "quoted_replacement" if is_quoted else "main"
        if is_quoted:
            self.state.set_quoted_replacement(True)
            # Assign a distinct operation_id for each confidently detected
            # amendment operation.
            self.state._operation_counter += 1
            self.state.operation_id = f"op{self.state._operation_counter}"
        else:
            # A new main article exits quoted replacement context.
            self.state.set_quoted_replacement(False)
            self.state.operation_id = None

        # Close previous article before starting a new one.
        self._close_article()

        candidate_article = f"Điều {article_number}"
        instance_id = self.state.push_structure("article", str(article_number), role)

        self.state.article = candidate_article
        self.state.article_title = article_title_text or None
        self.state.article_page = line_info.page
        self.seen_articles[article_number] = article_title_text
        self._article_occurrences[article_number] = (
            self._article_occurrences.get(article_number, 0) + 1
        )

        new_i = self._collect_multiline_article_title(i, classified)
        return new_i

    def _is_title_continuation(self, text: str) -> bool:
        stripped = text.strip().lstrip("\"'“”‘’([{")
        if not stripped:
            return False
        if len(stripped) > 120:
            return False
        return stripped[0].islower() or stripped[0].isdigit()

    def _is_structure_title_continuation(self, current_title: str, text: str) -> bool:
        stripped = text.strip().lstrip("\"'“”‘’([{")
        if not stripped or len(stripped) > 100:
            return False
        if current_title.rstrip().endswith(","):
            return True
        if stripped[0].islower() or stripped[0].isdigit():
            return True
        cased = [c for c in stripped if c.isalpha()]
        return bool(cased) and all(c.isupper() for c in cased)

    def _collect_structure_title(
        self, i: int, classified: List[LineInfo], kind: str
    ) -> tuple:
        """Collect a possibly multi-line chapter/section title."""
        stop_patterns = [CHAPTER_RE, SECTION_RE, ARTICLE_RE, CLAUSE_RE]
        max_title_len = 200
        max_title_lines = 5

        title = ""
        j = i + 1
        title_lines = 0
        while j < len(classified) and title_lines < max_title_lines:
            next_text = classified[j].text
            if any(p.match(next_text) for p in stop_patterns):
                break
            if is_article_intro_start(next_text):
                break
            candidate = f"{title} {next_text}".strip() if title else next_text
            if title and not self._is_structure_title_continuation(title, next_text):
                break
            if len(candidate) > max_title_len:
                break
            title = candidate
            j += 1
            title_lines += 1

        cleaned = self._clean_structure_title(title, kind)
        if cleaned != title:
            self.warnings.append(
                f"Stripped folio from {kind} title: {title!r} -> {cleaned!r}"
            )
        return cleaned, j

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
        if features.get("is_likely_money_amount"):
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

        # Emit any pending point before starting a new clause
        self._emit_point()

        # Emit clause_intro BEFORE starting the new clause (requirement 4.4)
        if self.state.clause_intro and self.state.clause is not None:
            self._emit_clause_intro()

        # Close previous clause
        self._emit_clause()

        self.state.clause = match.group(1)
        self.state.clause_parts = [match.group(2)]
        self.state.clause_pages = [line_info.page]
        self.state.clause_has_points = False

        # Push clause instance
        role = "quoted_replacement" if self.state._in_quoted_replacement else "main"
        self.state.push_structure("clause", match.group(1), role)

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

        # Emit clause_intro before first point (requirement 4.4)
        if self.state.clause_intro:
            self._emit_clause_intro()

        self._emit_point()

        self.state.point = match.group(1).lower()
        self.state.point_parts = [match.group(2)]
        self.state.point_pages = [line_info.page]

        # Push point instance
        role = "quoted_replacement" if self.state._in_quoted_replacement else "main"
        self.state.push_structure("point", match.group(1).lower(), role)

        return i + 1

    def _handle_content(self, line_info: LineInfo):
        text = line_info.text
        page = line_info.page

        # Route content to the correct bucket based on current structure state.
        if self.state.point is not None:
            self.state.point_parts.append(text)
            self.state.point_pages.append(page)
        elif self.state.clause is not None:
            self.state.clause_parts.append(text)
            self.state.clause_pages.append(page)
        elif self.state.article is not None:
            # Inside an article but before any clause -> article intro
            self.state.article_intro.append(text)
            self.state.article_intro_pages.append(page)
        elif self.state.section is not None:
            # Inside a section but before any article -> section intro
            if not hasattr(self.state, 'section_intro'):
                self.state.section_intro = []
                self.state.section_intro_pages = []
            self.state.section_intro.append(text)
            self.state.section_intro_pages.append(page)
        elif self.state.chapter is not None:
            # Inside a chapter but before any section/article -> chapter intro
            if not hasattr(self.state, 'chapter_intro'):
                self.state.chapter_intro = []
                self.state.chapter_intro_pages = []
            self.state.chapter_intro.append(text)
            self.state.chapter_intro_pages.append(page)
        elif not self.seen_articles:
            # Before any article -> preamble
            self.state.preamble.append(text)
            self.state.preamble_pages.append(page)
        else:
            # After all articles but before signature -> treat as article intro
            # of the last known article, or preamble if no article yet.
            if self.state.article:
                self.state.article_intro.append(text)
                self.state.article_intro_pages.append(page)
            else:
                self.state.preamble.append(text)
                self.state.preamble_pages.append(page)

    # ── Emit methods ──────────────────────────────────────────────────────

    def _emit_preamble(self):
        content = join_text(self.state.preamble)
        if content:
            pages = self.state.preamble_pages
            self.chunks.append(make_chunk(
                chapter=None,
                chapter_title=None,
                section=None,
                section_title=None,
                article=None,
                article_title=None,
                clause=None,
                point=None,
                content=content,
                page_start=min(pages) if pages else 1,
                page_end=max(pages) if pages else 1,
                chunk_type="preamble",
                document=self.document,
                structure_role="main",
            ))
        self.state.preamble = []
        self.state.preamble_pages = []

    def _emit_section_intro(self):
        content = join_text(getattr(self.state, 'section_intro', []))
        if content and self.state.section is not None:
            pages = getattr(self.state, 'section_intro_pages', [])
            self.chunks.append(make_chunk(
                chapter=self.state.chapter,
                chapter_title=self.state.chapter_title,
                section=self.state.section,
                section_title=self.state.section_title,
                article=None,
                article_title=None,
                clause=None,
                point=None,
                content=content,
                page_start=min(pages) if pages else 1,
                page_end=max(pages) if pages else 1,
                chunk_type="section_intro",
                document=self.document,
                structure_role=self.state._structure_role,
            ))
        self.state.section_intro = []
        self.state.section_intro_pages = []

    def _emit_chapter_intro(self):
        content = join_text(getattr(self.state, 'chapter_intro', []))
        if content and self.state.chapter is not None:
            pages = getattr(self.state, 'chapter_intro_pages', [])
            self.chunks.append(make_chunk(
                chapter=self.state.chapter,
                chapter_title=self.state.chapter_title,
                section=None,
                section_title=None,
                article=None,
                article_title=None,
                clause=None,
                point=None,
                content=content,
                page_start=min(pages) if pages else 1,
                page_end=max(pages) if pages else 1,
                chunk_type="chapter_intro",
                document=self.document,
                structure_role=self.state._structure_role,
            ))
        self.state.chapter_intro = []
        self.state.chapter_intro_pages = []

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
                chunk_type="article_intro",
                document=self.document,
                structure_instance_id=self.state.article_instance_id,
                parent_instance_id=self.state._current_instance()["parent_id"] if self.state._current_instance() else None,
                structure_role=self.state._structure_role,
                operation_id=self.state.operation_id,
            ))
        self.state.article_intro = []
        self.state.article_intro_pages = []

    def _emit_clause_intro(self):
        content = join_text(self.state.clause_intro)
        if content and self.state.article is not None and self.state.clause is not None:
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
                chunk_type="clause_intro",
                document=self.document,
                structure_instance_id=self.state.clause_instance_id,
                parent_instance_id=self.state.article_instance_id,
                structure_role=self.state._structure_role,
                operation_id=self.state.operation_id,
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
                    chunk_type="clause",
                    document=self.document,
                    structure_instance_id=self.state.clause_instance_id,
                    parent_instance_id=self.state.article_instance_id,
                    structure_role=self.state._structure_role,
                    operation_id=self.state.operation_id,
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
                    chunk_type="point",
                    document=self.document,
                    structure_instance_id=self.state.point_instance_id,
                    parent_instance_id=self.state.clause_instance_id,
                    structure_role=self.state._structure_role,
                    operation_id=self.state.operation_id,
                ))
        self.state.reset_point()

    def _close_article(self):
        self._emit_point()
        self._emit_clause()
        self._emit_article_intro()
        self.state.reset_article()

    def _close_all(self):
        self._emit_preamble()
        self._emit_section_intro()
        self._emit_chapter_intro()
        self._close_article()


def parse_legal_structure(
    lines: List[dict], document: Optional[DocumentMeta] = None
) -> ParseResult:
    parser = LegalStructureParser(document)
    return parser.parse(lines)