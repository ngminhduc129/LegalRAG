"""Classifier for legal document structure lines.

Responsibilities (classification only, NOT parsing):
- Classify each line as a structural candidate (chapter, section, article,
  clause, point) or plain content.
- Extract context-aware features from previous / current / next line.
- Provide canonical article / clause / point number extraction.
- Preserve raw text and provenance (page, line_index, text_source).

The classifier does NOT decide hierarchy.  Parsing decisions (whether a
candidate really starts a new article, whether a point belongs to a clause,
etc.) live in parser.py.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path

from legal_rag.parsing.constants import FUZZY_ARTICLE_TOKENS


@dataclass
class LineInfo:
    """A single text line with classification metadata.

    ``raw_text`` is preserved untouched for audit / debugging.  ``text`` is the
    normalized form used for matching.
    """

    text: str
    page: int
    line_index: int

    # Provenance
    raw_text: str = ""
    text_source: str = "text_layer"  # "text_layer" | "ocr"

    # Classification outputs
    is_structural_candidate: bool = False
    candidate_type: Optional[str] = None  # chapter | section | article | clause | point
    confidence: float = 0.0
    uncertainty: Optional[str] = None  # None | "low_confidence" | "ambiguous_context"
    uncertainty_reason: Optional[str] = None

    # Structure role (filled by parser, not classifier)
    structure_role: Optional[str] = None  # main | amendment | quoted_replacement | appendix

    features: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.raw_text:
            self.raw_text = self.text


# ── Structural regexes ──────────────────────────────────────────────────────

CHAPTER_RE = re.compile(r"^\s*Chương\s+([IVXLCDM]+|\d+)\s*$", re.IGNORECASE)
SECTION_RE = re.compile(r"^\s*Mục\s+(\d+)\s*$", re.IGNORECASE)
# Article heading: "Điều N." or "Điều N Title".  Requires "Điều" prefix.
ARTICLE_RE = re.compile(r"^\s*Điều\s+(\d+)\s*\.?\s*(.*?)\s*$", re.IGNORECASE)
# Fallback for corrupted headings (mojibake/OCR), e.g. "Di~u 220." or "Điu 220."
# Same group layout as ARTICLE_RE: (1)=number, (2)=title.  Only applied when
# the first token normalizes to a known "Điều" variant.
FUZZY_ARTICLE_RE = re.compile(r"^\s*\S+\s+(\d+)\s*\.?\s*(.*?)\s*$")
CLAUSE_RE = re.compile(r"^\s*(\d+)\s*\.\s+(.+?)\s*$")
POINT_RE = re.compile(r"^\s*([a-zđ])\s*\)\s*(.*?)\s*$", re.IGNORECASE)


LEGAL_REF_PREFIXES = (
    "THEO QUY ĐỊNH TẠI",
    "THEO QUY ĐỊNH CỦA",
    "QUY ĐỊNH TẠI",
    "CỦA ĐIỀU",
    "TẠI ĐIỀU",
    r"KHOẢN \d+ ĐIỀU",
    r"ĐIỀU \d+ CỦA",
    r"ĐIỀU \d+ CỦA BỘ LUẬT",
    "THEO ĐIỀU",
    "THEO KHOẢN",
)

ARTICLE_INTRO_STARTS = (
    "TRONG BỘ LUẬT", "TRONG LUẬT", "THEO QUY ĐỊNH",
    "CĂN CỨ", "QUỐC HỘI", "CHÍNH PHỦ", "BỘ TRƯỞNG",
    "CỤC TRƯỞNG", "HUYỆN TRƯỞNG", "XÃ TRƯỞNG",
)

# Token that can legitimately appear right after an article number in a heading
# (e.g. "Điều 1. Sửa đổi...").  Used to reject "Điều 110.000.000 đồng".
_HEADING_TITLE_STARTS = re.compile(r"[\wÀ-ỹà-ỹđĐ]")


# ── Helpers ─────────────────────────────────────────────────────────────────

def normalize_vi(text: str) -> str:
    """Lowercase, strip diacritics and non-letters for robust matching."""
    text = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z]", "", text)


def is_page_number(text: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}", text.strip()))


def looks_like_money_or_date(text: str) -> bool:
    """True for short money / date expressions starting with a number."""
    text = text.strip()
    return bool(re.match(r"^\d{1,2}\s*(tháng|ngày|năm|tuổi|triệu|nghìn|đồng|USD|%)", text, re.IGNORECASE))


def is_likely_money_amount(text: str) -> bool:
    """True for large money amounts like "110.000.000 đồng" that can be
    misread as an article number ("110") by a naive regex."""
    text = text.strip()
    # Groups of 3 digits separated by dots, optionally followed by currency word.
    return bool(re.match(r"^\d{1,3}(\.\d{3})+\s*(đồng|usd|tỷ|triệu|nghìn)?$", text, re.IGNORECASE))


def is_likely_legal_reference_in_content(text: str) -> bool:
    """True if the line is a cross-reference embedded in content, e.g.
    "khoản 5 Điều 13 của Bộ luật Lao động".  These must NOT start a new
    article heading."""
    text_upper = text.upper().strip()
    # Must contain both "ĐIỀU" and a reference-preposition context.
    if "ĐIỀU" not in text_upper:
        return False
    for prefix in LEGAL_REF_PREFIXES:
        if re.search(prefix.replace(" ", r"\s+"), text_upper):
            return True
    # Direct pattern: "KHOẢN N ĐIỀU M" or "ĐIỀU M CỦA ..."
    if re.search(r"(KHOẢN\s+\d+\s+ĐIỀU\s+\d+|ĐIỀU\s+\d+\s+CỦA)", text_upper):
        return True
    return False


def looks_like_clause_number_only(text: str) -> bool:
    text = text.strip()
    return bool(re.fullmatch(r"\d{1,3}", text)) and len(text) <= 3


def is_legal_reference_continuation(prev_text: str, curr_text: str) -> bool:
    """True if prev_text ends with a legal-reference prefix that makes curr_text
    a continuation (not a new heading)."""
    if not prev_text:
        return False
    prev_upper = prev_text.upper().strip()
    curr_upper = curr_text.upper().strip()

    for prefix in LEGAL_REF_PREFIXES:
        norm = prefix.replace(" ", "")
        if prev_upper.endswith(norm):
            return True
        if re.search(norm + r"$", prev_upper):
            return True

    if re.search(r"(KHOẢN|ĐIỀU)\s+\d+\s+(ĐIỀU|CHƯƠNG|MỤC)\s+\d+\s*$", prev_upper):
        return True

    if (prev_upper.endswith("TẠI") or prev_upper.endswith("THEO") or
        prev_upper.endswith("CỬA") or prev_upper.endswith("QUY ĐỊNH")):
        return True

    return False


def is_article_intro_start(text: str) -> bool:
    text_upper = text.upper().strip()
    return any(text_upper.startswith(p) for p in ARTICLE_INTRO_STARTS)


def is_noise_line(text: str) -> bool:
    """True for lines carrying no content (pure punctuation/symbols)."""
    stripped = text.strip()
    if not stripped:
        return True
    return not re.search(r"[\wÀ-ỹà-ỹđĐ]", stripped)


# ── Signature / appendix detection ──────────────────────────────────────────

SIGNATURE_LINE_RE = re.compile(r"^\s*Nơi nhận\s*:", re.IGNORECASE)
SIGNATURE_TITLE_RE = re.compile(
    r"^\s*(TM\.|KT\.|BỘ TRƯỞNG|THỦ TƯỚNG|CHỦ TỊCH|PHÓ THỦ TƯỚNG|THỨ TRƯỞNG)",
    re.IGNORECASE,
)
SIGNATURE_TITLE_ANY_RE = re.compile(
    r"\b(TM\.|KT\.|BỘ TRƯỞNG|THỦ TƯỚNG|CHỦ TỊCH|PHÓ THỦ TƯỚNG|THỨ TRƯỞNG)\b",
    re.IGNORECASE,
)
APPENDIX_LINE_RE = re.compile(
    r"^\s*(PHỤ LỤC|Phụ lục)(\s+[IVXLC\d]+)?\s*$|^\s*(Mẫu số|Biểu số)\b",
    re.IGNORECASE,
)
APPENDIX_HEADING_RE = re.compile(
    r"^\s*(PHỤ LỤC|Phụ lục)(\s+[IVXLC\d]+)?\s*$",
    re.IGNORECASE,
)
VBHN_SIGNATURE_RE = re.compile(
    r"^\s*(VAN PHONG QUOC HQI|VĂN PHÒNG QUỐC HỘI)\s*$",
    re.IGNORECASE,
)
SIGNATURE_END_RE = re.compile(r"\bLưu\s*[:：]", re.IGNORECASE)


def is_signature_start(text: str) -> bool:
    return bool(SIGNATURE_LINE_RE.match(text))


def is_signature_title(text: str) -> bool:
    return bool(SIGNATURE_TITLE_RE.match(text))


def has_signature_title_anywhere(text: str) -> bool:
    return bool(SIGNATURE_TITLE_ANY_RE.search(text))


def is_appendix_start(text: str) -> bool:
    return bool(APPENDIX_LINE_RE.match(text))


def is_appendix_heading(text: str) -> bool:
    return bool(APPENDIX_HEADING_RE.match(text))


def is_signature_end(text: str) -> bool:
    return bool(SIGNATURE_END_RE.search(text))


# ── Line cleaning ───────────────────────────────────────────────────────────

def clean_line(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return strip_heading_noise(text)


# OCR frequently prepends a stray symbol or a single stray letter to a real
# heading ("| Điều 11.", "ặ Điều 33.").  Strip it only when the rest of the
# line is an article heading with a title separator.
HEADING_NOISE_RE = re.compile(
    r"^\s*(?:\W{1,4}|(?![ởỞ])[^\W\d_])\s+(?=Điều\s+\d+\s*[\.\:])", re.IGNORECASE
)


def strip_heading_noise(text: str) -> str:
    """Remove a stray OCR prefix before a real article heading."""
    return HEADING_NOISE_RE.sub("", text, count=1)


# ── Article / clause / point extraction ─────────────────────────────────────

def match_article_heading(text: str) -> Optional[Tuple[int, str, bool]]:
    """Match an article heading, tolerating corrupted "Điều" spellings.

    Returns (number, title, fuzzy) or None.  fuzzy=True when ARTICLE_RE fails
    but the first token normalizes to a known variant (e.g. "Di~u", "Điu").

    Rejects false positives:
    - Starts with quotation mark (quoted amendment from another document).
    - Title is punctuation-only (e.g. "Điều 23;").
    - Title starts with digit / dot / comma (money continuation: "Điều 110.000.000 đồng").
    - Title looks like money / date.
    - Article number > 999 (unreasonable for Vietnamese legal docs).
    - Line looks like a legal reference embedded in content.
    """
    if text.lstrip().startswith(('"', "“", "”", "„", "'")):
        return None

    # Exact "Điều N[. Title]" pattern.
    m = ARTICLE_RE.match(text)
    if m:
        num = int(m.group(1))
        title = m.group(2).strip()
        if _reject_article_candidate(num, title, text):
            return None
        return num, title, False

    # Fuzzy fallback for corrupted OCR headings.
    m2 = FUZZY_ARTICLE_RE.match(text)
    if m2 and normalize_vi(m2.group(0).split()[0]) in FUZZY_ARTICLE_TOKENS:
        num = int(m2.group(1))
        title = m2.group(2).strip()
        if _reject_article_candidate(num, title, text):
            return None
        return num, title, True

    return None


def _reject_article_candidate(num: int, title: str, full_text: str) -> bool:
    """Return True if the candidate should be rejected as a false positive."""
    if num > 999:
        return True
    if title and not _HEADING_TITLE_STARTS.search(title):
        return True
    if title and re.match(r"^[\d.,]", title):
        return True
    if looks_like_money_or_date(title):
        return True
    if is_likely_money_amount(full_text):
        return True
    if is_likely_legal_reference_in_content(full_text):
        return True
    return False


def extract_article_number(text: str) -> Optional[int]:
    """Extract article number using the same validation as match_article_heading."""
    result = match_article_heading(text)
    if result is None:
        return None
    return result[0]


def extract_clause_number(text: str) -> Optional[int]:
    m = CLAUSE_RE.match(text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


# ── LineClassifier ──────────────────────────────────────────────────────────

class LineClassifier:
    def __init__(self):
        self.structural_patterns = [
            ("chapter", CHAPTER_RE),
            ("section", SECTION_RE),
            ("article", ARTICLE_RE),
            ("clause", CLAUSE_RE),
            ("point", POINT_RE),
        ]

    def classify_line(self, text: str, page: int, index: int,
                      prev_line: Optional["LineInfo"] = None,
                      next_line: Optional["LineInfo"] = None,
                      raw_text: Optional[str] = None,
                      text_source: str = "text_layer") -> LineInfo:
        """Classify a single line.

        ``raw_text`` preserves the original OCR / extraction output for audit.
        ``next_line`` is required for context-aware features (previously always
        None in classify_all()).
        """
        line = LineInfo(
            text=text,
            page=page,
            line_index=index,
            raw_text=raw_text if raw_text is not None else text,
            text_source=text_source,
        )

        candidate_type = None
        confidence = 0.0
        match = None

        for ctype, pattern in self.structural_patterns:
            m = pattern.match(text)
            if m:
                candidate_type = ctype
                match = m
                confidence = 0.8
                break

        fuzzy_heading = False
        if not candidate_type:
            fuzzy = match_article_heading(text)
            if fuzzy is not None and fuzzy[2]:
                candidate_type = "article"
                match = FUZZY_ARTICLE_RE.match(text)
                confidence = 0.5
                fuzzy_heading = True

        if not candidate_type:
            line.is_structural_candidate = False
            return line

        line.is_structural_candidate = True
        line.candidate_type = candidate_type
        line.confidence = confidence

        features = self._extract_features(
            text, candidate_type, match, prev_line, next_line
        )
        features["fuzzy_heading"] = fuzzy_heading
        line.features = features

        line.confidence = self._calculate_confidence(candidate_type, features)

        # Mark uncertain cases for parser to handle as REVIEW.
        if confidence < 0.5:
            line.uncertainty = "low_confidence"
            line.uncertainty_reason = f"low classifier confidence ({confidence:.2f})"
        if fuzzy_heading and confidence < 0.7:
            if line.uncertainty is None:
                line.uncertainty = "ambiguous_context"
                line.uncertainty_reason = "fuzzy OCR heading with moderate confidence"

        return line

    def _extract_features(self, text: str, candidate_type: str,
                          match: re.Match, prev_line: Optional[LineInfo],
                          next_line: Optional[LineInfo]) -> dict:
        text_upper = text.upper().strip()
        features = {}

        if candidate_type == "article":
            title = match.group(2).strip()
            features["has_title"] = bool(title)
            features["title_length"] = len(title)
            features["article_number"] = int(match.group(1))

            if next_line:
                next_text = next_line.text
                features["next_line_is_clause"] = bool(CLAUSE_RE.match(next_text))
                features["next_line_is_point"] = bool(POINT_RE.match(next_text))
                features["next_line_is_article"] = bool(ARTICLE_RE.match(next_text))
                features["next_line_is_section"] = bool(SECTION_RE.match(next_text))
                features["next_line_is_chapter"] = bool(CHAPTER_RE.match(next_text))
                features["followed_by_paragraph"] = (
                    next_text and not any([
                        CLAUSE_RE.match(next_text),
                        POINT_RE.match(next_text),
                        ARTICLE_RE.match(next_text),
                        SECTION_RE.match(next_text),
                        CHAPTER_RE.match(next_text),
                    ])
                )
            else:
                features["next_line_is_clause"] = False
                features["followed_by_paragraph"] = False

            if prev_line:
                features["prev_line_incomplete"] = is_legal_reference_continuation(
                    prev_line.text, text
                )
                features["preceded_by_legal_ref"] = is_legal_reference_continuation(
                    prev_line.text, text
                )
                # Previous line is a structural candidate → current is less likely
                # to be a fresh article (may be continuation or reference).
                prev_is_structural = prev_line.is_structural_candidate
                features["prev_is_structural"] = prev_is_structural
                if prev_is_structural:
                    features["prev_candidate_type"] = prev_line.candidate_type
            else:
                features["prev_line_incomplete"] = False
                features["preceded_by_legal_ref"] = False
                features["prev_is_structural"] = False

            features["ends_with_punct"] = text_upper.endswith((".", ":", ";"))
            features["starts_with_quotation"] = text_upper.startswith(('"', "“", "”"))

        elif candidate_type == "clause":
            clause_text = match.group(2).strip()
            features["clause_number"] = int(match.group(1))
            features["content_length"] = len(clause_text)
            features["is_number_only"] = looks_like_clause_number_only(text)
            features["looks_like_money_date"] = looks_like_money_or_date(text)
            features["is_likely_money_amount"] = is_likely_money_amount(text)

            if prev_line:
                features["preceded_by_article_intro"] = not bool(
                    CLAUSE_RE.match(prev_line.text) or
                    POINT_RE.match(prev_line.text) or
                    ARTICLE_RE.match(prev_line.text)
                )
                features["prev_is_clause"] = (
                    prev_line.is_structural_candidate and
                    prev_line.candidate_type == "clause"
                )
            else:
                features["preceded_by_article_intro"] = False
                features["prev_is_clause"] = False

        elif candidate_type == "point":
            features["point_letter"] = match.group(1).lower()
            features["content_length"] = len(match.group(2).strip())
            # Do NOT hardcode has_parent_clause — parser decides based on state.
            features["has_parent_clause"] = None

        elif candidate_type == "section":
            features["has_title"] = False
            if next_line:
                next_text = next_line.text
                features["next_line_is_article"] = bool(ARTICLE_RE.match(next_text))
                features["next_line_is_clause"] = bool(CLAUSE_RE.match(next_text))

        return features

    def _calculate_confidence(self, candidate_type: str, features: dict) -> float:
        base = 0.8

        if candidate_type == "article":
            if features.get("next_line_is_clause"):
                base += 0.15
            if features.get("has_title") and features.get("title_length", 0) > 5:
                base += 0.1
            if features.get("followed_by_paragraph"):
                base += 0.1
            if features.get("prev_line_incomplete") or features.get("preceded_by_legal_ref"):
                base -= 0.5
            if features.get("prev_is_structural"):
                base -= 0.2
            if features.get("starts_with_quotation"):
                base -= 0.6
            if not features.get("has_title") and features.get("title_length", 0) < 3:
                base -= 0.3
            if features.get("ends_with_punct"):
                base += 0.05

        elif candidate_type == "clause":
            if features.get("is_number_only") or features.get("looks_like_money_date"):
                base -= 0.4
            if features.get("is_likely_money_amount"):
                base -= 0.5
            if features.get("preceded_by_article_intro"):
                base += 0.1
            if features.get("prev_is_clause"):
                base -= 0.1
            if features.get("content_length", 0) < 3:
                base -= 0.2

        elif candidate_type == "point":
            base += 0.1

        elif candidate_type == "section":
            if features.get("next_line_is_article"):
                base += 0.1

        return min(1.0, max(0.0, base))

    def classify_all(self, lines: List[dict]) -> List[LineInfo]:
        """Classify all lines with real next_line context (look-ahead by 1)."""
        classified: List[LineInfo] = []
        for i, line_data in enumerate(lines):
            prev = classified[i - 1] if i > 0 else None
            next_raw = lines[i + 1] if i + 1 < len(lines) else None
            next_line = None
            if next_raw is not None:
                next_line = LineInfo(
                    text=next_raw["text"],
                    page=next_raw["page"],
                    line_index=i + 1,
                    raw_text=next_raw.get("raw_text"),
                    text_source=next_raw.get("text_source", "text_layer"),
                )
            classified.append(
                self.classify_line(
                    line_data["text"],
                    line_data["page"],
                    i,
                    prev_line=prev,
                    next_line=next_line,
                    raw_text=line_data.get("raw_text"),
                    text_source=line_data.get("text_source", "text_layer"),
                )
            )
        return classified


# NOTE: extract_pdf_lines() lives in legal_rag.corpus.extractor.  Do NOT
# duplicate PDF extraction here — classifier should not implement extraction.
