import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path

from legal_rag.parsing.constants import FUZZY_ARTICLE_TOKENS


@dataclass
class LineInfo:
    text: str
    page: int
    index: int

    is_structural_candidate: bool = False
    candidate_type: Optional[str] = None
    confidence: float = 0.0

    features: dict = field(default_factory=dict)

    raw_text: str = ""

    def __post_init__(self):
        self.raw_text = self.text


CHAPTER_RE = re.compile(r"^\s*Chương\s+([IVXLCDM]+|\d+)\s*$", re.IGNORECASE)
SECTION_RE = re.compile(r"^\s*Mục\s+(\d+)\s*$", re.IGNORECASE)
ARTICLE_RE = re.compile(r"^\s*Điều\s+(\d+)\s*\.?\s*(.*?)\s*$", re.IGNORECASE)
# Fallback for corrupted headings (mojibake/OCR), e.g. "Di~u 220." or "Điu 220.".
# Same group layout as ARTICLE_RE: (1)=number, (2)=title. Only applied when the
# first token normalizes to a known "Điều" variant (see match_article_heading).
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


def normalize_vi(text: str) -> str:
    """Lowercase, strip diacritics and non-letters for robust matching."""
    text = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z]", "", text)


def match_article_heading(text: str) -> Optional[Tuple[int, str, bool]]:
    """Match an article heading, tolerating corrupted "Điều" spellings.

    Returns (number, title, fuzzy) or None. fuzzy=True when ARTICLE_RE fails
    but the first token normalizes to a known variant (e.g. "Di~u", "Điu").

    A heading that starts with a quotation mark is a quoted amendment of
    another document ("Điều 4. Sử dụng người lao động nước ngoài" inside
    "Điều 1. Sửa đổi..."), i.e. content of the enclosing article, never a new
    article of the current document.
    """
    if text.lstrip().startswith(('"', "“", "”", "„", "'")):
        return None
    m = ARTICLE_RE.match(text)
    if m:
        return int(m.group(1)), m.group(2).strip(), False
    m2 = FUZZY_ARTICLE_RE.match(text)
    if m2 and normalize_vi(m2.group(0).split()[0]) in FUZZY_ARTICLE_TOKENS:
        return int(m2.group(1)), m2.group(2).strip(), True
    return None


def is_noise_line(text: str) -> bool:
    """True for lines carrying no content (pure punctuation/symbols)."""
    stripped = text.strip()
    if not stripped:
        return True
    return not re.search(r"[\wÀ-ỹà-ỹđĐ]", stripped)


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
    """True only for an actual signature-block opening line.

    The previous implementation normalized the whole line and matched
    ``"noinhan"`` with ``startswith``.  That also matched quoted text such as
    ``“Nơi nhận”`` inside an amendment and distribution-list lines such as
    ``- Văn phòng Quốc hội;``.  Requiring the raw ``Nơi nhận:`` prefix avoids
    both false positives.
    """
    return bool(SIGNATURE_LINE_RE.match(text))


def is_signature_title(text: str) -> bool:
    return bool(SIGNATURE_TITLE_RE.match(text))


def has_signature_title_anywhere(text: str) -> bool:
    """True if a signer title appears anywhere in the line.

    Some PDFs merge the distribution header and the signer title on one line
    (``"Nơi nhận: BỘ TRƯỞNG"``), so an anchored match alone misses the block.
    """
    return bool(SIGNATURE_TITLE_ANY_RE.search(text))


def is_appendix_start(text: str) -> bool:
    return bool(APPENDIX_LINE_RE.match(text))


def is_appendix_heading(text: str) -> bool:
    """True only for a top-level ``PHỤ LỤC ...`` heading."""
    return bool(APPENDIX_HEADING_RE.match(text))


def is_signature_end(text: str) -> bool:
    """True for the closing distribution-list line (``Lưu: VT, ...``)."""
    return bool(SIGNATURE_END_RE.search(text))


def clean_line(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return strip_heading_noise(text)


# OCR frequently prepends a stray symbol or a single stray letter to a real
# heading ("| Điều 11.", "ặ Điều 33.").  Strip it only when the rest of the
# line is an article heading with a title separator, so cross references
# ("theo Điều 154 Bộ luật Lao động") are left untouched.
HEADING_NOISE_RE = re.compile(
    r"^\s*(?:\W{1,4}|(?![ởỞ])[^\W\d_])\s+(?=Điều\s+\d+\s*[\.\:])", re.IGNORECASE
)


def strip_heading_noise(text: str) -> str:
    """Remove a stray OCR prefix before a real article heading."""
    return HEADING_NOISE_RE.sub("", text, count=1)


def is_page_number(text: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}", text.strip()))


def looks_like_money_or_date(text: str) -> bool:
    text = text.strip()
    return bool(re.match(r"^\d{1,2}\s*(tháng|ngày|năm|tuổi|triệu|nghìn|đồng|USD|%)", text, re.IGNORECASE))


def looks_like_clause_number_only(text: str) -> bool:
    text = text.strip()
    return bool(re.fullmatch(r"\d{1,3}", text)) and len(text) <= 3


def is_legal_reference_continuation(prev_text: str, curr_text: str) -> bool:
    if not prev_text:
        return False
    prev_upper = prev_text.upper().strip()
    curr_upper = curr_text.upper().strip()

    for prefix in LEGAL_REF_PREFIXES:
        if prev_upper.endswith(prefix.replace(" ", "")):
            return True
        if re.search(prefix.replace(" ", r"\s+") + r"$", prev_upper):
            return True

    # Only match legal references with cross-reference context (e.g., "KHOẢN 1 ĐIỀU 5")
    # Not standalone headers like "MỤC 1" or "CHƯƠNG I"
    if re.search(r"(KHOẢN|ĐIỀU)\s+\d+\s+(ĐIỀU|CHƯƠNG|MỤC)\s+\d+\s*$", prev_upper):
        return True

    if (prev_upper.endswith("TẠI") or prev_upper.endswith("THEO") or
        prev_upper.endswith("CỬA") or prev_upper.endswith("QUY ĐỊNH")):
        return True

    return False


def is_article_intro_start(text: str) -> bool:
    text_upper = text.upper().strip()
    return any(text_upper.startswith(p) for p in ARTICLE_INTRO_STARTS)


def extract_article_number(text: str) -> Optional[int]:
    m = ARTICLE_RE.match(text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def extract_clause_number(text: str) -> Optional[int]:
    m = CLAUSE_RE.match(text)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


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
                      next_line: Optional["LineInfo"] = None) -> LineInfo:
        line = LineInfo(text=text, page=page, index=index)

        candidate_type = None
        confidence = 0.0

        for ctype, pattern in self.structural_patterns:
            match = pattern.match(text)
            if match:
                candidate_type = ctype
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

        return line

    def _extract_features(self, text: str, candidate_type: str,
                          match: re.Match, prev_line: Optional[LineInfo],
                          next_line: Optional[LineInfo]) -> dict:
        text_upper = text.upper().strip()
        features = {}

        if candidate_type == "article":
            features["has_title"] = bool(match.group(2).strip())
            features["title_length"] = len(match.group(2).strip())
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
            else:
                features["prev_line_incomplete"] = False
                features["preceded_by_legal_ref"] = False

            features["ends_with_punct"] = text_upper.endswith((".", ":", ";"))

        elif candidate_type == "clause":
            clause_text = match.group(2).strip()
            features["clause_number"] = int(match.group(1))
            features["content_length"] = len(clause_text)
            features["is_number_only"] = looks_like_clause_number_only(text)
            features["looks_like_money_date"] = looks_like_money_or_date(text)

            if prev_line:
                features["preceded_by_article_intro"] = not bool(
                    CLAUSE_RE.match(prev_line.text) or
                    POINT_RE.match(prev_line.text) or
                    ARTICLE_RE.match(prev_line.text)
                )
            else:
                features["preceded_by_article_intro"] = False

        elif candidate_type == "point":
            features["point_letter"] = match.group(1).lower()
            features["content_length"] = len(match.group(2).strip())
            features["has_parent_clause"] = True

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
            if not features.get("has_title") and features.get("title_length", 0) < 3:
                base -= 0.3
            if features.get("ends_with_punct"):
                base += 0.05

        elif candidate_type == "clause":
            if features.get("is_number_only") or features.get("looks_like_money_date"):
                base -= 0.4
            if features.get("preceded_by_article_intro"):
                base += 0.1
            if features.get("content_length", 0) < 3:
                base -= 0.2

        elif candidate_type == "point":
            base += 0.1

        elif candidate_type == "section":
            if features.get("next_line_is_article"):
                base += 0.1

        return min(1.0, max(0.0, base))

    def classify_all(self, lines: List[dict]) -> List[LineInfo]:
        classified = []
        for i, line_data in enumerate(lines):
            prev = classified[i - 1] if i > 0 else None
            next_ = None
            classified.append(
                self.classify_line(
                    line_data["text"],
                    line_data["page"],
                    i,
                    prev_line=prev,
                    next_line=next_
                )
            )
        return classified


def extract_pdf_lines(pdf_path: Path) -> List[dict]:
    import fitz
    lines = []
    with fitz.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            page_num = page_index + 1
            page_text = page.get_text("text", sort=True)
            for raw_line in page_text.splitlines():
                line = clean_line(raw_line)
                if not line:
                    continue
                if is_page_number(line):
                    continue
                lines.append({"text": line, "page": page_num})
    return lines