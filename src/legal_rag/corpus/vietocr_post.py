"""Post-processing rules for VietOCR output on Vietnamese legal documents.

Fixes systematic recognition errors observed on the dev set, e.g. "Điều" →
"Điêu", "bổ sung" → "bô sung", missing diacritics.  Applied per output line
inside VietOcrEngine so the cached text is already cleaned.
"""

# Ordered list of (wrong, correct) replacements.  Exact string match (no
# regex) so Vietnamese diacritics are preserved predictably.
FIXES = [
    # Legal-term system errors
    ("Điêu", "Điều"),
    ("điêu", "điều"),
    ("ĐIÊU", "ĐIỀU"),
    # Missing diacritics / confusions
    ("bô sung", "bổ sung"),
    ("Bô sung", "Bổ sung"),
    ("sửa đôi", "sửa đổi"),
    ("thể ", "thể "),  # no-op guard, kept for documentation
    ("l)oanh", "Doanh"),
    ("l7", "17"),
    ("Ì", "6"),
]


def postprocess(text: str) -> str:
    """Apply deterministic fixes to one line of VietOCR output."""
    for wrong, correct in FIXES:
        if wrong in text:
            text = text.replace(wrong, correct)
    return text


def postprocess_lines(lines: list[str]) -> list[str]:
    """Apply post-processing to a list of lines, dropping empties."""
    out = []
    for line in lines:
        fixed = postprocess(line)
        if fixed.strip():
            out.append(fixed)
    return out
