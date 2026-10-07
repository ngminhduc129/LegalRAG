"""Chunk cleaning utilities."""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


# =========================================================
# RECOGNITION FUNCTIONS
# =========================================================

def is_page_number(line: str) -> bool:
    """Check if a line is only a page number."""
    line = line.strip()
    return bool(re.fullmatch(r"\d{1,3}", line))


def is_superscript_footnote(text: str) -> bool:
    """Check for Unicode superscript footnote numbers."""
    return bool(re.search(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+$", text))


def remove_superscript_footnote(text: str) -> str:
    """Remove superscript footnote numbers from end of string."""
    new_text = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+$", "", text).rstrip()
    return new_text


def remove_known_footnote_number(text: str) -> str:
    """
    Handle case where PDF converts superscript footnote to regular number.
    Only applies to known headings.
    """
    known_headings = [
        "ĐIỀU KHOẢN THI HÀNH",
        "ĐIỀU KHOẢN THI HÀNH.",
        "ĐIỀU KHOẢN CHUYỂN TIẾP",
    ]

    stripped = text.strip()

    for heading in known_headings:
        pattern = (
            r"^"
            + re.escape(heading.rstrip("."))
            + r"\s*\d+$"
        )
        if re.fullmatch(pattern, stripped, flags=re.IGNORECASE):
            return heading.rstrip(".")

    return text


def clean_metadata_field(text: str) -> str:
    """Clean metadata field (chapter_title, article_title, etc.)"""
    if not text:
        return text
    # Remove known heading footnotes
    text = remove_known_footnote_number(text)
    # Remove superscript footnotes
    text = remove_superscript_footnote(text)
    return text


# =========================================================
# CONTENT CLEANING
# =========================================================

def is_suspicious_ending(content: str) -> bool:
    """
    Check if last line is only a number (page number or footnote).
    Does NOT flag legitimate cases like "khoản 2", "Điều 36", "02 tháng", "15 ngày", "2026".
    """
    lines = content.strip().splitlines()
    if not lines:
        return False
    last_line = lines[-1].strip()
    return bool(re.fullmatch(r"\d{1,3}", last_line))


def clean_content(content: str) -> Tuple[str, List[str], List[str]]:
    """
    Clean chunk content.

    Returns:
        cleaned_content, actions, suspicious
    """
    actions = []
    suspicious = []

    if not content:
        return "", actions, suspicious

    # Normalize line endings
    content = content.replace("\r\n", "\n")
    content = content.replace("\r", "\n")

    # Split lines
    raw_lines = content.split("\n")
    cleaned_lines = []

    for line in raw_lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # 1. PAGE NUMBER
        if is_page_number(line):
            actions.append(f"REMOVE_PAGE_NUMBER:{line}")
            continue

        # 2. SUPERSCRIPT FOOTNOTE
        if is_superscript_footnote(line):
            new_line = remove_superscript_footnote(line)
            if new_line != line:
                actions.append(f"REMOVE_SUPERSCRIPT_FOOTNOTE:{line} -> {new_line}")
                line = new_line

        # 3. FOOTNOTE AS REGULAR NUMBER
        new_line = remove_known_footnote_number(line)
        if new_line != line:
            actions.append(f"REMOVE_HEADING_FOOTNOTE:{line} -> {new_line}")
            line = new_line

        # 4. KEEP NON-EMPTY
        if line:
            cleaned_lines.append(line)

    # Join back
    cleaned = "\n".join(cleaned_lines)

    # Normalize spaces
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # Max 2 consecutive newlines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    cleaned = cleaned.strip()

    # =====================================================
    # SUSPICIOUS CHECKS
    # =====================================================

    # Remaining page numbers
    for line in cleaned.split("\n"):
        if is_page_number(line):
            suspicious.append(f"PAGE_NUMBER_REMAIN:{line}")

    # Chunk ends with standalone number
    if is_suspicious_ending(cleaned):
        suspicious.append("ENDS_WITH_NUMBER")

    # Remaining superscript footnote
    if re.search(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+$", cleaned):
        suspicious.append("POSSIBLE_FOOTNOTE")

    return cleaned, actions, suspicious


# =========================================================
# CLEAN A CHUNK
# =========================================================

def clean_chunk(chunk: Dict) -> Dict:
    """Clean a single chunk."""
    new_chunk = dict(chunk)

    # Clean metadata fields
    for field in ["chapter_title", "article_title"]:
        if new_chunk.get(field):
            new_chunk[field] = clean_metadata_field(new_chunk[field])

    original_content = chunk.get("content", "")
    cleaned_content, actions, suspicious = clean_content(original_content)

    new_chunk["content"] = cleaned_content

    if actions:
        new_chunk["cleaned"] = True
        new_chunk["clean_actions"] = actions
    else:
        new_chunk["cleaned"] = False
        new_chunk["clean_actions"] = []

    if suspicious:
        new_chunk["needs_review"] = True
        new_chunk["review_reasons"] = suspicious
    else:
        new_chunk["needs_review"] = False
        new_chunk["review_reasons"] = []

    return new_chunk


# =========================================================
# I/O
# =========================================================

def load_chunks(path: Path) -> List[Dict]:
    """Load chunks from JSONL file."""
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON error at line {line_number}: {e}")
    return chunks


def save_chunks(chunks: List[Dict], path: Path):
    """Save chunks to JSONL file."""
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")