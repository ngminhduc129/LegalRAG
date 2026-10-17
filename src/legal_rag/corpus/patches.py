"""Page patches: replace broken PDF text-layer pages with verified content.

Some pages of a source PDF can be raster scans whose hidden text layer is
unusable (e.g. lossy non-Vietnamese OCR noise). A patch file describes the
verified replacement content for one page, plus safety anchors and an
optional cross-check against the raw OCR evidence.

Patch file schema (JSON):
    {
      "version": 1,
      "document_id": "labor_code_2019",        # optional; filters loading
      "page": 86,                              # 1-based PDF page to replace
      "lines": ["...", "..."],                 # verified content, one per line
      "apply": {
        "anchors": {                           # optional safety checks
          "prev_line_endswith": "...",         # last kept line before the page
          "first_line_startswith": "...",      # first replacement line
          "last_line_endswith": "..."          # last replacement line
        },
        "ocr_evidence": {                      # optional cross-check
          "path": "data/ocr/xxx.json",
          "page_key": "86",
          "min_lcs_coverage": 0.82
        }
      }
    }
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from legal_rag.parsing.classifier import normalize_vi


class PatchError(RuntimeError):
    """Raised when a page patch cannot be applied safely."""


def _stripped_blob(texts: Iterable[str]) -> str:
    """Diacritics-stripped, letters-only blob used for subsequence matching."""
    return normalize_vi(" ".join(texts))


def _lcs_len(a: str, b: str) -> int:
    """Length of the longest common subsequence of two strings."""
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b):
            if ca == cb:
                cur[j + 1] = prev[j] + 1
            else:
                cur[j + 1] = prev[j + 1] if prev[j + 1] >= cur[j] else cur[j]
        prev = cur
    return prev[-1]


def lcs_coverage(verified_lines: Sequence[str], ocr_texts: Sequence[str]) -> float:
    """Fraction of OCR letters recovered (in order) by the verified lines.

    Scan OCR tends to drop whole accented letters, so subsequence containment
    (LCS) is the meaningful metric, not positional string similarity.
    """
    a = _stripped_blob(verified_lines)
    b = _stripped_blob(ocr_texts)
    if not a or not b:
        return 0.0
    return _lcs_len(a, b) / len(b)


def load_patches(
    paths: Sequence[Path], document_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Load patch files from JSON files and/or directories (sorted, deduped).

    Patches whose ``document_id`` is set and differs from ``document_id`` are
    skipped, so one shared patch directory can serve several documents.
    """
    files: List[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"patch path not found: {path}")

    patches: List[Dict[str, Any]] = []
    for file in dict.fromkeys(files):
        data = json.loads(file.read_text(encoding="utf-8"))
        if document_id is not None and data.get("document_id") not in (None, document_id):
            continue
        data["_path"] = str(file)
        patches.append(data)
    return patches


def _check_ocr_evidence(
    page: int, lines: Sequence[str], evidence: Dict[str, Any]
) -> Optional[float]:
    """LCS cross-check of the verified lines against the raw OCR evidence."""
    path = Path(evidence["path"])
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get(str(evidence.get("page_key", page)), [])
    ocr_texts = []
    for item in raw:
        text = item.get("text", "") if isinstance(item, dict) else str(item)
        text = text.strip()
        if text and not re.fullmatch(r"\d{1,3}", text):
            ocr_texts.append(text)
    coverage = lcs_coverage(lines, ocr_texts)
    minimum = float(evidence.get("min_lcs_coverage", 0.8))
    if coverage < minimum:
        raise PatchError(
            f"page {page}: OCR cross-check coverage {coverage:.4f} < {minimum} "
            f"(evidence: {path})"
        )
    return coverage


def apply_page_patches(
    lines: List[Dict[str, Any]], patches: Sequence[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Replace every extracted line of a patched page with the patch content.

    Returns ``(new_lines, reports)`` where each report records what was
    replaced and the OCR cross-check coverage (if configured).
    """
    reports: List[Dict[str, Any]] = []
    for patch in patches:
        lines, report = _apply_one(lines, patch)
        reports.append(report)
    return lines, reports


def _apply_one(
    lines: List[Dict[str, Any]], patch: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    page = int(patch["page"])
    new_texts = [str(t).strip() for t in patch.get("lines", []) if str(t).strip()]
    if not new_texts:
        raise PatchError(f"page {page}: patch has no content lines")

    idx = [i for i, line in enumerate(lines) if line.get("page") == page]
    if not idx:
        raise PatchError(f"page {page}: no extracted lines on this page")
    first, last = idx[0], idx[-1]
    if len(idx) != last - first + 1:
        raise PatchError(f"page {page}: extracted lines are not contiguous")

    anchors = (patch.get("apply") or {}).get("anchors") or {}
    prev_end = anchors.get("prev_line_endswith")
    if prev_end:
        prev_text = lines[first - 1]["text"].rstrip() if first > 0 else ""
        if not prev_text.endswith(prev_end):
            raise PatchError(
                f"page {page}: previous line does not end with {prev_end!r} "
                f"(got ...{prev_text[-60:]!r})"
            )
    first_anchor = anchors.get("first_line_startswith")
    if first_anchor and not new_texts[0].startswith(first_anchor):
        raise PatchError(
            f"page {page}: first patch line does not start with {first_anchor!r}"
        )
    last_anchor = anchors.get("last_line_endswith")
    if last_anchor and not new_texts[-1].endswith(last_anchor):
        raise PatchError(
            f"page {page}: last patch line does not end with {last_anchor!r}"
        )

    report: Dict[str, Any] = {
        "page": page,
        "path": patch.get("_path"),
        "replaced_lines": last - first + 1,
        "inserted_lines": len(new_texts),
    }
    evidence = (patch.get("apply") or {}).get("ocr_evidence")
    if evidence:
        report["ocr_coverage"] = _check_ocr_evidence(page, new_texts, evidence)

    patched = (
        lines[:first]
        + [{"text": text, "page": page} for text in new_texts]
        + lines[last + 1:]
    )
    return patched, report
