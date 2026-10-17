"""Unit tests for page patches (broken text-layer pages)."""

import json
from pathlib import Path

import pytest

from legal_rag.corpus.patches import (
    PatchError,
    apply_page_patches,
    load_patches,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_PATCH = REPO_ROOT / "data" / "ocr" / "patches" / "labor_code_2019_page86.json"


def _line(text: str, page: int) -> dict:
    return {"text": text, "page": page}


def test_replace_page_between_anchors():
    lines = [
        _line("trang 1 ... quy định của pháp luật về", 1),
        _line("rác trang 2 a", 2),
        _line("rác trang 2 b", 2),
        _line("rác trang 2 c", 2),
        _line("trang 3", 3),
    ]
    patch = {
        "page": 2,
        "apply": {
            "anchors": {
                "prev_line_endswith": "quy định của pháp luật về",
                "first_line_startswith": "lao động mà",
                "last_line_endswith": "trong Bộ luật này.",
            }
        },
        "lines": ["lao động mà ...", "và kết thúc trong Bộ luật này."],
    }

    out, reports = apply_page_patches(lines, [patch])

    assert [line["text"] for line in out] == [
        "trang 1 ... quy định của pháp luật về",
        "lao động mà ...",
        "và kết thúc trong Bộ luật này.",
        "trang 3",
    ]
    assert [line["page"] for line in out] == [1, 2, 2, 3]
    assert reports[0]["replaced_lines"] == 3
    assert reports[0]["inserted_lines"] == 2


def test_bad_anchor_raises():
    lines = [_line("không khớp", 1), _line("rác", 2)]
    patch = {
        "page": 2,
        "apply": {"anchors": {"prev_line_endswith": "quy định của pháp luật về"}},
        "lines": ["mới"],
    }
    with pytest.raises(PatchError):
        apply_page_patches(lines, [patch])


def test_missing_page_raises():
    patch = {"page": 5, "lines": ["mới"]}
    with pytest.raises(PatchError):
        apply_page_patches([_line("x", 1)], [patch])


def test_load_patches_directory_filters_document(tmp_path):
    (tmp_path / "a_page1.json").write_text(
        json.dumps({"page": 1, "document_id": "doc_a", "lines": ["x"]}),
        encoding="utf-8",
    )
    (tmp_path / "b_page2.json").write_text(
        json.dumps({"page": 2, "document_id": "doc_b", "lines": ["y"]}),
        encoding="utf-8",
    )
    (tmp_path / "c_page3.json").write_text(
        json.dumps({"page": 3, "lines": ["z"]}),
        encoding="utf-8",
    )

    assert [p["page"] for p in load_patches([tmp_path])] == [1, 2, 3]
    doc_a = load_patches([tmp_path], document_id="doc_a")
    assert [p["page"] for p in doc_a] == [1, 3]  # document-less patches load for all


def test_page86_patch_matches_ocr_evidence(monkeypatch):
    if not REAL_PATCH.exists():
        pytest.skip("page-86 patch not available")
    monkeypatch.chdir(REPO_ROOT)  # OCR evidence path is repo-relative

    patches = load_patches([REAL_PATCH], document_id="labor_code_2019")
    assert len(patches) == 1
    patch = patches[0]
    assert patch["page"] == 86
    assert len(patch["lines"]) == 27

    stream = (
        [_line("... theo quy định của pháp luật về", 85)]
        + [_line(f"rác {i}", 86) for i in range(41)]
        + [_line("trang 87", 87)]
    )
    out, reports = apply_page_patches(stream, patches)

    assert out[1]["text"].startswith("lao động mà")
    assert out[-2]["text"].endswith("trong Bộ luật này.")
    assert reports[0]["replaced_lines"] == 41
    assert reports[0]["ocr_coverage"] >= 0.82
