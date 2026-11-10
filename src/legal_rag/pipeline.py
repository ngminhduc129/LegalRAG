"""Standard build pipeline: PDF -> (page patches) -> corpus -> legal-aware chunks -> validation.

All outputs are run-scoped (``<document_id>_<run_id>...``); a run id that
already has outputs is rejected instead of overwriting older results.
"""

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from legal_rag.chunking.legal_aware import LegalAwareChunker, save_csv, save_jsonl
from legal_rag.chunking.validator import validate_chunks
from legal_rag.corpus.extractor import extract_pdf_lines, save_corpus
from legal_rag.corpus.patches import apply_page_patches, load_patches
from legal_rag.corpus.registry import resolve_document
from legal_rag.parsing.parser import parse_legal_structure


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except Exception:
        return None


def run_pipeline(
    pdf_path: Path,
    run_id: str,
    document_id: str = "labor_code_2019",
    patch_paths: Optional[Sequence[Path]] = None,
    corpus_dir: Path = Path("data/processed"),
    chunks_dir: Path = Path("data/chunks"),
    do_validate: bool = True,
) -> Dict[str, Any]:
    """Run the standard pipeline and write run-scoped outputs.

    Outputs (never overwrite an older run):
      <corpus_dir>/<document_id>_<run_id>_corpus.jsonl (+ _preview.csv)
      <chunks_dir>/legal_aware_<run_id>/<document_id>_legal_aware.jsonl (+ .csv)
      <chunks_dir>/legal_aware_<run_id>/validation_report.txt
      <chunks_dir>/legal_aware_<run_id>/build_manifest.json
    """
    pdf_path = Path(pdf_path)
    corpus_dir = Path(corpus_dir)
    chunks_dir = Path(chunks_dir)
    document = resolve_document(document_id)

    corpus_jsonl = corpus_dir / f"{document_id}_{run_id}_corpus.jsonl"
    run_dir = chunks_dir / f"legal_aware_{run_id}"
    if corpus_jsonl.exists() or run_dir.exists():
        raise FileExistsError(
            f"run id {run_id!r} already has outputs ({corpus_jsonl} / {run_dir}); "
            "choose a new --run-id (old runs are never overwritten)"
        )

    empty_pages: List[int] = []

    def _note_empty(page_num: int, name: Optional[str]) -> None:
        empty_pages.append(page_num)

    raw_lines = extract_pdf_lines(
        pdf_path, on_empty_page=_note_empty, pdf_name=str(pdf_path)
    )
    patches = load_patches(list(patch_paths or []), document_id=document_id)
    lines, patch_reports = apply_page_patches(raw_lines, patches)

    parse_result = parse_legal_structure(lines, document)
    corpus_chunks = parse_result.chunks
    corpus_jsonl, corpus_csv = save_corpus(
        corpus_chunks, corpus_dir, f"{document_id}_{run_id}"
    )

    chunker = LegalAwareChunker(document_id=document)
    final_chunks = chunker.chunk(corpus_chunks)

    run_dir.mkdir(parents=True, exist_ok=True)
    chunks_jsonl = run_dir / f"{document_id}_legal_aware.jsonl"
    chunks_csv = run_dir / f"{document_id}_legal_aware.csv"
    save_jsonl(final_chunks, chunks_jsonl)
    save_csv(final_chunks, chunks_csv)

    parse_report = {
        "document_id": document.document_id,
        "document_name": document.document_name,
        "document_number": document.document_number,
        "warnings": parse_result.warnings,
        "empty_pages": parse_result.empty_pages,
        "dropped_lines": parse_result.dropped_lines,
        "articles_seen": parse_result.articles_seen,
        "missing_articles": parse_result.missing_articles,
        "duplicate_article_numbers": parse_result.duplicate_article_numbers,
        "signature_lines": parse_result.signature_lines,
        "appendix_chunks": parse_result.appendix_chunks,
    }
    parse_report_path = run_dir / "parse_report.json"
    parse_report_path.write_text(
        json.dumps(parse_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    validation = None
    if do_validate:
        report = validate_chunks(final_chunks)
        validation_path = run_dir / "validation_report.txt"
        validation_path.write_text(report.summary(), encoding="utf-8")
        validation = {
            "errors": report.errors,
            "warnings": report.warnings,
            "infos": report.infos,
            "report": str(validation_path),
        }

    for report in patch_reports:
        if report.get("path"):
            report["sha256"] = _sha256(Path(report["path"]))

    manifest = {
        "run_id": run_id,
        "document_id": document.document_id,
        "document_name": document.document_name,
        "document_number": document.document_number,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "pdf": {
            "path": str(pdf_path),
            "sha256": _sha256(pdf_path),
            "raw_lines": len(raw_lines),
            "empty_pages": empty_pages,
        },
        "patches": patch_reports,
        "counts": {
            "lines_after_patches": len(lines),
            "corpus_chunks": len(corpus_chunks),
            "final_chunks": len(final_chunks),
        },
        "parse_report": str(parse_report_path),
        "validation": validation,
        "outputs": {
            "corpus_jsonl": str(corpus_jsonl),
            "corpus_csv": str(corpus_csv),
            "chunks_jsonl": str(chunks_jsonl),
            "chunks_csv": str(chunks_csv),
        },
    }
    manifest_path = run_dir / "build_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "patch_reports": patch_reports,
        "raw_lines": len(raw_lines),
        "line_count": len(lines),
        "corpus_chunks": len(corpus_chunks),
        "final_chunks": len(final_chunks),
        "chunk_types": dict(Counter(c["chunk_type"] for c in final_chunks).most_common()),
        "articles_seen": len(parse_result.articles_seen),
        "missing_articles": parse_result.missing_articles,
        "validation": validation,
        "manifest_path": manifest_path,
        "parse_report_path": parse_report_path,
        "corpus_jsonl": corpus_jsonl,
        "chunks_jsonl": chunks_jsonl,
    }
