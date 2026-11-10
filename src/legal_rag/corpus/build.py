"""Batch corpus build for every document in the registry.

The build is run-scoped: ``data/chunks/<run_id>/`` is created once and never
overwritten.  Every document gets an explicit status so a scanned PDF can never
be silently skipped.

Output contract (per run)::

    <chunks_dir>/<run_id>/all_<strategy>.jsonl   merged chunks
    <chunks_dir>/<run_id>/by_document/<id>.jsonl per-document chunks
    <chunks_dir>/<run_id>/chunk_stats.json
    <chunks_dir>/<run_id>/corpus_status.json
    <chunks_dir>/<run_id>/run_manifest.json
    <chunks_dir>/<run_id>/parse_reports/<id>.json
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from legal_rag.chunking.legal_aware import (
    FixedSizeChunker,
    LegalAwareChunker,
    save_jsonl,
)
from legal_rag.chunking.validator import validate_chunks
from legal_rag.corpus.audit import audit_pdf
from legal_rag.corpus.extractor import extract_pdf_lines
from legal_rag.corpus.ocr import (
    DEFAULT_DPI,
    OcrEngine,
    OcrUnavailable,
    get_engine,
    load_cached_pages,
    ocr_document,
)
from legal_rag.corpus.patches import apply_page_patches, load_patches
from legal_rag.corpus.registry import DocumentMeta, DocumentRegistry, load_registry
from legal_rag.parsing.parser import parse_legal_structure

STRATEGIES = ("legal_aware", "fixed_300", "fixed_500", "fixed_800")


@dataclass
class DocumentBuildResult:
    document_id: str
    document_name: str
    status: str
    pdf: Optional[str] = None
    pages: int = 0
    pages_with_text: int = 0
    pages_ocr_required: int = 0
    pages_ocr_used: int = 0
    pages_ocr_empty: int = 0
    corpus_chunks: int = 0
    final_chunks: int = 0
    articles_seen: int = 0
    missing_articles: int = 0
    appendix_chunks: int = 0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "status": self.status,
            "pdf": self.pdf,
            "pages": self.pages,
            "pages_with_text": self.pages_with_text,
            "pages_ocr_required": self.pages_ocr_required,
            "pages_ocr_used": self.pages_ocr_used,
            "pages_ocr_empty": self.pages_ocr_empty,
            "corpus_chunks": self.corpus_chunks,
            "final_chunks": self.final_chunks,
            "articles_seen": self.articles_seen,
            "missing_articles": self.missing_articles,
            "appendix_chunks": self.appendix_chunks,
            "warnings": self.warnings,
            "error": self.error,
        }


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


def _make_chunker(strategy: str, document: DocumentMeta):
    if strategy == "legal_aware":
        return LegalAwareChunker(document_id=document)
    if strategy.startswith("fixed_"):
        size = int(strategy.split("_", 1)[1])
        return FixedSizeChunker(chunk_size=size, overlap=50, document_id=document)
    raise ValueError(f"unknown strategy: {strategy!r}")


def _resolve_ocr_pages(
    document: DocumentMeta,
    audit,
    engine: Optional[OcrEngine],
    ocr_cache: Path,
    dpi: int,
    force_ocr: bool,
    ocr_workers: int = 1,
) -> tuple[Dict[int, List[str]], int, Optional[str]]:
    """Return OCR text for pages lacking a text layer.

    Returns ``(pages, ocr_used, error)``.  Cached pages are reused first so a
    re-run with no engine can still use previously OCR'd pages.
    """
    missing_pages = list(audit.pages_needing_ocr)
    if not missing_pages:
        return {}, 0, None

    pages: Dict[int, List[str]] = {}
    if engine is not None:
        cached = load_cached_pages(ocr_cache, document.document_id, engine, dpi=dpi)
        pending = [
            page for page in missing_pages
            if force_ocr or page not in cached or not cached[page]
        ]
        pages.update({page: cached[page] for page in missing_pages if page in cached})
        if pending:
            try:
                fresh = ocr_document(
                    document.pdf,
                    engine,
                    document_id=document.document_id,
                    pages=pending,
                    cache_root=ocr_cache,
                    dpi=dpi,
                    force=force_ocr,
                    workers=ocr_workers,
                )
                pages.update(fresh)
            except Exception as exc:  # pragma: no cover - engine dependent
                return pages, len(pages), f"{type(exc).__name__}: {exc}"
    else:
        # No engine available, but previously cached pages may still be reused.
        for engine_name, lang in (
            ("tesseract", "vie"),
            ("easyocr", "vi"),
            ("paddleocr", "vi"),
        ):
            try:
                cached_engine = get_engine(engine_name, lang=lang)
            except OcrUnavailable:
                continue
            cached = load_cached_pages(
                ocr_cache, document.document_id, cached_engine, dpi=dpi
            )
            pages.update({page: cached[page] for page in missing_pages if page in cached})
            if pages:
                break

    return pages, len(pages), None


def build_document(
    document: DocumentMeta,
    output_dir: Path,
    strategy: str = "legal_aware",
    patch_paths: Optional[Sequence[Path]] = None,
    engine: Optional[OcrEngine] = None,
    ocr_cache: Path = Path("data/ocr"),
    dpi: int = DEFAULT_DPI,
    force_ocr: bool = False,
    ocr_workers: int = 1,
) -> tuple[DocumentBuildResult, List[Dict[str, Any]]]:
    """Build one document; never raises for per-document failures."""
    result = DocumentBuildResult(
        document_id=document.document_id,
        document_name=document.document_name,
        status="unknown",
        pdf=str(document.pdf) if document.pdf else None,
    )

    if document.pdf is None or not Path(document.pdf).exists():
        result.status = "missing_pdf"
        result.error = "registry entry has no readable PDF"
        return result, []

    audit = audit_pdf(document.pdf, document)
    result.pages = audit.pages
    result.pages_with_text = audit.pages_with_text
    result.pages_ocr_required = len(audit.pages_needing_ocr)

    if audit.status == "source_corrupt":
        result.status = "source_corrupt"
        result.error = audit.error
        return result, []

    ocr_pages, ocr_used, ocr_error = _resolve_ocr_pages(
        document, audit, engine, Path(ocr_cache), dpi, force_ocr, ocr_workers
    )
    result.pages_ocr_used = ocr_used
    empty_ocr_pages = [
        page for page in audit.pages_needing_ocr if not ocr_pages.get(page)
    ]
    result.pages_ocr_empty = len(empty_ocr_pages)
    if empty_ocr_pages:
        result.warnings.append(f"OCR_EMPTY_PAGES: {empty_ocr_pages}")
    if ocr_error:
        result.warnings.append(f"OCR_ERROR: {ocr_error}")

    if audit.pages and len(audit.pages_needing_ocr) == audit.pages and not ocr_pages:
        result.status = "ocr_required"
        result.error = ocr_error or (
            "no embedded text layer and no OCR engine/cached pages available"
        )
        return result, []

    try:
        lines = extract_pdf_lines(document.pdf, ocr_pages=ocr_pages)
        patches = load_patches(list(patch_paths or []), document_id=document.document_id)
        lines, _ = apply_page_patches(lines, patches)
        parse_result = parse_legal_structure(lines, document)
    except Exception as exc:
        result.status = "parse_failed"
        result.error = f"{type(exc).__name__}: {exc}"
        result.warnings.append(traceback.format_exc(limit=3))
        return result, []

    result.corpus_chunks = len(parse_result.chunks)
    result.articles_seen = len(parse_result.articles_seen)
    result.missing_articles = len(parse_result.missing_articles)
    result.appendix_chunks = parse_result.appendix_chunks
    result.warnings.extend(parse_result.warnings)

    if not parse_result.chunks:
        result.status = "empty"
        result.error = "parser produced no chunks"
        return result, []

    chunker = _make_chunker(strategy, document)
    try:
        final_chunks = chunker.chunk(parse_result.chunks)
    except Exception as exc:
        result.status = "parse_failed"
        result.error = f"chunking failed: {type(exc).__name__}: {exc}"
        result.warnings.append(traceback.format_exc(limit=3))
        return result, []

    for chunk in final_chunks:
        chunk["retrieval_eligible"] = chunk.get("chunk_type") != "appendix"

    result.final_chunks = len(final_chunks)
    if result.articles_seen == 0:
        if result.appendix_chunks:
            result.status = "appendix_only"
        elif result.pages_ocr_required and (
            not ocr_pages or result.pages_ocr_empty
        ):
            result.status = (
                "ocr_required" if result.pages_with_text == 0 else "partial_text"
            )
        else:
            result.status = "empty"
    elif result.pages_ocr_required and (not ocr_pages or result.pages_ocr_empty):
        result.status = "partial_text"
    else:
        result.status = "ok"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(final_chunks, output_dir / f"{document.document_id}.jsonl")
    report_path = output_dir.parent / "parse_reports" / f"{document.document_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "document_id": document.document_id,
                "warnings": parse_result.warnings,
                "dropped_lines": parse_result.dropped_lines,
                "articles_seen": parse_result.articles_seen,
                "missing_articles": parse_result.missing_articles,
                "duplicate_article_numbers": parse_result.duplicate_article_numbers,
                "signature_lines": parse_result.signature_lines,
                "appendix_chunks": parse_result.appendix_chunks,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result, final_chunks


def run_corpus_build(
    run_id: str,
    strategy: str = "legal_aware",
    registry: Optional[DocumentRegistry] = None,
    document_ids: Optional[Iterable[str]] = None,
    chunks_dir: Path = Path("data/chunks"),
    patch_paths: Optional[Sequence[Path]] = None,
    ocr_engine: str = "auto",
    ocr_cache: Path = Path("data/ocr"),
    dpi: int = DEFAULT_DPI,
    force_ocr: bool = False,
    include_inactive: bool = True,
    ocr_workers: int = 1,
) -> Dict[str, Any]:
    """Build all selected documents into a fresh run directory."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected {STRATEGIES}")

    registry = registry or load_registry()
    documents = registry.select(
        document_ids=document_ids, include_inactive=include_inactive
    )
    if not documents:
        raise ValueError("no documents selected for build")

    run_dir = Path(chunks_dir) / run_id
    if run_dir.exists():
        raise FileExistsError(
            f"run directory already exists: {run_dir}; choose a new --run-id"
        )
    by_document_dir = run_dir / "by_document"
    by_document_dir.mkdir(parents=True, exist_ok=False)

    engine: Optional[OcrEngine] = None
    engine_error: Optional[str] = None
    try:
        engine = get_engine(ocr_engine)
    except OcrUnavailable as exc:
        engine_error = str(exc)

    results: List[DocumentBuildResult] = []
    all_chunks: List[Dict[str, Any]] = []
    total_documents = len(documents)
    for index, document in enumerate(documents, 1):
        result, chunks = build_document(
            document,
            by_document_dir,
            strategy=strategy,
            patch_paths=patch_paths,
            engine=engine,
            ocr_cache=ocr_cache,
            dpi=dpi,
            force_ocr=force_ocr,
            ocr_workers=ocr_workers,
        )
        results.append(result)
        for chunk in chunks:
            chunk["run_id"] = run_id
            chunk["strategy"] = strategy
        all_chunks.extend(chunks)
        print(
            f"[{index}/{total_documents}] {document.document_id}: "
            f"{result.status} pages={result.pages} ocr={result.pages_ocr_used} "
            f"chunks={result.final_chunks}",
            file=sys.stderr,
            flush=True,
        )

    all_chunks_path = run_dir / f"all_{strategy}.jsonl"
    save_jsonl(all_chunks, all_chunks_path)

    status_counts = Counter(result.status for result in results)
    chunk_type_counts = Counter(chunk.get("chunk_type") for chunk in all_chunks)
    status_path = run_dir / "corpus_status.json"
    status_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "strategy": strategy,
                "documents": [result.to_dict() for result in results],
                "status_counts": dict(status_counts),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    stats_path = run_dir / "chunk_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "strategy": strategy,
                "total_chunks": len(all_chunks),
                "retrieval_eligible": sum(
                    1 for chunk in all_chunks if chunk.get("retrieval_eligible")
                ),
                "chunk_types": dict(chunk_type_counts.most_common()),
                "by_document": {
                    result.document_id: result.final_chunks for result in results
                },
                "per_document": [
                    {
                        "document_id": result.document_id,
                        "status": result.status,
                        "chunks": result.final_chunks,
                        "articles_seen": result.articles_seen,
                    }
                    for result in results
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    validation = validate_chunks(all_chunks) if all_chunks else None
    manifest = {
        "run_id": run_id,
        "strategy": strategy,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "ocr_engine": engine.name if engine else None,
        "ocr_engine_error": engine_error,
        "ocr_psm": getattr(engine, "psm", None),
        "dpi": dpi,
        "ocr_workers": ocr_workers,
        "documents_requested": len(documents),
        "documents_built": sum(1 for result in results if result.final_chunks),
        "status_counts": dict(status_counts),
        "pdf_hashes": {
            result.document_id: _sha256(Path(result.pdf))
            for result in results
            if result.pdf and Path(result.pdf).exists()
        },
        "validation": (
            {
                "errors": validation.errors,
                "warnings": validation.warnings,
                "infos": validation.infos,
            }
            if validation
            else None
        ),
        "outputs": {
            "all_chunks": str(all_chunks_path),
            "corpus_status": str(status_path),
            "chunk_stats": str(stats_path),
            "by_document": str(by_document_dir),
        },
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "strategy": strategy,
        "documents": [result.to_dict() for result in results],
        "status_counts": dict(status_counts),
        "total_chunks": len(all_chunks),
        "all_chunks_path": all_chunks_path,
        "status_path": status_path,
        "stats_path": stats_path,
        "manifest_path": manifest_path,
        "validation": manifest["validation"],
    }
