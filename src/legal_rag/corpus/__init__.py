"""Corpus extraction and processing.

Heavy modules are imported lazily: ``corpus.registry`` is imported by the parser,
and the extractor imports the parser, so eager imports here would create a
circular import.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "extract_pdf_lines",
    "parse_corpus",
    "save_corpus",
    "DocumentMeta",
    "DocumentRegistry",
    "load_registry",
    "resolve_document",
]


def __getattr__(name: str) -> Any:
    if name in {"extract_pdf_lines", "parse_corpus", "save_corpus"}:
        from legal_rag.corpus import extractor

        return getattr(extractor, name)
    if name in {
        "DocumentMeta",
        "DocumentRegistry",
        "load_registry",
        "resolve_document",
    }:
        from legal_rag.corpus import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
