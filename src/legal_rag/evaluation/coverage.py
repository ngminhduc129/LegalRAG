"""Gold coverage: can every test reference be found in the built chunk corpus?

This is the check that answers "chunk output must be sufficient for the test
set".  If a gold article has no retrieval-eligible chunk, no retriever can ever
return it; that is a corpus problem, not a retrieval problem.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from legal_rag.qa.schema import QARecord, RelevantRef, load_qa, normalize_article, normalize_clause


@dataclass
class CoverageResult:
    total_queries: int = 0
    queries_with_all_refs: int = 0
    missing_refs: List[Dict[str, Any]] = field(default_factory=list)
    total_refs: int = 0
    covered_refs: int = 0

    @property
    def coverage(self) -> float:
        if not self.total_refs:
            return 0.0
        return self.covered_refs / self.total_refs

    @property
    def query_coverage(self) -> float:
        if not self.total_queries:
            return 0.0
        return self.queries_with_all_refs / self.total_queries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "queries_with_all_refs": self.queries_with_all_refs,
            "query_coverage": round(self.query_coverage, 4),
            "total_refs": self.total_refs,
            "covered_refs": self.covered_refs,
            "ref_coverage": round(self.coverage, 4),
            "missing_refs": self.missing_refs,
        }


def chunk_key(chunk: Dict[str, Any]) -> Optional[tuple]:
    """(document_id, article) for a chunk, or None if it has no article."""
    article = normalize_article(chunk.get("article"))
    if article is None:
        return None
    return (chunk.get("document_id") or "unknown", article)


def chunk_clause_key(chunk: Dict[str, Any]) -> Optional[tuple]:
    key = chunk_key(chunk)
    if key is None:
        return None
    return (*key, normalize_clause(chunk.get("clause")))


def _ref_in_index(
    ref: RelevantRef,
    article_keys: set,
    clause_keys: set,
    level: str,
) -> bool:
    if level == "clause" and ref.clause is not None:
        return (ref.document_id, ref.article, ref.clause) in clause_keys
    return (ref.document_id, ref.article) in article_keys


def check_coverage(
    records: Iterable[QARecord],
    chunks: Iterable[Dict[str, Any]],
    level: str = "article",
    retrieval_eligible_only: bool = True,
) -> CoverageResult:
    article_keys = set()
    clause_keys = set()
    for chunk in chunks:
        if retrieval_eligible_only and chunk.get("retrieval_eligible") is False:
            continue
        key = chunk_key(chunk)
        if key is not None:
            article_keys.add(key)
        clause_key = chunk_clause_key(chunk)
        if clause_key is not None:
            clause_keys.add(clause_key)

    result = CoverageResult()
    for record in records:
        result.total_queries += 1
        all_covered = True
        for ref in record.relevant:
            result.total_refs += 1
            if _ref_in_index(ref, article_keys, clause_keys, level):
                result.covered_refs += 1
            else:
                all_covered = False
                result.missing_refs.append(
                    {
                        "query_id": record.id,
                        "document_id": ref.document_id,
                        "article": ref.article,
                        "clause": ref.clause,
                        "level": level,
                    }
                )
        if all_covered:
            result.queries_with_all_refs += 1
    return result


def load_chunks(path: Path | str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def coverage_report(
    qa_path: Path | str,
    chunks_path: Path | str,
    split: Optional[str] = None,
    level: str = "article",
) -> Dict[str, Any]:
    records = load_qa(qa_path, split=split)
    chunks = load_chunks(chunks_path)
    result = check_coverage(records, chunks, level=level)
    return {
        "qa_path": str(qa_path),
        "chunks_path": str(chunks_path),
        "split": split,
        "level": level,
        "chunks": len(chunks),
        **result.to_dict(),
    }
