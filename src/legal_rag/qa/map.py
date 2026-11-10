"""Map gold QA references to concrete chunk ids in a built corpus.

This is the bridge between annotation and retrieval evaluation: a question is
only usable for a run if every gold reference resolves to at least one
retrieval-eligible chunk.  The mapping never guesses: an unresolved reference
is reported as missing instead of being dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from legal_rag.qa.schema import (
    QARecord,
    RelevantRef,
    load_qa,
    normalize_article,
    normalize_clause,
)


@dataclass
class MappedQuestion:
    id: str
    question: str
    gold_chunk_ids: List[str] = field(default_factory=list)
    matched_refs: List[Dict[str, Any]] = field(default_factory=list)
    missing_refs: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return not self.missing_refs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "gold_chunk_ids": self.gold_chunk_ids,
            "matched_refs": self.matched_refs,
            "missing_refs": self.missing_refs,
            "is_complete": self.is_complete,
        }


def _chunk_key(chunk: Dict[str, Any], level: str) -> Optional[tuple]:
    article = normalize_article(chunk.get("article"))
    if article is None:
        return None
    document_id = chunk.get("document_id") or "unknown"
    if level == "clause":
        clause = normalize_clause(chunk.get("clause"))
        if clause is not None:
            return (document_id, article, clause)
    return (document_id, article)


def _ref_key(ref: RelevantRef, level: str) -> tuple:
    if level == "clause" and ref.clause is not None:
        return (ref.document_id, ref.article, ref.clause)
    return (ref.document_id, ref.article)


def map_gold_to_chunks(
    records: Iterable[QARecord],
    chunks: Iterable[Dict[str, Any]],
    level: str = "article",
    retrieval_eligible_only: bool = True,
) -> List[MappedQuestion]:
    """Resolve each gold reference to chunk ids at the requested level."""
    if level not in ("article", "clause"):
        raise ValueError(f"unknown level {level!r}; expected article|clause")

    index: Dict[tuple, List[str]] = {}
    for chunk in chunks:
        if retrieval_eligible_only and chunk.get("retrieval_eligible") is False:
            continue
        key = _chunk_key(chunk, level)
        if key is None:
            continue
        chunk_id = chunk.get("chunk_id")
        if chunk_id:
            index.setdefault(key, []).append(str(chunk_id))

    mapped: List[MappedQuestion] = []
    for record in records:
        entry = MappedQuestion(id=record.id, question=record.question)
        seen = set()
        for ref in record.relevant:
            key = _ref_key(ref, level)
            chunk_ids = index.get(key, [])
            if chunk_ids:
                entry.matched_refs.append(
                    {**ref.to_dict(), "level": level, "chunk_ids": sorted(chunk_ids)}
                )
                for chunk_id in chunk_ids:
                    if chunk_id not in seen:
                        seen.add(chunk_id)
                        entry.gold_chunk_ids.append(chunk_id)
            else:
                entry.missing_refs.append({**ref.to_dict(), "level": level})
        mapped.append(entry)
    return mapped


def load_chunks(path: Path | str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def write_mapping(mapped: Sequence[MappedQuestion], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in mapped:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
    return path


def mapping_summary(mapped: Sequence[MappedQuestion]) -> Dict[str, Any]:
    total = len(mapped)
    complete = sum(1 for entry in mapped if entry.is_complete)
    total_refs = sum(len(entry.matched_refs) + len(entry.missing_refs) for entry in mapped)
    matched_refs = sum(len(entry.matched_refs) for entry in mapped)
    return {
        "total_questions": total,
        "complete_questions": complete,
        "incomplete_questions": total - complete,
        "question_coverage": round(complete / total, 4) if total else 0.0,
        "total_refs": total_refs,
        "matched_refs": matched_refs,
        "ref_coverage": round(matched_refs / total_refs, 4) if total_refs else 0.0,
        "total_gold_chunk_ids": sum(len(entry.gold_chunk_ids) for entry in mapped),
    }


def map_report(
    qa_path: Path | str,
    chunks_path: Path | str,
    level: str = "article",
    split: Optional[str] = None,
    output_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    records = load_qa(qa_path, split=split)
    chunks = load_chunks(chunks_path)
    mapped = map_gold_to_chunks(records, chunks, level=level)
    if output_path is not None:
        write_mapping(mapped, output_path)
    return {
        "qa_path": str(qa_path),
        "chunks_path": str(chunks_path),
        "split": split,
        "level": level,
        **mapping_summary(mapped),
        "questions": [entry.to_dict() for entry in mapped],
    }
