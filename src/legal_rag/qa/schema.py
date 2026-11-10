"""QA dataset schema shared by coverage checks and retrieval evaluation.

The loader is intentionally tolerant about input field names because the gold
file may be produced by different annotation rounds.  It is strict about the
one thing that matters: every record must have an id, a question, and at least
one explicit legal reference (document + article).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_ARTICLE_RE = re.compile(r"(\d+)")


class QASchemaError(ValueError):
    """Raised when a QA record cannot be interpreted."""


def normalize_article(value: Any) -> Optional[str]:
    """Return the bare article number as a string (``"Điều 35"`` -> ``"35"``)."""
    if value is None:
        return None
    match = _ARTICLE_RE.search(str(value))
    return match.group(1) if match else None


def normalize_clause(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    match = re.search(r"(\d+)", text)
    return match.group(1) if match else text


def normalize_point(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


@dataclass(frozen=True)
class RelevantRef:
    """One gold legal reference."""

    document_id: str
    article: str
    clause: Optional[str] = None
    point: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "document_id": self.document_id,
            "article": self.article,
        }
        if self.clause is not None:
            data["clause"] = self.clause
        if self.point is not None:
            data["point"] = self.point
        return data

    @classmethod
    def from_value(cls, value: Any, default_document_id: str) -> "RelevantRef":
        if isinstance(value, dict):
            document_id = str(
                value.get("document_id")
                or value.get("doc_id")
                or default_document_id
            )
            article = normalize_article(
                value.get("article")
                or value.get("article_id")
                or value.get("dieu")
            )
            if article is None:
                raise QASchemaError(f"reference has no article: {value!r}")
            return cls(
                document_id=document_id,
                article=article,
                clause=normalize_clause(
                    value.get("clause") or value.get("khoan")
                ),
                point=normalize_point(value.get("point") or value.get("diem")),
            )
        if isinstance(value, (int, float)):
            return cls(document_id=default_document_id, article=str(int(value)))
        text = str(value).strip()
        if ":" in text:
            document_id, article_text = text.split(":", 1)
            article = normalize_article(article_text)
            if article is None:
                raise QASchemaError(f"reference has no article: {value!r}")
            return cls(document_id=document_id.strip() or default_document_id, article=article)
        article = normalize_article(text)
        if article is None:
            raise QASchemaError(f"reference has no article: {value!r}")
        return cls(document_id=default_document_id, article=article)


@dataclass
class QARecord:
    id: str
    question: str
    relevant: List[RelevantRef] = field(default_factory=list)
    split: str = "test"
    question_type: Optional[str] = None
    mentions_article_number: Optional[bool] = None
    reference_answer: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "relevant": [ref.to_dict() for ref in self.relevant],
            "split": self.split,
            "question_type": self.question_type,
            "mentions_article_number": self.mentions_article_number,
            "reference_answer": self.reference_answer,
            "notes": self.notes,
        }


def _first(data: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return None


def parse_record(data: Dict[str, Any], line_number: int = 0) -> QARecord:
    question_id = _first(data, ("id", "qid", "question_id", "query_id"))
    question = _first(data, ("question", "query", "text"))
    if not question_id:
        raise QASchemaError(f"line {line_number}: record has no id")
    if not question:
        raise QASchemaError(f"line {line_number}: record {question_id!r} has no question")

    default_document_id = str(data.get("document_id") or "labor_code_2019")
    raw_refs = _first(
        data,
        (
            "relevant",
            "relevant_articles",
            "gold_articles",
            "articles",
            "relevant_article_ids",
            "gold",
        ),
    )
    if raw_refs is None:
        raise QASchemaError(
            f"line {line_number}: record {question_id!r} has no relevant references"
        )
    if isinstance(raw_refs, (str, int, float, dict)):
        raw_refs = [raw_refs]
    refs = [RelevantRef.from_value(value, default_document_id) for value in raw_refs]
    if not refs:
        raise QASchemaError(
            f"line {line_number}: record {question_id!r} has an empty reference list"
        )

    mentions = data.get("mentions_article_number")
    if mentions is None:
        mentions = bool(re.search(r"Điều\s*\d+", str(question), re.IGNORECASE))

    return QARecord(
        id=str(question_id),
        question=str(question),
        relevant=refs,
        split=str(data.get("split") or "test"),
        question_type=data.get("question_type") or data.get("type"),
        mentions_article_number=bool(mentions),
        reference_answer=data.get("reference_answer") or data.get("answer"),
        notes=data.get("notes"),
    )


def load_qa(path: Path | str, split: Optional[str] = None) -> List[QARecord]:
    path = Path(path)
    records: List[QARecord] = []
    seen_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QASchemaError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            record = parse_record(data, line_number)
            if record.id in seen_ids:
                raise QASchemaError(f"{path}:{line_number}: duplicate id {record.id!r}")
            seen_ids.add(record.id)
            if split is None or record.split == split:
                records.append(record)
    return records


def write_qa(records: List[QARecord], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return path
