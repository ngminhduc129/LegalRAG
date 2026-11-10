"""
Structure identity registry and resolution for legal chunking.

Provides:
- ``StructureRegistry``: indexes legal structures by their parser-generated
  ``structure_instance_id`` values, stores parent relationships, and preserves
  ``structure_role`` and ``operation_id``.  Never silently generates
  replacement IDs for missing parser IDs.
- ``IdentityResolver``: resolves chunk-level ``parent_id`` from
  ``parent_instance_id`` without conflating chunk identity with legal
  structure identity.
- ``OperationDetector``: assigns ``operation_id`` only for amendment patterns
  the parser can confidently identify.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StructureRecord:
    """Record for one legal structure occurrence."""

    instance_id: str
    parent_instance_id: Optional[str]
    structure_role: str
    operation_id: Optional[str]
    chunk_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "parent_instance_id": self.parent_instance_id,
            "structure_role": self.structure_role,
            "operation_id": self.operation_id,
            "chunk_ids": list(self.chunk_ids),
        }


class StructureRegistry:
    """Indexes structures by ``structure_instance_id``.

    Duplicate instance IDs are detected and warned; the first authoritative
    parser record wins and is never overwritten by a later chunker-generated
    fallback.
    """

    def __init__(self) -> None:
        self._by_id: Dict[str, StructureRecord] = {}
        self._duplicate_warnings: List[str] = []
        self._unresolved_parent_warnings: List[str] = []

    @property
    def records(self) -> Dict[str, StructureRecord]:
        return self._by_id

    def register(
        self,
        instance_id: str,
        parent_instance_id: Optional[str] = None,
        structure_role: str = "main",
        operation_id: Optional[str] = None,
    ) -> StructureRecord:
        """Register a structure.  Returns the existing record on duplicate ID."""
        if instance_id in self._by_id:
            self._duplicate_warnings.append(
                f"Duplicate structure_instance_id: {instance_id}"
            )
            return self._by_id[instance_id]

        record = StructureRecord(
            instance_id=instance_id,
            parent_instance_id=parent_instance_id,
            structure_role=structure_role,
            operation_id=operation_id,
        )
        self._by_id[instance_id] = record
        return record

    def add_chunk_id(self, instance_id: str, chunk_id: str) -> None:
        record = self._by_id.get(instance_id)
        if record is not None and chunk_id not in record.chunk_ids:
            record.chunk_ids.append(chunk_id)

    def resolve_parent(self, instance_id: str) -> Optional[StructureRecord]:
        record = self._by_id.get(instance_id)
        if record is None:
            return None
        if record.parent_instance_id is None:
            return None
        parent = self._by_id.get(record.parent_instance_id)
        if parent is None:
            self._unresolved_parent_warnings.append(
                f"Unresolved parent reference: {instance_id} -> "
                f"{record.parent_instance_id}"
            )
        return parent

    def chunk_ids_for(self, instance_id: str) -> List[str]:
        record = self._by_id.get(instance_id)
        if record is None:
            return []
        return list(record.chunk_ids)

    def warnings(self) -> List[str]:
        return list(self._duplicate_warnings) + list(self._unresolved_parent_warnings)


# Natural container chunk types: when a parent structure emits multiple chunks,
# prefer the chunk whose type is the natural container for the child type.
_CONTAINER_PREFERENCE: Dict[str, set[str]] = {
    "point": {"clause_intro", "clause", "article_intro", "article"},
    "clause": {"article_intro", "article", "clause_intro"},
    "clause_intro": {"article_intro", "article", "section_intro", "chapter_intro"},
    "article_intro": {"section_intro", "chapter_intro"},
}


class IdentityResolver:
    """Resolves chunk-level ``parent_id`` from ``parent_instance_id``.

    The mapping policy (documented for audit):
    - ``parent_id`` is distinct from ``parent_instance_id``.
    - For a ``point``: parent = the output ``chunk_id`` of the structure that
      owns its ``parent_instance_id``.
    - When one parent structure produces multiple chunks, prefer the natural
      container chunk type; otherwise use the first chunk of that structure.
    - Never produce cross-document parent references.
    """

    def __init__(self, registry: StructureRegistry) -> None:
        self.registry = registry

    def resolve_parent_chunk_id(
        self,
        instance_id: str,
        child_chunk_type: str,
        document_id: str,
        all_chunks: List[Dict[str, Any]],
    ) -> Optional[str]:
        record = self.registry.records.get(instance_id)
        if record is None:
            return None

        parent_record = self.registry.resolve_parent(instance_id)
        if parent_record is None:
            return None

        parent_chunk_ids = self.registry.chunk_ids_for(parent_record.instance_id)
        if not parent_chunk_ids:
            return None

        preferred = _CONTAINER_PREFERENCE.get(child_chunk_type, set())

        # First pass: prefer a natural container chunk within the same document.
        for chunk in all_chunks:
            cid = chunk.get("chunk_id")
            if (
                cid in parent_chunk_ids
                and chunk.get("document_id") == document_id
                and chunk.get("chunk_type") in preferred
            ):
                return cid

        # Second pass: first chunk of the parent structure within the document.
        for chunk in all_chunks:
            cid = chunk.get("chunk_id")
            if cid in parent_chunk_ids and chunk.get("document_id") == document_id:
                return cid

        return None


class OperationDetector:
    """Detects confidently-supported amendment operations.

    Only the pattern ``Sửa đổi Điều X như sau`` followed by the corresponding
    replacement heading is currently supported.  Other amendment forms are left
    as ``operation_id = None`` with a warning.
    """

    AMENDMENT_INTRO_RE = re.compile(
        r"SỬA\s+ĐỔI\s+ĐIỀU\s+(\d+)\s+NHƯ\s+SAU", re.IGNORECASE
    )

    def __init__(self) -> None:
        self._counter: int = 0
        self._operation_ids: Dict[str, str] = {}

    def detect_operation(self, text: str) -> Optional[str]:
        """Return ``op_N`` for a confidently detected amendment intro, else None."""
        match = self.AMENDMENT_INTRO_RE.search(text or "")
        if not match:
            return None
        article_number = match.group(1)
        if article_number in self._operation_ids:
            return self._operation_ids[article_number]
        self._counter += 1
        op_id = f"op{self._counter}"
        self._operation_ids[article_number] = op_id
        return op_id

    @property
    def operation_ids(self) -> Dict[str, str]:
        return dict(self._operation_ids)
