"""Document registry and per-document metadata.

The registry is the single source of truth for document identity.  Parser and
chunker code must never fall back to a hard-coded document name/number: doing
so silently mislabels every document that is not the consolidated Labour Code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

VALID_STATUS = {"current", "amended", "repealed", "unknown"}
VALID_PROFILES = {"bo_luat", "nghi_dinh", "thong_tu", "quyet_dinh", "unknown"}

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "configs" / "corpus" / "documents.yaml"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class RegistryError(ValueError):
    """Raised when the document registry is invalid."""


@dataclass(frozen=True)
class DocumentMeta:
    """Immutable metadata for one legal document."""

    document_id: str
    document_name: str
    document_number: str
    pdf: Optional[Path] = None
    doc_type: str = "unknown"
    issuer: Optional[str] = None
    effective_from: Optional[str] = None
    effective_to: Optional[str] = None
    status: str = "unknown"
    expected_articles: Optional[int] = None
    profile: str = "unknown"
    signature_markers: tuple[str, ...] = ("noinhan",)
    appendix_markers: tuple[str, ...] = ("phuluc", "mauso", "bieuso")
    include: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.document_id:
            raise RegistryError("document_id must not be empty")
        if self.status not in VALID_STATUS:
            raise RegistryError(
                f"{self.document_id}: invalid status {self.status!r}; "
                f"expected one of {sorted(VALID_STATUS)}"
            )
        if self.profile not in VALID_PROFILES:
            raise RegistryError(
                f"{self.document_id}: invalid profile {self.profile!r}; "
                f"expected one of {sorted(VALID_PROFILES)}"
            )
        for field_name in ("effective_from", "effective_to"):
            value = getattr(self, field_name)
            if value is not None and not _DATE_RE.match(value):
                raise RegistryError(
                    f"{self.document_id}: {field_name} must be YYYY-MM-DD, got {value!r}"
                )
        if (
            self.effective_from
            and self.effective_to
            and self.effective_from > self.effective_to
        ):
            raise RegistryError(
                f"{self.document_id}: effective_from is after effective_to"
            )
        if self.expected_articles is not None and self.expected_articles < 0:
            raise RegistryError(
                f"{self.document_id}: expected_articles must be non-negative"
            )

    def as_metadata(self) -> Dict[str, Any]:
        """Fields copied into every corpus/chunk record."""
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "document_number": self.document_number,
            "document_type": self.doc_type,
            "issuer": self.issuer,
            "effective_from": self.effective_from,
            "effective_to": self.effective_to,
            "status": self.status,
        }

    def is_current(self) -> bool:
        return self.status == "current"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMeta":
        if "document_id" not in data:
            raise RegistryError("registry entry is missing document_id")
        pdf = data.get("pdf")
        if pdf is not None:
            path = Path(pdf)
            if not path.is_absolute():
                path = REPO_ROOT / path
            pdf = path
        markers = data.get("signature_markers")
        appendix = data.get("appendix_markers")
        return cls(
            document_id=str(data["document_id"]),
            document_name=str(data.get("document_name") or data["document_id"]),
            document_number=str(data.get("document_number") or ""),
            pdf=pdf,
            doc_type=str(data.get("doc_type") or "unknown"),
            issuer=data.get("issuer"),
            effective_from=data.get("effective_from"),
            effective_to=data.get("effective_to"),
            status=str(data.get("status") or "unknown"),
            expected_articles=data.get("expected_articles"),
            profile=str(data.get("profile") or "unknown"),
            signature_markers=tuple(markers) if markers else ("noinhan",),
            appendix_markers=tuple(appendix) if appendix else ("phuluc", "mauso", "bieuso"),
            include=bool(data.get("include", True)),
            notes=str(data.get("notes") or ""),
        )


@dataclass
class DocumentRegistry:
    """Validated collection of document metadata."""

    documents: List[DocumentMeta] = field(default_factory=list)

    def __post_init__(self) -> None:
        ids = [doc.document_id for doc in self.documents]
        duplicates = sorted({doc_id for doc_id in ids if ids.count(doc_id) > 1})
        if duplicates:
            raise RegistryError(f"duplicate document_id values: {duplicates}")

    def get(self, document_id: str) -> DocumentMeta:
        for document in self.documents:
            if document.document_id == document_id:
                return document
        raise KeyError(f"document_id not found in registry: {document_id}")

    def select(
        self,
        document_ids: Optional[Iterable[str]] = None,
        include_inactive: bool = True,
    ) -> List[DocumentMeta]:
        selected = []
        wanted = set(document_ids) if document_ids else None
        for document in self.documents:
            if not document.include:
                continue
            if wanted is not None and document.document_id not in wanted:
                continue
            if not include_inactive and not document.is_current():
                continue
            selected.append(document)
        return selected


def load_registry(path: Optional[Path | str] = None) -> DocumentRegistry:
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    if not registry_path.exists():
        raise FileNotFoundError(f"document registry not found: {registry_path}")
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        entries = data.get("documents", [])
    else:
        entries = data or []
    if not isinstance(entries, list):
        raise RegistryError("registry must be a list or a mapping with 'documents'")
    documents = [DocumentMeta.from_dict(entry) for entry in entries]
    return DocumentRegistry(documents)


@lru_cache(maxsize=4)
def _cached_registry(path_str: str) -> DocumentRegistry:
    return load_registry(Path(path_str))


def resolve_document(value: DocumentMeta | str | None) -> DocumentMeta:
    """Resolve a registry id to metadata, without hard-coded fallbacks.

    Unknown ids (e.g. unit tests using ``"test"``) get a neutral metadata
    object whose name/number are empty, never the Labour Code metadata.
    """
    if isinstance(value, DocumentMeta):
        return value
    document_id = value or "unknown"
    registry_path = DEFAULT_REGISTRY_PATH
    if registry_path.exists():
        try:
            return _cached_registry(str(registry_path)).get(document_id)
        except KeyError:
            pass
    return DocumentMeta(
        document_id=document_id,
        document_name="",
        document_number="",
    )
