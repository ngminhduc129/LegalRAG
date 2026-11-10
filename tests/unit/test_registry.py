"""Unit tests for the document registry."""

import pytest

from legal_rag.corpus.registry import (
    DocumentMeta,
    RegistryError,
    load_registry,
    resolve_document,
)


def test_registry_loads_all_documents():
    registry = load_registry()
    assert len(registry.documents) >= 22
    ids = [doc.document_id for doc in registry.documents]
    assert len(ids) == len(set(ids))
    assert registry.get("labor_code_2019").document_number == "125/VBHN-VPQH"


def test_resolve_unknown_id_is_neutral():
    doc = resolve_document("unit_test_doc")
    assert doc.document_id == "unit_test_doc"
    assert doc.document_name == ""
    assert doc.document_number == ""


def test_resolve_accepts_document_meta():
    meta = DocumentMeta(document_id="x", document_name="X", document_number="1")
    assert resolve_document(meta) is meta


def test_metadata_fields_are_copied():
    meta = DocumentMeta(
        document_id="x",
        document_name="X",
        document_number="1",
        doc_type="Nghị định",
        issuer="Chính phủ",
        status="current",
    )
    data = meta.as_metadata()
    assert data["document_id"] == "x"
    assert data["document_type"] == "Nghị định"
    assert data["status"] == "current"


def test_invalid_status_rejected():
    with pytest.raises(RegistryError):
        DocumentMeta(document_id="x", document_name="X", document_number="1", status="bogus")


def test_invalid_date_rejected():
    with pytest.raises(RegistryError):
        DocumentMeta(
            document_id="x",
            document_name="X",
            document_number="1",
            effective_from="01/01/2020",
        )


def test_select_filters_by_id():
    registry = load_registry()
    selected = registry.select(document_ids=["labor_code_2019"])
    assert [doc.document_id for doc in selected] == ["labor_code_2019"]
