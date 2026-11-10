"""Unit tests for QA schema, coverage, and retrieval metrics."""

import json

import pytest

from legal_rag.evaluation.coverage import check_coverage, coverage_report
from legal_rag.evaluation.retrieval import (
    evaluate_retrieval,
    is_relevant,
    load_chunks,
)
from legal_rag.qa.schema import (
    QASchemaError,
    RelevantRef,
    load_qa,
    parse_record,
)


def _record(**kwargs):
    data = {
        "id": "q1",
        "question": "Người lao động được nghỉ bao nhiêu ngày?",
        "relevant_articles": [
            {"document_id": "labor_code_2019", "article": "Điều 113"}
        ],
    }
    data.update(kwargs)
    return data


def test_parse_record_normalizes_article():
    record = parse_record(_record(), 1)
    assert record.relevant[0].article == "113"
    assert record.relevant[0].document_id == "labor_code_2019"


def test_parse_record_accepts_string_refs():
    record = parse_record(
        _record(relevant_articles=["labor_code_2019:113", "35"]), 1
    )
    assert [ref.article for ref in record.relevant] == ["113", "35"]


def test_parse_record_requires_references():
    with pytest.raises(QASchemaError):
        parse_record({"id": "q1", "question": "Q?"}, 1)


def test_load_qa_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "qa.jsonl"
    path.write_text(
        json.dumps(_record()) + "\n" + json.dumps(_record()) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(QASchemaError):
        load_qa(path)


def _chunk(document_id="labor_code_2019", article="Điều 113", clause=None, **extra):
    return {
        "chunk_id": f"{document_id}_{article}_{clause}",
        "document_id": document_id,
        "article": article,
        "clause": clause,
        "content": "content",
        **extra,
    }


def test_coverage_finds_missing_article():
    record = parse_record(_record(), 1)
    chunks = [_chunk(article="Điều 1")]
    result = check_coverage([record], chunks)
    assert result.total_queries == 1
    assert result.covered_refs == 0
    assert result.missing_refs[0]["article"] == "113"


def test_coverage_ignores_appendix_chunks():
    record = parse_record(_record(), 1)
    chunks = [_chunk(article="Điều 113", retrieval_eligible=False)]
    assert check_coverage([record], chunks).covered_refs == 0
    assert (
        check_coverage([record], chunks, retrieval_eligible_only=False).covered_refs == 1
    )


def test_is_relevant_clause_level():
    ref = RelevantRef(document_id="labor_code_2019", article="113", clause="2")
    assert is_relevant(_chunk(clause="2"), [ref], "clause")
    assert not is_relevant(_chunk(clause="1"), [ref], "clause")
    assert is_relevant(_chunk(clause="1"), [ref], "article")


class DummyRetriever:
    def __init__(self, mapping):
        self.mapping = mapping

    def search(self, query, k=10):
        return [
            (chunk["chunk_id"], 1.0, chunk)
            for chunk in self.mapping.get(query, [])[:k]
        ]


def test_retrieval_metrics_hit_and_mrr():
    record = parse_record(_record(), 1)
    retriever = DummyRetriever(
        {
            record.question: [
                _chunk(article="Điều 1"),
                _chunk(article="Điều 113"),
            ]
        }
    )
    result = evaluate_retrieval([record], retriever, k_values=[1, 2])
    metrics = result["metrics"]
    assert metrics["hit_rate_at_k"]["1"] == 0.0
    assert metrics["hit_rate_at_k"]["2"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["ref_recall_at_k"]["2"] == 1.0


def test_retrieval_metrics_no_hit():
    record = parse_record(_record(), 1)
    retriever = DummyRetriever({record.question: [_chunk(article="Điều 1")]})
    metrics = evaluate_retrieval([record], retriever, k_values=[1, 5])["metrics"]
    assert metrics["hit_rate_at_k"]["5"] == 0.0
    assert metrics["mrr"] == 0.0


def test_retrieval_metrics_split_by_article_number_mention():
    with_number = parse_record(
        _record(id="q1", question="Theo Điều 113, nghỉ hằng năm bao nhiêu ngày?"), 1
    )
    without_number = parse_record(
        _record(id="q2", question="Nghỉ hằng năm bao nhiêu ngày?"), 2
    )
    retriever = DummyRetriever(
        {
            with_number.question: [_chunk(article="Điều 113")],
            without_number.question: [_chunk(article="Điều 1")],
        }
    )
    metrics = evaluate_retrieval(
        [with_number, without_number], retriever, k_values=[1]
    )["metrics"]
    groups = metrics["by_mentions_article_number"]
    assert groups["mentions"]["num_queries"] == 1
    assert groups["mentions"]["hit_rate_at_k"]["1"] == 1.0
    assert groups["no_mention"]["num_queries"] == 1
    assert groups["no_mention"]["hit_rate_at_k"]["1"] == 0.0


def test_load_chunks_filters_appendix(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        json.dumps(_chunk(article="Điều 1"))
        + "\n"
        + json.dumps(_chunk(article="Điều 2", retrieval_eligible=False))
        + "\n",
        encoding="utf-8",
    )
    assert len(load_chunks(path)) == 1
    assert len(load_chunks(path, retrieval_eligible_only=False)) == 2


def test_coverage_report_roundtrip(tmp_path):
    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        json.dumps(_chunk(article="Điều 113")) + "\n", encoding="utf-8"
    )
    report = coverage_report(qa_path, chunks_path)
    assert report["query_coverage"] == 1.0
    assert report["ref_coverage"] == 1.0
