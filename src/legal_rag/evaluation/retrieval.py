"""Retrieval evaluation with frozen, explicitly documented metric definitions.

Hit definition (must stay identical across chunking strategies):

* article level: a retrieved chunk is relevant if
  ``chunk.document_id == ref.document_id`` and the chunk article number equals
  the reference article number.
* clause level: additionally the chunk clause must equal the reference clause
  (references without a clause are matched at article level).

Recall@k here is the query-level hit rate: fraction of queries with at least
one relevant chunk in the top ``k``.  ``ref_recall@k`` additionally reports how
many distinct gold references were found.  Both are reported so the two
readings cannot be confused.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from legal_rag.qa.schema import (
    QARecord,
    RelevantRef,
    load_qa,
    normalize_article,
    normalize_clause,
)

DEFAULT_K_VALUES = (1, 3, 5, 10, 20)


class Retriever(Protocol):
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        ...


def _ref_matches_chunk(ref: RelevantRef, chunk: Dict[str, Any], level: str) -> bool:
    if (chunk.get("document_id") or "unknown") != ref.document_id:
        return False
    if normalize_article(chunk.get("article")) != ref.article:
        return False
    if level == "clause" and ref.clause is not None:
        if normalize_clause(chunk.get("clause")) != ref.clause:
            return False
    return True


def is_relevant(chunk: Dict[str, Any], refs: Sequence[RelevantRef], level: str) -> bool:
    return any(_ref_matches_chunk(ref, chunk, level) for ref in refs)


@dataclass
class QueryResult:
    query_id: str
    question: str
    relevant: List[RelevantRef]
    retrieved: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "relevant": [ref.to_dict() for ref in self.relevant],
            "retrieved": [
                {
                    "rank": rank,
                    "chunk_id": chunk.get("chunk_id"),
                    "document_id": chunk.get("document_id"),
                    "article": chunk.get("article"),
                    "clause": chunk.get("clause"),
                    "score": score,
                }
                for rank, (chunk, score) in enumerate(self.retrieved, 1)
            ],
        }


def _dcg(relevances: Sequence[int]) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances))


def _aggregate(
    rows: List[Dict[str, Any]],
    k_values: Sequence[int],
    level: str,
) -> Dict[str, Any]:
    """Aggregate one group of already-retrieved queries into metrics.

    ``rows`` hold the per-query relevance vector and the gold references, so
    grouping (e.g. by ``mentions_article_number``) never re-runs retrieval and
    the metric definition stays identical across groups.
    """
    n = len(rows)
    total_refs = sum(len(row["relevant"]) for row in rows)
    hits = {k: 0 for k in k_values}
    ref_found = {k: 0 for k in k_values}
    precision_sum = {k: 0.0 for k in k_values}
    ndcg_sum = {k: 0.0 for k in k_values}
    mrr_sum = 0.0

    for row in rows:
        relevances = row["relevances"]
        ranked_chunks = row["ranked_chunks"]
        first_rank = next((i + 1 for i, rel in enumerate(relevances) if rel), None)
        if first_rank is not None:
            mrr_sum += 1.0 / first_rank

        for k in k_values:
            top = relevances[:k]
            if any(top):
                hits[k] += 1
            precision_sum[k] += sum(top) / k if k else 0.0
            found = set()
            for ref in row["relevant"]:
                if any(
                    _ref_matches_chunk(ref, chunk, level)
                    for chunk, _ in ranked_chunks[:k]
                ):
                    found.add((ref.document_id, ref.article, ref.clause))
            ref_found[k] += len(found)
            ideal = sorted(relevances, reverse=True)[:k]
            dcg = _dcg(top)
            idcg = _dcg(ideal)
            ndcg_sum[k] += (dcg / idcg) if idcg > 0 else 0.0

    def _mean(value: float) -> float:
        return round(value / n, 4) if n else 0.0

    return {
        "num_queries": n,
        "total_gold_refs": total_refs,
        "hit_rate_at_k": {str(k): _mean(hits[k]) for k in k_values},
        "precision_at_k": {str(k): _mean(precision_sum[k]) for k in k_values},
        "ndcg_at_k": {str(k): _mean(ndcg_sum[k]) for k in k_values},
        "ref_recall_at_k": {
            str(k): round(ref_found[k] / total_refs, 4) if total_refs else 0.0
            for k in k_values
        },
        "mrr": _mean(mrr_sum),
    }


def evaluate_retrieval(
    records: Iterable[QARecord],
    retriever: Retriever,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    level: str = "article",
    max_k: Optional[int] = None,
) -> Dict[str, Any]:
    k_values = sorted({int(k) for k in k_values if int(k) > 0})
    if not k_values:
        raise ValueError("k_values must contain at least one positive integer")
    fetch_k = max_k or max(k_values)

    records = list(records)
    if not records:
        raise ValueError("no QA records to evaluate")

    per_query: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    for record in records:
        results = retriever.search(record.question, k=fetch_k)
        ranked_chunks = [(chunk, score) for _, score, chunk in results]

        entry = QueryResult(
            query_id=record.id,
            question=record.question,
            relevant=record.relevant,
            retrieved=ranked_chunks,
        )
        per_query.append(entry.to_dict())

        rows.append(
            {
                "relevances": [
                    1 if is_relevant(chunk, record.relevant, level) else 0
                    for chunk, _ in ranked_chunks
                ],
                "relevant": record.relevant,
                "ranked_chunks": ranked_chunks,
                "mentions_article_number": bool(record.mentions_article_number),
            }
        )

    metrics = {
        "level": level,
        "k_values": k_values,
        **_aggregate(rows, k_values, level),
        # Article-number mentions can be matched from metadata text rather than
        # legal content; report both groups so the effect is visible.
        "by_mentions_article_number": {
            "mentions": _aggregate(
                [row for row in rows if row["mentions_article_number"]],
                k_values,
                level,
            ),
            "no_mention": _aggregate(
                [row for row in rows if not row["mentions_article_number"]],
                k_values,
                level,
            ),
        },
        "definitions": {
            "hit_rate_at_k": "fraction of queries with >=1 relevant chunk in top k (article level unless level=clause)",
            "ref_recall_at_k": "distinct gold refs retrieved in top k / total gold refs",
            "precision_at_k": "relevant chunks in top k / k, averaged over queries",
            "ndcg_at_k": "binary-relevance nDCG over retrieved chunks",
            "mrr": "mean reciprocal rank of the first relevant chunk",
            "by_mentions_article_number": "same metrics split by whether the question names an article number",
        },
    }
    return {"metrics": metrics, "per_query": per_query}


def load_chunks(path: Path | str, retrieval_eligible_only: bool = True) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chunk = json.loads(line)
            if retrieval_eligible_only and chunk.get("retrieval_eligible") is False:
                continue
            chunks.append(chunk)
    return chunks


class BM25Retriever:
    def __init__(self, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        from legal_rag.retrieval.bm25 import BM25Index

        self.index = BM25Index(k1=k1, b=b)
        self.index.add_documents(chunks)
        self.tokenizer = __import__(
            "legal_rag.retrieval.bm25", fromlist=["TOKENIZER"]
        ).TOKENIZER

    def search(self, query: str, k: int = 10):
        return self.index.search(query, k=k)


class DenseRetriever:
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        max_length: int = 256,
    ):
        from legal_rag.retrieval.dense import DenseIndex

        self.index = DenseIndex(model_name=model_name, max_length=max_length)
        self.index.add_documents(chunks)
        self.model_name = model_name

    def search(self, query: str, k: int = 10):
        return self.index.search(query, k=k)


class HybridEvaluatorRetriever:
    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        fusion: str = "rrf",
        rrf_k: int = 60,
        weight_bm25: float = 0.5,
        weight_dense: float = 0.5,
        fetch_k: int = 50,
    ):
        from legal_rag.retrieval.hybrid import HybridRetriever

        self.retriever = HybridRetriever(
            bm25_index=bm25.index,
            dense_index=dense.index,
            fusion_method=fusion,
            rrf_k=rrf_k,
            weight_bm25=weight_bm25,
            weight_dense=weight_dense,
        )
        self.fetch_k = fetch_k
        self.fusion = fusion
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 10):
        return self.retriever.search(query, k=k, fetch_k=max(self.fetch_k, k))


def build_retriever(
    method: str,
    chunks: List[Dict[str, Any]],
    fusion: str = "rrf",
    rrf_k: int = 60,
    weight_bm25: float = 0.5,
    weight_dense: float = 0.5,
    fetch_k: int = 50,
    dense_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> Retriever:
    if method == "bm25":
        return BM25Retriever(chunks)
    if method == "dense":
        return DenseRetriever(chunks, model_name=dense_model)
    if method == "hybrid":
        bm25 = BM25Retriever(chunks)
        dense = DenseRetriever(chunks, model_name=dense_model)
        return HybridEvaluatorRetriever(
            bm25,
            dense,
            fusion=fusion,
            rrf_k=rrf_k,
            weight_bm25=weight_bm25,
            weight_dense=weight_dense,
            fetch_k=fetch_k,
        )
    raise ValueError(f"unknown retrieval method {method!r}; expected bm25/dense/hybrid")


def retrieval_eval_report(
    qa_path: Path | str,
    chunks_path: Path | str,
    method: str = "bm25",
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    split: Optional[str] = None,
    level: str = "article",
    fusion: str = "rrf",
    rrf_k: int = 60,
    weight_bm25: float = 0.5,
    weight_dense: float = 0.5,
    fetch_k: int = 50,
    dense_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> Dict[str, Any]:
    records = load_qa(qa_path, split=split)
    chunks = load_chunks(chunks_path)
    retriever = build_retriever(
        method,
        chunks,
        fusion=fusion,
        rrf_k=rrf_k,
        weight_bm25=weight_bm25,
        weight_dense=weight_dense,
        fetch_k=fetch_k,
        dense_model=dense_model,
    )
    result = evaluate_retrieval(records, retriever, k_values=k_values, level=level)
    result["config"] = {
        "qa_path": str(qa_path),
        "chunks_path": str(chunks_path),
        "method": method,
        "split": split,
        "level": level,
        "fusion": fusion if method == "hybrid" else None,
        "rrf_k": rrf_k if method == "hybrid" else None,
        "weight_bm25": weight_bm25 if method == "hybrid" else None,
        "weight_dense": weight_dense if method == "hybrid" else None,
        "fetch_k": fetch_k if method == "hybrid" else None,
        "dense_model": dense_model if method in {"dense", "hybrid"} else None,
        "bm25_tokenizer": getattr(
            __import__("legal_rag.retrieval.bm25", fromlist=["TOKENIZER"]),
            "TOKENIZER",
            "unknown",
        ),
        "num_chunks": len(chunks),
    }
    return result
