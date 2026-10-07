"""
Hybrid Retrieval with Reciprocal Rank Fusion (RRF).

Combines BM25 and dense retrieval results.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from legal_rag.retrieval.bm25 import BM25Index, build_bm25_index
from legal_rag.retrieval.dense import DenseIndex, build_dense_index


def rrf_fusion(
    bm25_results: List[Tuple[str, float, Dict[str, Any]]],
    dense_results: List[Tuple[str, float, Dict[str, Any]]],
    k: int = 60,
    top_k: int = 20,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Reciprocal Rank Fusion.

    score(d) = sum(1 / (k + rank_i(d))) for each system i

    Args:
        bm25_results: List of (doc_id, score, doc) from BM25
        dense_results: List of (doc_id, score, doc) from dense
        k: RRF parameter (default 60)
        top_k: Number of results to return

    Returns:
        Fused results sorted by RRF score
    """
    scores = {}
    doc_map = {}

    # BM25 ranks
    for rank, (doc_id, score, doc) in enumerate(bm25_results, 1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        doc_map[doc_id] = doc

    # Dense ranks
    for rank, (doc_id, score, doc) in enumerate(dense_results, 1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    # Sort by RRF score
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [(doc_id, score, doc_map[doc_id]) for doc_id, score in fused]


def weighted_fusion(
    bm25_results: List[Tuple[str, float, Dict[str, Any]]],
    dense_results: List[Tuple[str, float, Dict[str, Any]]],
    weight_bm25: float = 0.5,
    weight_dense: float = 0.5,
    top_k: int = 20,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Weighted score fusion.

    Normalizes scores to [0, 1] then combines with weights.
    """
    scores = {}
    doc_map = {}

    # Normalize BM25 scores
    bm25_scores = [s for _, s, _ in bm25_results]
    if bm25_scores:
        max_bm25 = max(bm25_scores)
        min_bm25 = min(bm25_scores)
        range_bm25 = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1.0
    else:
        range_bm25 = 1.0
        min_bm25 = 0.0

    for doc_id, score, doc in bm25_results:
        norm_score = (score - min_bm25) / range_bm25
        scores[doc_id] = scores.get(doc_id, 0.0) + weight_bm25 * norm_score
        doc_map[doc_id] = doc

    # Normalize dense scores
    dense_scores = [s for _, s, _ in dense_results]
    if dense_scores:
        max_dense = max(dense_scores)
        min_dense = min(dense_scores)
        range_dense = max_dense - min_dense if max_dense != min_dense else 1.0
    else:
        range_dense = 1.0
        min_dense = 0.0

    for doc_id, score, doc in dense_results:
        norm_score = (score - min_dense) / range_dense
        scores[doc_id] = scores.get(doc_id, 0.0) + weight_dense * norm_score
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [(doc_id, score, doc_map[doc_id]) for doc_id, score in fused]


class HybridRetriever:
    """Hybrid retriever combining BM25 and dense search."""

    def __init__(
        self,
        bm25_index: BM25Index,
        dense_index: DenseIndex,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        weight_bm25: float = 0.5,
        weight_dense: float = 0.5,
    ):
        self.bm25 = bm25_index
        self.dense = dense_index
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.weight_bm25 = weight_bm25
        self.weight_dense = weight_dense

    def search(self, query: str, k: int = 10, fetch_k: int = 50) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search using hybrid retrieval.

        Args:
            query: Search query
            k: Final number of results
            fetch_k: Number of results to fetch from each system before fusion
        """
        # Fetch more results for better fusion
        bm25_results = self.bm25.search(query, k=fetch_k)
        dense_results = self.dense.search(query, k=fetch_k)

        if self.fusion_method == "rrf":
            fused = rrf_fusion(bm25_results, dense_results, k=self.rrf_k, top_k=k)
        elif self.fusion_method == "weighted":
            fused = weighted_fusion(
                bm25_results, dense_results,
                weight_bm25=self.weight_bm25,
                weight_dense=self.weight_dense,
                top_k=k
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        return fused

    @classmethod
    def from_config(
        cls,
        chunks_path: Path,
        bm25_index_path: Path,
        dense_index_path: Path,
        config: Dict[str, Any],
        rebuild: bool = False,
    ) -> "HybridRetriever":
        """Create hybrid retriever from config."""
        # Load or build BM25
        if bm25_index_path.exists() and not rebuild:
            bm25 = BM25Index.load(bm25_index_path)
        else:
            bm25 = build_bm25_index(chunks_path, bm25_index_path)

        # Load or build dense
        if dense_index_path.exists() and not rebuild:
            dense = DenseIndex.load(dense_index_path)
        else:
            dense = build_dense_index(chunks_path, dense_index_path)

        return cls(
            bm25_index=bm25,
            dense_index=dense,
            fusion_method=config.get("fusion_method", "rrf"),
            rrf_k=config.get("rrf_k", 60),
            weight_bm25=config.get("weight_bm25", 0.5),
            weight_dense=config.get("weight_dense", 0.5),
        )


def run_hybrid_retrieval(
    query: str,
    chunks_path: Path = Path(r"E:\LegalRAG\data\chunks\legal_aware\labor_code_2019_legal_aware.jsonl"),
    bm25_index_path: Path = Path(r"E:\LegalRAG\data\indexes\bm25\bm25_legal_aware.pkl"),
    dense_index_path: Path = Path(r"E:\LegalRAG\data\indexes\dense\dense_legal_aware.pkl"),
    config: Optional[Dict[str, Any]] = None,
    k: int = 10,
    fetch_k: int = 50,
    rebuild: bool = False,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """Run hybrid retrieval for a query."""
    if config is None:
        config = {
            "fusion_method": "rrf",
            "rrf_k": 60,
            "weight_bm25": 0.5,
            "weight_dense": 0.5,
        }

    retriever = HybridRetriever.from_config(
        chunks_path=chunks_path,
        bm25_index_path=bm25_index_path,
        dense_index_path=dense_index_path,
        config=config,
        rebuild=rebuild,
    )

    return retriever.search(query, k=k, fetch_k=fetch_k)