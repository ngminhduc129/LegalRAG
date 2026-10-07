"""Retrieval module for Vietnamese legal documents."""

from legal_rag.retrieval.base import BaseRetriever
from legal_rag.retrieval.bm25 import BM25Index, build_bm25_index, tokenize_vietnamese
from legal_rag.retrieval.dense import DenseIndex, build_dense_index
from legal_rag.retrieval.hybrid import (
    HybridRetriever,
    rrf_fusion,
    weighted_fusion,
    run_hybrid_retrieval,
)
from legal_rag.retrieval.reranker import Reranker, run_reranker

__all__ = [
    "BaseRetriever",
    "BM25Index",
    "build_bm25_index",
    "tokenize_vietnamese",
    "DenseIndex",
    "build_dense_index",
    "HybridRetriever",
    "rrf_fusion",
    "weighted_fusion",
    "run_hybrid_retrieval",
    "Reranker",
    "run_reranker",
]