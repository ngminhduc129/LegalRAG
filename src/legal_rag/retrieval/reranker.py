"""
Cross-encoder Reranker for Hybrid Retrieval.

Optional reranking of top-k results using a cross-encoder model.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

try:
    from sentence_transformers import CrossEncoder
    import torch
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


class Reranker:
    """Cross-encoder reranker for retrieval results."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=device, max_length=512)

    def rerank(
        self,
        query: str,
        results: List[Tuple[str, float, Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Rerank results using cross-encoder.

        Args:
            query: Search query
            results: List of (doc_id, score, doc) from retrieval
            top_k: Number of top results to return (None = all)

        Returns:
            Reranked results with cross-encoder scores
        """
        if not results:
            return []

        # Prepare pairs for cross-encoder
        pairs = []
        for doc_id, score, doc in results:
            # Create passage with legal context
            parts = []
            if doc.get("chapter"):
                parts.append(doc["chapter"])
            if doc.get("section"):
                parts.append(doc["section"])
            if doc.get("article"):
                parts.append(doc["article"])
            if doc.get("article_title"):
                parts.append(doc["article_title"])
            if doc.get("clause"):
                parts.append(f"Khoản {doc['clause']}")
            if doc.get("point"):
                parts.append(f"Điểm {doc['point']}")
            parts.append(doc.get("content", ""))
            passage = " ".join(parts)
            pairs.append([query, passage])

        # Get cross-encoder scores
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        # Combine with original results
        reranked = []
        for i, (doc_id, orig_score, doc) in enumerate(results):
            reranked.append((doc_id, float(scores[i]), doc))

        # Sort by cross-encoder score
        reranked.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        return reranked


def run_reranker(
    query: str,
    results: List[Tuple[str, float, Dict[str, Any]]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: Optional[int] = None,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """Convenience function to rerank results."""
    reranker = Reranker(model_name=model_name)
    return reranker.rerank(query, results, top_k=top_k)