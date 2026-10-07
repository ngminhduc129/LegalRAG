"""
Dense Retrieval for Vietnamese Legal Documents.

Uses sentence-transformers for multilingual embeddings.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class DenseIndex:
    """Dense vector index for semantic search."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.documents: List[Dict[str, Any]] = []
        self.doc_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.is_built = False

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Add documents and compute embeddings."""
        self.documents = chunks
        self.doc_ids = [c["chunk_id"] for c in chunks]

        # Prepare texts for embedding
        texts = []
        for chunk in chunks:
            # Include legal hierarchy in the text for better retrieval
            parts = []
            if chunk.get("chapter"):
                parts.append(chunk["chapter"])
            if chunk.get("section"):
                parts.append(chunk["section"])
            if chunk.get("article"):
                parts.append(chunk["article"])
            if chunk.get("article_title"):
                parts.append(chunk["article_title"])
            if chunk.get("clause"):
                parts.append(f"Khoản {chunk['clause']}")
            if chunk.get("point"):
                parts.append(f"Điểm {chunk['point']}")
            parts.append(chunk.get("content", ""))
            texts.append(" ".join(parts))

        print(f"Computing embeddings for {len(texts)} documents...")
        self.embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # For cosine similarity
        )
        self.is_built = True
        print(f"Dense index built: {len(self.documents)} docs, embedding dim={self.embeddings.shape[1]}")

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for top-k documents using cosine similarity."""
        if not self.is_built:
            raise RuntimeError("Index not built. Call add_documents() first.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Cosine similarity (dot product since normalized)
        scores = np.dot(self.embeddings, query_embedding.T).flatten()

        # Get top-k
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            results.append((self.doc_ids[idx], float(scores[idx]), self.documents[idx]))

        return results

    def save(self, path: Path):
        """Save index to disk."""
        data = {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "documents": self.documents,
            "doc_ids": self.doc_ids,
            "embeddings": self.embeddings,
            "is_built": self.is_built,
        }
        with path.open("wb") as f:
            pickle.dump(data, f)
        print(f"Dense index saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "DenseIndex":
        """Load index from disk."""
        with path.open("rb") as f:
            data = pickle.load(f)

        idx = cls(
            model_name=data["model_name"],
            device=data["device"],
            batch_size=data["batch_size"],
        )
        idx.documents = data["documents"]
        idx.doc_ids = data["doc_ids"]
        idx.embeddings = data["embeddings"]
        idx.is_built = data["is_built"]

        print(f"Dense index loaded from {path}: {len(idx.documents)} docs")
        return idx


def build_dense_index(
    chunks_path: Path,
    index_path: Path,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device: Optional[str] = None,
    batch_size: int = 32,
) -> DenseIndex:
    """Build dense index from chunks file."""
    print(f"Loading chunks from {chunks_path}...")
    chunks = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"Building dense index with {len(chunks)} chunks...")
    index = DenseIndex(model_name=model_name, device=device, batch_size=batch_size)
    index.add_documents(chunks)
    index.save(index_path)
    return index