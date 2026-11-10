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
        max_length: int = 256,
    ):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        if max_length <= 0:
            raise ValueError("max_length must be greater than zero")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_length
        self.documents: List[Dict[str, Any]] = []
        self.doc_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.is_built = False
        self.oversized_chunk_count: int = 0

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the embedding model's tokenizer.

        Falls back to a whitespace estimate when the tokenizer is unavailable,
        so this helper never raises and never changes chunking behaviour.
        """
        try:
            return len(self.model.tokenizer.encode(text, add_special_tokens=True))
        except AttributeError:
            return len(text.split())

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

        # Warn about chunks that exceed the model's max_length; do NOT alter
        # chunking strategy here (caller decides how to split oversized chunks).
        oversized = 0
        for chunk, text in zip(chunks, texts):
            n_tokens = self._count_tokens(text)
            if n_tokens > self.max_length:
                oversized += 1
                chunk_id = chunk.get("chunk_id", "<unknown>")
                print(
                    f"[WARN] chunk {chunk_id} exceeds max_length={self.max_length} "
                    f"tokens ({n_tokens}); content will be truncated by the model"
                )
        self.oversized_chunk_count = oversized
        print(
            f"Computing embeddings for {len(texts)} documents... "
            f"{oversized} chunk(s) exceed max_length={self.max_length}"
        )
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
        if self.embeddings is None:
            raise RuntimeError("Index not built. Call add_documents() before save().")
        if self.embeddings.shape[0] != len(self.documents):
            raise ValueError(
                f"Embedding count ({self.embeddings.shape[0]}) does not match "
                f"document count ({len(self.documents)}); refusing to save"
            )
        data = {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "documents": self.documents,
            "doc_ids": self.doc_ids,
            "embeddings": self.embeddings,
            "is_built": self.is_built,
            "vector_count": int(self.embeddings.shape[0]),
            "oversized_chunk_count": int(self.oversized_chunk_count),
        }
        with path.open("wb") as f:
            pickle.dump(data, f)
        print(
            f"Dense index saved to {path}: "
            f"{self.embeddings.shape[0]} vectors for {len(self.documents)} documents"
        )

    @classmethod
    def load(cls, path: Path, device: Optional[str] = None) -> "DenseIndex":
        """Load index from disk."""
        with path.open("rb") as f:
            data = pickle.load(f)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        idx = cls(
            model_name=data["model_name"],
            device=device,
            batch_size=data["batch_size"],
            max_length=data.get("max_length", 256),
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
    max_length: int = 256,
) -> DenseIndex:
    """Build dense index from chunks file."""
    print(f"Loading chunks from {chunks_path}...")
    chunks = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"Building dense index with {len(chunks)} chunks...")
    index = DenseIndex(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    index.add_documents(chunks)
    index.save(index_path)
    return index