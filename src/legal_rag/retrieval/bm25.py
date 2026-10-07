"""
BM25 Retrieval for Vietnamese Legal Documents.

Uses underthesea or pyvi for Vietnamese tokenization.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import math

try:
    from underthesea import word_tokenize
    TOKENIZER = "underthesea"
except ImportError:
    try:
        from pyvi import ViTokenizer
        TOKENIZER = "pyvi"
    except ImportError:
        TOKENIZER = "simple"


def tokenize_vietnamese(text: str) -> List[str]:
    """Tokenize Vietnamese text."""
    text = text.lower().strip()
    if TOKENIZER == "underthesea":
        return word_tokenize(text, format="text").split()
    elif TOKENIZER == "pyvi":
        return ViTokenizer.tokenize(text).split()
    else:
        # Simple fallback: split on non-word chars
        import re
        return re.findall(r"\w+", text)


class BM25Index:
    """BM25 index for Vietnamese legal documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Dict[str, Any]] = []
        self.doc_ids: List[str] = []
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.inverted_index: Dict[str, Dict[int, int]] = {}  # term -> {doc_idx: freq}
        self.doc_freq: Dict[str, int] = {}  # term -> number of docs containing term
        self.is_built = False

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Add documents to the index."""
        self.documents = chunks
        self.doc_ids = [c["chunk_id"] for c in chunks]

        # Tokenize all documents
        tokenized_docs = []
        for chunk in chunks:
            content = chunk.get("content", "")
            tokens = tokenize_vietnamese(content)
            tokenized_docs.append(tokens)

        # Compute document lengths
        self.doc_lengths = [len(tokens) for tokens in tokenized_docs]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0

        # Build inverted index
        self.inverted_index = {}
        self.doc_freq = {}

        for doc_idx, tokens in enumerate(tokenized_docs):
            term_counts = Counter(tokens)
            for term, freq in term_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][doc_idx] = freq
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

        self.is_built = True
        print(f"BM25 index built: {len(self.documents)} docs, {len(self.inverted_index)} terms, avgdl={self.avgdl:.1f}")

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for top-k documents."""
        if not self.is_built:
            raise RuntimeError("Index not built. Call add_documents() first.")

        query_tokens = tokenize_vietnamese(query)
        if not query_tokens:
            return []

        scores = {}
        N = len(self.documents)

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            df = self.doc_freq[term]
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

            for doc_idx, freq in self.inverted_index[term].items():
                doc_len = self.doc_lengths[doc_idx]
                tf = freq

                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score = idf * numerator / denominator

                scores[doc_idx] = scores.get(doc_idx, 0.0) + score

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for doc_idx, score in ranked:
            results.append((self.doc_ids[doc_idx], score, self.documents[doc_idx]))

        return results

    def save(self, path: Path):
        """Save index to disk."""
        data = {
            "k1": self.k1,
            "b": self.b,
            "documents": self.documents,
            "doc_ids": self.doc_ids,
            "doc_lengths": self.doc_lengths,
            "avgdl": self.avgdl,
            "inverted_index": self.inverted_index,
            "doc_freq": self.doc_freq,
            "is_built": self.is_built,
            "tokenizer": TOKENIZER,
        }
        with path.open("wb") as f:
            pickle.dump(data, f)
        print(f"BM25 index saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """Load index from disk."""
        with path.open("rb") as f:
            data = pickle.load(f)

        idx = cls(k1=data["k1"], b=data["b"])
        idx.documents = data["documents"]
        idx.doc_ids = data["doc_ids"]
        idx.doc_lengths = data["doc_lengths"]
        idx.avgdl = data["avgdl"]
        idx.inverted_index = data["inverted_index"]
        idx.doc_freq = data["doc_freq"]
        idx.is_built = data["is_built"]

        print(f"BM25 index loaded from {path}: {len(idx.documents)} docs, tokenizer={data.get('tokenizer', 'unknown')}")
        return idx


def build_bm25_index(
    chunks_path: Path,
    index_path: Path,
    k1: float = 1.5,
    b: float = 0.75,
) -> BM25Index:
    """Build BM25 index from chunks file."""
    print(f"Loading chunks from {chunks_path}...")
    chunks = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    print(f"Building BM25 index with {len(chunks)} chunks...")
    index = BM25Index(k1=k1, b=b)
    index.add_documents(chunks)
    index.save(index_path)
    return index