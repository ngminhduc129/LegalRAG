"""Base retriever interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""

    @abstractmethod
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """Add documents to the index."""
        pass

    @abstractmethod
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for top-k documents."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save index to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseRetriever":
        """Load index from disk."""
        pass