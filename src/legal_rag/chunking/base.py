"""Base chunker interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse lines into chunks."""
        pass

    @abstractmethod
    def chunk_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Parse PDF directly into chunks."""
        pass