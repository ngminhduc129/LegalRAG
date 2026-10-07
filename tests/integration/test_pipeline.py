"""Integration tests for the full pipeline."""

import pytest
from pathlib import Path


class TestPipelineIntegration:
    @pytest.mark.integration
    def test_parse_to_chunks(self):
        """Test parsing PDF and creating chunks."""
        # This would require actual PDF and is slow
        pytest.skip("Requires PDF file and takes time")

    @pytest.mark.integration
    def test_chunk_to_index(self):
        """Test creating index from chunks."""
        pytest.skip("Requires chunks and takes time")

    @pytest.mark.integration
    def test_retrieve_and_generate(self):
        """Test full retrieve + generate pipeline."""
        pytest.skip("Requires LLM API key")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])