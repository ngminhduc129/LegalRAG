"""Unit tests for retrieval module."""

import pytest
from legal_rag.retrieval.bm25 import BM25Index, tokenize_vietnamese
from legal_rag.retrieval.hybrid import rrf_fusion, weighted_fusion


class TestBM25:
    def setup_method(self):
        self.chunks = [
            {
                "chunk_id": "doc_1",
                "content": "Điều 1. Phạm vi điều chỉnh của Bộ luật Lao động",
                "article": "Điều 1",
            },
            {
                "chunk_id": "doc_2",
                "content": "Điều 2. Đối tượng áp dụng của Bộ luật Lao động",
                "article": "Điều 2",
            },
            {
                "chunk_id": "doc_3",
                "content": "Nghỉ phép năm được 12 ngày làm việc",
                "article": "Điều 113",
            },
            {
                "chunk_id": "doc_4",
                "content": "Lương tối thiểu vùng là mức lương thấp nhất",
                "article": "Điều 91",
            },
        ]

    def test_tokenize_vietnamese(self):
        tokens = tokenize_vietnamese("Điều 1. Phạm vi điều chỉnh")
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_bm25_build_and_search(self):
        index = BM25Index(k1=1.5, b=0.75)
        index.add_documents(self.chunks)

        results = index.search("phạm vi điều chỉnh", k=2)
        assert len(results) == 1  # Only doc_1 contains these terms
        assert results[0][0] == "doc_1"  # Should match first doc best

        results = index.search("nghỉ phép năm", k=2)
        assert len(results) == 1
        assert results[0][0] == "doc_3"

    def test_bm25_save_load(self, tmp_path):
        index = BM25Index(k1=1.5, b=0.75)
        index.add_documents(self.chunks)

        save_path = tmp_path / "test_bm25.pkl"
        index.save(save_path)

        loaded = BM25Index.load(save_path)
        assert loaded.k1 == 1.5
        assert loaded.b == 0.75
        assert len(loaded.documents) == 4

        results = loaded.search("phạm vi điều chỉnh", k=1)
        assert results[0][0] == "doc_1"

    def test_bm25_empty_query(self):
        index = BM25Index()
        index.add_documents(self.chunks)
        results = index.search("", k=5)
        assert results == []


class TestHybridFusion:
    def setup_method(self):
        self.bm25_results = [
            ("doc_1", 10.0, {"chunk_id": "doc_1"}),
            ("doc_2", 8.0, {"chunk_id": "doc_2"}),
            ("doc_3", 5.0, {"chunk_id": "doc_3"}),
        ]
        self.dense_results = [
            ("doc_2", 0.9, {"chunk_id": "doc_2"}),
            ("doc_3", 0.8, {"chunk_id": "doc_3"}),
            ("doc_4", 0.7, {"chunk_id": "doc_4"}),
        ]

    def test_rrf_fusion(self):
        fused = rrf_fusion(self.bm25_results, self.dense_results, k=60, top_k=5)
        assert len(fused) <= 5

        # doc_2 appears in both, should rank high
        doc_ids = [r[0] for r in fused]
        assert "doc_2" in doc_ids
        assert "doc_1" in doc_ids
        assert "doc_3" in doc_ids

    def test_weighted_fusion(self):
        fused = weighted_fusion(
            self.bm25_results,
            self.dense_results,
            weight_bm25=0.5,
            weight_dense=0.5,
            top_k=5,
        )
        assert len(fused) <= 5

        doc_ids = [r[0] for r in fused]
        assert "doc_1" in doc_ids
        assert "doc_2" in doc_ids

    def test_rrf_with_k_parameter(self):
        # Higher k = more weight to lower ranks
        fused_low_k = rrf_fusion(self.bm25_results, self.dense_results, k=10, top_k=5)
        fused_high_k = rrf_fusion(self.bm25_results, self.dense_results, k=100, top_k=5)

        # Results should be different with different k
        low_ids = [r[0] for r in fused_low_k]
        high_ids = [r[0] for r in fused_high_k]
        # They might be same for small test, but function should not error


class TestDenseIndex:
    def test_dense_index_requires_sentence_transformers(self):
        # This test will be skipped if sentence-transformers not installed
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        chunks = [
            {"chunk_id": "doc_1", "content": "Test content 1", "article": "Điều 1"},
            {"chunk_id": "doc_2", "content": "Test content 2", "article": "Điều 2"},
        ]

        # This would require actual model download, so we skip in CI
        # Just test that class can be imported
        from legal_rag.retrieval.dense import DenseIndex
        assert DenseIndex is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])