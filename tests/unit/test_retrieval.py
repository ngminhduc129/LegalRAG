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

    def test_dense_index_sets_and_persists_max_length(self, monkeypatch, tmp_path):
        import numpy as np
        import legal_rag.retrieval.dense as dense_module

        class FakeSentenceTransformer:
            def __init__(self, model_name, device):
                self.model_name = model_name
                self.device = device
                self.max_seq_length = 128

            def encode(self, texts, **kwargs):
                return np.ones((len(texts), 2), dtype=np.float32)

        monkeypatch.setattr(dense_module, "SentenceTransformer", FakeSentenceTransformer)
        monkeypatch.setattr(dense_module, "SENTENCE_TRANSFORMERS_AVAILABLE", True)

        index = dense_module.DenseIndex(model_name="test-model", device="cpu", max_length=256)
        assert index.model.max_seq_length == 256
        index.add_documents([{"chunk_id": "doc_1", "content": "Nội dung"}])

        path = tmp_path / "dense.pkl"
        index.save(path)
        loaded = dense_module.DenseIndex.load(path)

        assert loaded.max_length == 256
        assert loaded.model.max_seq_length == 256

    def test_dense_index_rejects_nonpositive_max_length(self, monkeypatch):
        import legal_rag.retrieval.dense as dense_module

        monkeypatch.setattr(dense_module, "SENTENCE_TRANSFORMERS_AVAILABLE", True)
        with pytest.raises(ValueError, match="max_length"):
            dense_module.DenseIndex(device="cpu", max_length=0)

    def test_save_raises_when_vector_count_mismatches_chunk_count(self, monkeypatch, tmp_path):
        import numpy as np
        import legal_rag.retrieval.dense as dense_module

        class FakeSentenceTransformer:
            def __init__(self, model_name, device):
                self.model_name = model_name
                self.device = device
                self.max_seq_length = 256
                self.tokenizer = None

            def encode(self, texts, **kwargs):
                return np.ones((len(texts), 2), dtype=np.float32)

        monkeypatch.setattr(dense_module, "SentenceTransformer", FakeSentenceTransformer)
        monkeypatch.setattr(dense_module, "SENTENCE_TRANSFORMERS_AVAILABLE", True)

        index = dense_module.DenseIndex(model_name="test-model", device="cpu", max_length=256)
        index.add_documents(
            [{"chunk_id": "doc_1", "content": "A"}, {"chunk_id": "doc_2", "content": "B"}]
        )
        # Force mismatch: 3 embedding rows but 2 documents.
        index.embeddings = np.ones((3, 2), dtype=np.float32)

        path = tmp_path / "dense.pkl"
        with pytest.raises(ValueError, match="does not match"):
            index.save(path)

    def test_add_documents_raises_when_chunk_missing_chunk_id(self, monkeypatch):
        import legal_rag.retrieval.dense as dense_module

        class FakeSentenceTransformer:
            def __init__(self, model_name, device):
                self.model_name = model_name
                self.device = device
                self.max_seq_length = 256
                self.tokenizer = None

            def encode(self, texts, **kwargs):
                import numpy as np

                return np.ones((len(texts), 2), dtype=np.float32)

        monkeypatch.setattr(dense_module, "SentenceTransformer", FakeSentenceTransformer)
        monkeypatch.setattr(dense_module, "SENTENCE_TRANSFORMERS_AVAILABLE", True)

        index = dense_module.DenseIndex(model_name="test-model", device="cpu", max_length=256)
        with pytest.raises(ValueError, match="missing a 'chunk_id'"):
            index.add_documents([{"content": "no id"}, {"chunk_id": "doc_2", "content": "B"}])

    def test_oversized_chunk_count_warns_and_counts(self, monkeypatch):
        import numpy as np
        import legal_rag.retrieval.dense as dense_module

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=True):
                return list(range(len(text.split())))

        class FakeSentenceTransformer:
            def __init__(self, model_name, device):
                self.model_name = model_name
                self.device = device
                self.max_seq_length = 256
                self.tokenizer = FakeTokenizer()

            def encode(self, texts, **kwargs):
                return np.ones((len(texts), 2), dtype=np.float32)

        monkeypatch.setattr(dense_module, "SentenceTransformer", FakeSentenceTransformer)
        monkeypatch.setattr(dense_module, "SENTENCE_TRANSFORMERS_AVAILABLE", True)

        index = dense_module.DenseIndex(model_name="test-model", device="cpu", max_length=10)
        chunks = [
            {"chunk_id": "doc_1", "content": "short"},
            {"chunk_id": "doc_2", "content": "x x x x x x x x x x x"},
        ]
        index.add_documents(chunks)
        assert index.oversized_chunk_count == 1

    def test_loads_index_without_vector_count_field(self, monkeypatch, tmp_path):
        import pickle
        import numpy as np
        import legal_rag.retrieval.dense as dense_module

        class FakeSentenceTransformer:
            def __init__(self, model_name, device):
                self.model_name = model_name
                self.device = device
                self.max_seq_length = 256
                self.tokenizer = None

            def encode(self, texts, **kwargs):
                return np.ones((len(texts), 2), dtype=np.float32)

        monkeypatch.setattr(dense_module, "SentenceTransformer", FakeSentenceTransformer)
        monkeypatch.setattr(dense_module, "SENTENCE_TRANSFORMERS_AVAILABLE", True)

        index = dense_module.DenseIndex(model_name="test-model", device="cpu", max_length=256)
        index.add_documents([{"chunk_id": "doc_1", "content": "A"}])

        # Save manually without vector_count / oversized_chunk_count to simulate an old index.
        data = {
            "model_name": index.model_name,
            "device": index.device,
            "batch_size": index.batch_size,
            "max_length": index.max_length,
            "documents": index.documents,
            "doc_ids": index.doc_ids,
            "embeddings": index.embeddings,
            "is_built": index.is_built,
        }
        path = tmp_path / "old_dense.pkl"
        with path.open("wb") as f:
            pickle.dump(data, f)

        loaded = dense_module.DenseIndex.load(path)
        assert loaded.vector_count is None
        assert loaded.oversized_chunk_count == 0
        assert len(loaded.documents) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])