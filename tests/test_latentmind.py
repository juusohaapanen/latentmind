import pytest
import numpy as np
from pathlib import Path

from latentmind import LSAModel, LatentMind

CORPUS = [
    "Latent semantic analysis uses singular value decomposition",
    "TF-IDF weights terms by frequency across the corpus",
    "Vector spaces allow semantically similar concepts to cluster",
    "SVD reduces a high dimensional matrix to lower dimensions",
    "Cosine similarity measures the angle between two vectors",
    "Word embeddings represent words as dense vectors",
    "uv is a fast Python package manager written in Rust",
    "scikit-learn provides TfidfVectorizer and TruncatedSVD",
    "numpy is the foundation for numerical computing in Python",
]


# ---------------------------------------------------------------------------
# LSAModel
# ---------------------------------------------------------------------------

class TestLSAModel:
    def test_fit_returns_normalised_vectors(self):
        model = LSAModel(n_components=4)
        vectors = model.fit(CORPUS)
        assert vectors.shape == (len(CORPUS), 4)
        norms = np.linalg.norm(vectors, axis=1)
        # Zero vectors are valid (document projects to zero in LSA space);
        # every non-zero vector must be unit length.
        for norm in norms:
            assert norm == pytest.approx(0.0, abs=1e-6) or norm == pytest.approx(1.0, abs=1e-6)

    def test_fit_sets_fitted_flag(self):
        model = LSAModel(n_components=4)
        assert not model.fitted
        model.fit(CORPUS)
        assert model.fitted

    def test_transform_requires_fitted_model(self):
        model = LSAModel(n_components=4)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.transform(["some text"])

    def test_transform_projects_new_text(self):
        model = LSAModel(n_components=4)
        model.fit(CORPUS)
        vectors = model.transform(["singular value decomposition for topics"])
        assert vectors.shape == (1, 4)
        norm = np.linalg.norm(vectors[0])
        assert pytest.approx(norm, abs=1e-6) == 1.0

    def test_transform_batch(self):
        model = LSAModel(n_components=4)
        model.fit(CORPUS)
        vectors = model.transform(["text one", "text two", "text three"])
        assert vectors.shape == (3, 4)


# ---------------------------------------------------------------------------
# LatentMind
# ---------------------------------------------------------------------------

class TestLatentMind:
    def test_empty_store(self, tmp_path):
        mind = LatentMind(tmp_path)
        assert mind.count() == 0

    def test_search_returns_empty_when_no_memories(self, tmp_path):
        mind = LatentMind(tmp_path)
        assert mind.search("anything") == []

    def test_add_returns_id(self, tmp_path):
        mind = LatentMind(tmp_path)
        mind.add_batch(CORPUS)
        mem_id = mind.add("a new memory about vectors")
        assert mem_id == f"mem_{len(CORPUS):05d}"

    def test_add_increments_count(self, tmp_path):
        mind = LatentMind(tmp_path)
        mind.add_batch(CORPUS)
        before = mind.count()
        mind.add("one more memory")
        assert mind.count() == before + 1

    def test_add_batch_populates_store(self, tmp_path):
        mind = LatentMind(tmp_path)
        mind.add_batch(CORPUS)
        assert mind.count() == len(CORPUS)

    def test_search_returns_top_n_results(self, tmp_path):
        mind = LatentMind(tmp_path)
        mind.add_batch(CORPUS)
        results = mind.search("vector space embeddings", top_n=3)
        assert len(results) == 3

    def test_search_scores_are_between_zero_and_one(self, tmp_path):
        mind = LatentMind(tmp_path)
        mind.add_batch(CORPUS)
        results = mind.search("machine learning", top_n=5)
        for score, _ in results:
            assert 0.0 <= score <= 1.0

    def test_search_ranks_relevant_result_higher(self, tmp_path):
        mind = LatentMind(tmp_path)
        mind.add_batch(CORPUS)
        results = mind.search("SVD dimensionality reduction", top_n=5)
        texts = [text for _, text in results]
        # The most directly relevant memory should appear in results
        assert any("SVD" in t or "dimension" in t for t in texts)

    def test_search_top_n_capped_by_corpus_size(self, tmp_path):
        # CORPUS has 9 entries (> K=8), so LSA fits cleanly.
        # Requesting more results than the corpus size should return all of them.
        mind = LatentMind(tmp_path)
        mind.add_batch(CORPUS)
        results = mind.search("vector", top_n=50)
        assert len(results) == len(CORPUS)

    def test_recent_returns_last_n(self, tmp_path):
        mind = LatentMind(tmp_path)
        mind.add_batch(CORPUS)
        recent = mind.recent(n=3)
        assert recent == CORPUS[-3:]

    def test_recent_returns_all_when_fewer_than_n(self, tmp_path):
        mind = LatentMind(tmp_path)
        mind.add_batch(CORPUS[:3])
        recent = mind.recent(n=10)
        assert len(recent) == 3

    def test_recent_empty_store(self, tmp_path):
        mind = LatentMind(tmp_path)
        assert mind.recent() == []

    def test_persistence_across_instances(self, tmp_path):
        mind1 = LatentMind(tmp_path)
        mind1.add_batch(CORPUS)

        mind2 = LatentMind(tmp_path)
        assert mind2.count() == len(CORPUS)

    def test_session_start_consolidation(self, tmp_path, capsys):
        mind = LatentMind(tmp_path)
        mind.add_batch(CORPUS)
        mind.session_start_consolidation()
        captured = capsys.readouterr()
        assert "consolidating" in captured.out
        assert str(len(CORPUS)) in captured.out
