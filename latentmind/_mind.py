"""LatentMind: orchestrates LSA model and ChromaDB collection."""

from pathlib import Path

from ._lsa import LSAModel
from ._store import get_chroma_collection, LATENTMIND_DIR


class LatentMind:
    def __init__(self, latentmind_dir: Path = LATENTMIND_DIR, model: LSAModel | None = None):
        self.latentmind_dir = latentmind_dir
        self.collection = get_chroma_collection(latentmind_dir / "chroma")
        self.model      = model if model is not None else LSAModel()
        self._refit_if_needed()

    # --- internal -----------------------------------------------------------

    def _all(self) -> tuple[list[str], list[str]]:
        result = self.collection.get()
        return result["ids"] or [], result["documents"] or []

    def _refit_if_needed(self):
        ids, texts = self._all()
        if len(texts) <= self.model.n_components:
            return
        vectors = self.model.fit(texts)
        self.collection.update(ids=ids, embeddings=vectors.tolist())

    # --- public -------------------------------------------------------------

    def add(self, text: str, metadata: dict | None = None) -> str:
        """Add a single memory, return its id."""
        _, texts  = self._all()
        next_id   = f"mem_{len(texts):05d}"
        dim       = self.model.n_components

        self.collection.add(
            ids=[next_id],
            documents=[text],
            embeddings=[[0.0] * dim],
            metadatas=[metadata or {"source": "manual"}],
        )

        new_total = len(texts) + 1
        if new_total % self.model.refit_every == 0:
            self._refit_if_needed()
        elif self.model.fitted:
            vec = self.model.transform([text])[0]
            self.collection.update(ids=[next_id], embeddings=[vec.tolist()])

        return next_id

    def add_batch(self, texts: list[str], metadatas: list[dict] | None = None):
        """Add many memories at once, then refit."""
        _, existing = self._all()
        offset  = len(existing)
        ids     = [f"mem_{offset + i:05d}" for i in range(len(texts))]
        dim     = self.model.n_components
        metas   = metadatas or [{"source": "batch"} for _ in texts]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=[[0.0] * dim] * len(texts),
            metadatas=metas,
        )
        self._refit_if_needed()

    def search(self, query: str, top_n: int = 5) -> list[tuple[float, str]]:
        """Return (similarity_score, text) pairs for the closest memories."""
        if not self.model.fitted or self.collection.count() == 0:
            return []

        vec    = self.model.transform([query])[0]
        result = self.collection.query(
            query_embeddings=[vec.tolist()],
            n_results=min(top_n, self.collection.count()),
        )

        scores = [round(1 - d, 3) for d in result["distances"][0]]
        texts  = result["documents"][0]
        return list(zip(scores, texts))

    def count(self) -> int:
        return self.collection.count()

    def recent(self, n: int = 5) -> list[str]:
        """Return the n most recently added memories."""
        total = self.collection.count()
        if total == 0:
            return []
        offset = max(0, total - n)
        result = self.collection.get(limit=n, offset=offset)
        return result["documents"] or []

    def session_start_consolidation(self):
        """
        Call at the start of every Claude Code session.
        Refits LSA on the full corpus so memories added during
        the previous session get properly integrated.
        """
        print("Session start: consolidating memories...")
        self._refit_if_needed()
        print(f"Ready — {self.count()} memories in the mind.\n")
