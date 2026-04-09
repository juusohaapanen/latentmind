import numpy as np
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from pathlib import Path

LATENTMIND_DIR = Path(".latentmind")  # all persistence lives here
K           = 8                       # latent dimensions
REFIT_EVERY = 20                      # refit LSA after this many new memories

# ---------------------------------------------------------------------------
# ChromaDB  —  we manage embeddings ourselves so no embedding_function needed
# ---------------------------------------------------------------------------

def get_chroma_collection(persist_dir: Path) -> chromadb.Collection:
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(
        name="memories",
        metadata={"hnsw:space": "cosine"},
    )

# ---------------------------------------------------------------------------
# LSA model
# ---------------------------------------------------------------------------

class LSAModel:
    def __init__(self, n_components: int = K):
        self.n_components = n_components
        self.tfidf  = TfidfVectorizer()
        self.svd    = TruncatedSVD(n_components=n_components, random_state=42)
        self.fitted = False

    def fit(self, texts: list[str]) -> np.ndarray:
        """Fit on the full corpus, return normalised LSA matrix."""
        matrix      = self.tfidf.fit_transform(texts)
        lsa         = self.svd.fit_transform(matrix)
        self.fitted = True
        explained   = self.svd.explained_variance_ratio_.sum()
        print(f"  LSA refit: {len(texts)} memories, "
              f"{matrix.shape[1]} terms, "
              f"{self.n_components} components, "
              f"{explained:.1%} variance explained")
        return normalize(lsa)

    def transform(self, texts: list[str]) -> np.ndarray:
        """Project new texts into the fitted LSA space."""
        if not self.fitted:
            raise RuntimeError("Model not fitted — call fit() first")
        return normalize(self.svd.transform(self.tfidf.transform(texts)))

# ---------------------------------------------------------------------------
# MemoryPalace
# ---------------------------------------------------------------------------

class LatentMind:
    def __init__(self, latentmind_dir: Path = LATENTMIND_DIR):
        self.latentmind_dir = latentmind_dir
        self.collection = get_chroma_collection(latentmind_dir / "chroma")
        self.model      = LSAModel()
        self._refit_if_needed()

    # --- internal -----------------------------------------------------------

    def _all(self) -> tuple[list[str], list[str]]:
        result = self.collection.get()
        return result["ids"] or [], result["documents"] or []

    def _refit_if_needed(self):
        ids, texts = self._all()
        if not texts:
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
        if new_total % REFIT_EVERY == 0:
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

        # ChromaDB returns cosine distance (0=identical) — convert to similarity
        scores = [round(1 - d, 3) for d in result["distances"][0]]
        texts  = result["documents"][0]
        return list(zip(scores, texts))

    def count(self) -> int:
        return self.collection.count()

    def session_start_consolidation(self):
        """
        Call this at the start of every Claude Code session.
        Refits LSA on the full corpus so memories added during
        the previous session get properly integrated.
        """
        print("Session start: consolidating memories...")
        self._refit_if_needed()
        print(f"Ready — {self.count()} memories in the mind.\n")

