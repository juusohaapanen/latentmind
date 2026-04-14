"""LSA model: TF-IDF vectorisation + truncated SVD, no storage dependencies."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

K           = 8   # latent dimensions
REFIT_EVERY = 20  # refit LSA after this many new memories


class LSAModel:
    def __init__(
        self,
        n_components: int = K,
        refit_every: int = REFIT_EVERY,
        # TF-IDF parameters
        min_df: int = 1,               # ignore terms appearing in fewer than N docs
        max_df: float = 0.95,          # ignore terms appearing in more than X% of docs
        ngram_range: tuple[int, int] = (1, 1),  # (1,1)=unigrams, (1,2)=+bigrams
        max_features: int | None = None,        # vocabulary size cap (None=unlimited)
    ):
        self.n_components = n_components
        self.refit_every  = refit_every
        self.tfidf = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            max_features=max_features,
        )
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
