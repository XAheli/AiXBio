from __future__ import annotations

"""
Baseline screening methods for comparison.

1. K-mer sequence similarity (proxy for BLAST/homology screening)
2. ESM-2 cosine nearest-neighbor
3. ESM-2 linear classifier
"""
import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score

from src.config import SEED

logger = logging.getLogger(__name__)


class KmerScreener:
    """
    K-mer based sequence similarity screening.
    Proxy for traditional homology-based screening (BLAST, HMM profile).
    """

    def __init__(self, k: int = 5):
        self.k = k
        self.threat_profiles: list[set[str]] = []

    def _kmerize(self, seq: str) -> set[str]:
        return {seq[i:i+self.k] for i in range(len(seq) - self.k + 1)}

    def fit(self, sequences: list[str], labels: np.ndarray):
        """Store k-mer profiles for threat sequences."""
        self.threat_profiles = []
        for seq, label in zip(sequences, labels):
            if label == 1:
                self.threat_profiles.append(self._kmerize(seq))
        logger.info(f"KmerScreener fitted with {len(self.threat_profiles)} threat profiles")

    def predict_proba(self, sequences: list[str]) -> np.ndarray:
        """Return max Jaccard similarity to any known threat."""
        scores = []
        for seq in sequences:
            query_kmers = self._kmerize(seq)
            max_sim = 0.0
            for threat_kmers in self.threat_profiles:
                intersection = len(query_kmers & threat_kmers)
                union = len(query_kmers | threat_kmers)
                sim = intersection / union if union > 0 else 0.0
                max_sim = max(max_sim, sim)
            scores.append(max_sim)
        return np.array(scores)


class CosineNNScreener:
    """
    Cosine nearest-neighbor screening in embedding space.
    Represents "just compute cosine similarity" — the naive embedding approach.
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.threat_embeddings: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray, labels: np.ndarray):
        """Store normalized threat embeddings."""
        threat_mask = labels == 1
        self.threat_embeddings = normalize(embeddings[threat_mask])
        logger.info(f"CosineNNScreener fitted with {self.threat_embeddings.shape[0]} threat embeddings")

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Return mean cosine similarity to k nearest threats."""
        query_norm = normalize(embeddings)
        # Cosine similarity: (n_query, n_threats)
        sims = query_norm @ self.threat_embeddings.T

        # Average of top-k similarities
        if sims.shape[1] <= self.n_neighbors:
            return sims.mean(axis=1)

        top_k = np.partition(sims, -self.n_neighbors, axis=1)[:, -self.n_neighbors:]
        return top_k.mean(axis=1)


class LinearScreener:
    """
    Logistic regression on embeddings.
    The simplest learned approach — a linear decision boundary.
    """

    def __init__(self, C: float = 1.0):
        self.model = LogisticRegression(
            C=C, class_weight="balanced", max_iter=1000,
            random_state=SEED, solver="lbfgs",
        )

    def fit(self, embeddings: np.ndarray, labels: np.ndarray):
        self.model.fit(embeddings, labels)
        train_auroc = roc_auc_score(labels, self.model.predict_proba(embeddings)[:, 1])
        logger.info(f"LinearScreener train AUROC: {train_auroc:.4f}")

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(embeddings)[:, 1]


class KNNEmbeddingScreener:
    """KNN classifier on embeddings — slightly more flexible than cosine NN."""

    def __init__(self, n_neighbors: int = 5):
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors, metric="cosine", weights="distance",
        )

    def fit(self, embeddings: np.ndarray, labels: np.ndarray):
        self.model.fit(normalize(embeddings), labels)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(normalize(embeddings))[:, 1]
