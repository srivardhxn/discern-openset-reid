"""
Gallery-Adaptive Whitening.
Estimates the empirical mean and shrinkage covariance matrix over enrolled embeddings.
Whitening suppresses features shared across the gallery (e.g. uniform color, broad silhouette)
and amplifies discriminative, fine-grained identity features.
Automatically re-fits when gallery entries are added or removed.
"""

from __future__ import annotations
import numpy as np
from typing import Optional


class GalleryAdaptiveWhitening:
    """
    Gallery-Adaptive Covariance Whitener with Shrinkage Regularization.
    """
    def __init__(self, shrinkage: float = 0.15, eps: float = 1e-5):
        self.shrinkage = shrinkage
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.whitening_matrix: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, embeddings: np.ndarray) -> GalleryAdaptiveWhitening:
        """
        Fits mean and shrinkage whitening projection on enrolled gallery embeddings.
        embeddings: (N, D) array of L2-normalized vectors.
        """
        N, D = embeddings.shape
        if N < 8:
            self.mean = np.mean(embeddings, axis=0) if N > 0 else np.zeros(D, dtype=np.float32)
            self.whitening_matrix = np.eye(D, dtype=np.float32)
            self.is_fitted = False
            return self

        # Empirical mean
        self.mean = np.mean(embeddings, axis=0)
        X_centered = embeddings - self.mean

        # Sample covariance
        cov = np.dot(X_centered.T, X_centered) / max(1, N - 1)

        # Shrinkage covariance regularization: (1 - lambda)*cov + lambda*I
        cov_shrunk = (1.0 - self.shrinkage) * cov + self.shrinkage * np.eye(D)

        # Eigen-decomposition of symmetric matrix
        eigenvals, eigenvecs = np.linalg.eigh(cov_shrunk)

        # Clamp eigenvalues to prevent division by near-zero
        eigenvals = np.clip(eigenvals, self.eps, None)

        # Whitening transformation: W = V * diag(1 / sqrt(lambda)) * V^T
        inv_sqrt_vals = 1.0 / np.sqrt(eigenvals)
        self.whitening_matrix = np.dot(eigenvecs, np.dot(np.diag(inv_sqrt_vals), eigenvecs.T)).astype(np.float32)
        self.is_fitted = True
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Applies centering, whitening projection, and re-normalizes to unit sphere.
        embeddings: (N, D) or (D,)
        """
        if not self.is_fitted or self.whitening_matrix is None or self.mean is None:
            # If not fitted or disabled, return original L2-normalized vectors
            norm = np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-7
            return embeddings / norm

        is_1d = (embeddings.ndim == 1)
        X = np.atleast_2d(embeddings)

        X_centered = X - self.mean
        X_whitened = np.dot(X_centered, self.whitening_matrix)

        # Re-normalize to unit sphere
        norms = np.linalg.norm(X_whitened, axis=1, keepdims=True) + 1e-7
        X_normalized = X_whitened / norms

        if is_1d:
            return X_normalized[0]
        return X_normalized
