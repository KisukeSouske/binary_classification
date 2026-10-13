"""Principal component analysis, from scratch on top of numpy's SVD.

Used to compress an image (hundreds of pixels) into a handful of numbers
before the classifier sees it, so the model always has far fewer parameters
than training samples. Like `preprocessing.standardize`, it must be fitted on
the training data only and then applied unchanged to anything else.
"""
import numpy as np


class PCA:
    """Projects data onto its `n_components` directions of largest variance.

    >>> pca = PCA(n_components=10).fit(X_train)
    >>> Z_train, Z_test = pca.transform(X_train), pca.transform(X_test)
    """

    def __init__(self, n_components: int):
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}.")
        self.n_components = int(n_components)
        self.mean: np.ndarray | None = None
        self.components: np.ndarray | None = None  # shape (n_components, n_features)
        self.explained_variance_ratio: np.ndarray | None = None

    def fit(self, X) -> "PCA":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.ndim} dimensions.")
        n_samples, n_features = X.shape
        # Centered data has rank at most n_samples - 1, so asking for more
        # components than that would return directions with zero variance.
        max_components = min(n_samples - 1, n_features)
        if self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} is more than the data supports "
                f"({max_components} for {n_samples} samples x {n_features} features)."
            )

        self.mean = X.mean(axis=0)
        # Rows of Vt are the principal directions, sorted by singular value.
        _, singular_values, Vt = np.linalg.svd(X - self.mean, full_matrices=False)
        variance = singular_values ** 2
        self.components = Vt[: self.n_components]
        self.explained_variance_ratio = variance[: self.n_components] / variance.sum()
        return self

    def transform(self, X) -> np.ndarray:
        if self.components is None:
            raise ValueError("PCA is not fitted yet; call fit(...) first.")
        return (np.asarray(X, dtype=float) - self.mean) @ self.components.T

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)
