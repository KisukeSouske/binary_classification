import numpy as np

def standardize(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    return (X_train - mean) / std, (X_test - mean) / std

def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_fraction: float,
    random_state: int | None = None,
):
    """
    Split X and y into a train and a test set by random permutation.
    test_fraction: share of the samples held out for testing, in [0, 1)
    random_state: seed for the shuffling RNG; None uses non-deterministic entropy
    Returns: X_train, X_test, y_train, y_test
    """
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in [0, 1); got {test_fraction}.")
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(y))
    n_test = int(len(y) * test_fraction)
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]