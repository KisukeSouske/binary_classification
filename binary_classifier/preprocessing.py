import numpy as np

def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features to zero mean and unit standard deviation (Z-score).

    Parameters:
    X (np.ndarray): Feature values, shape (n_samples,) or (n_samples, n_features).

    Returns:
    tuple[np.ndarray, np.ndarray, np.ndarray]: (X_standardized, mean, std), where
        mean and std are computed per feature (axis=0) and can be reused to apply
        the same transformation to new data: (new_X - mean) / std.

    Constant features have zero standard deviation; their std is reported as 1 so
    the division is safe and the column simply becomes all zeros.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std

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