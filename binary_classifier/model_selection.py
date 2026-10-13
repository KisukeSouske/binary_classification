"""Cross-validation splitters.

Each function returns a list of `n_splits` test-index arrays that together
cover every sample exactly once; the training indices of a fold are simply
all the others (see `train_indices`).
"""
import numpy as np


def _check_n_splits(n_splits: int, n_items: int, what: str):
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}.")
    if n_splits > n_items:
        raise ValueError(f"n_splits={n_splits} is more than the {n_items} {what} available.")


def stratified_k_fold(y, n_splits: int = 5, random_state: int | None = None) -> list[np.ndarray]:
    """
    K folds that each keep (as closely as possible) the class proportions of y.
    Every class is shuffled and dealt round-robin into the folds, so with a
    small dataset no fold ends up missing a class.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    for label in np.unique(y):
        _check_n_splits(n_splits, int(np.sum(y == label)), f"samples of class {label}")

    folds = [[] for _ in range(n_splits)]
    offset = 0
    for label in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == label))
        for position, index in enumerate(indices):
            # the offset carries on between classes, so leftovers don't all
            # pile up in the first folds
            folds[(offset + position) % n_splits].append(index)
        offset += len(indices)
    return [np.sort(np.array(fold, dtype=int)) for fold in folds]


def group_k_fold(groups, n_splits: int = 5, random_state: int | None = None) -> list[np.ndarray]:
    """
    K folds in which all samples sharing a group label land in the same fold.

    Use it when samples inside a group are not independent - e.g. photos taken
    in the same session - so the test fold never contains a near-copy of a
    training sample. Groups are placed largest first into the currently
    smallest fold, which keeps fold sizes balanced.
    """
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    _check_n_splits(n_splits, len(unique_groups), "groups")

    rng = np.random.default_rng(random_state)
    sizes = np.array([np.sum(groups == g) for g in unique_groups])
    # shuffle first so groups of equal size are dealt in a seed-dependent order
    order = rng.permutation(len(unique_groups))
    order = order[np.argsort(-sizes[order], kind="stable")]

    folds = [[] for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=int)
    for g in order:
        target = int(np.argmin(fold_sizes))
        members = np.flatnonzero(groups == unique_groups[g])
        folds[target].extend(members)
        fold_sizes[target] += len(members)
    return [np.sort(np.array(fold, dtype=int)) for fold in folds]


def train_indices(test_indices: np.ndarray, n_samples: int) -> np.ndarray:
    """The complement of a fold's test indices."""
    return np.setdiff1d(np.arange(n_samples), test_indices)
