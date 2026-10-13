from binary_classifier import model_selection
import numpy as np
import pytest


def assert_partition(folds, n):
    # every sample is tested exactly once
    everything = np.concatenate(folds)
    assert sorted(everything) == list(range(n))


# Stratified k-fold tests
def test_stratified_folds_partition_the_samples():
    y = np.array([0] * 42 + [1] * 46)
    folds = model_selection.stratified_k_fold(y, n_splits=5, random_state=0)
    assert len(folds) == 5
    assert_partition(folds, len(y))


def test_stratified_folds_keep_class_proportions():
    y = np.array([0] * 20 + [1] * 30)
    for fold in model_selection.stratified_k_fold(y, n_splits=5, random_state=1):
        assert np.sum(y[fold] == 0) == 4
        assert np.sum(y[fold] == 1) == 6


def test_stratified_fold_sizes_differ_by_at_most_one():
    y = np.array([0] * 42 + [1] * 46)
    sizes = [len(f) for f in model_selection.stratified_k_fold(y, n_splits=5, random_state=0)]
    assert max(sizes) - min(sizes) <= 1


def test_stratified_folds_are_reproducible():
    y = np.array([0] * 10 + [1] * 10)
    a = model_selection.stratified_k_fold(y, 5, random_state=3)
    b = model_selection.stratified_k_fold(y, 5, random_state=3)
    c = model_selection.stratified_k_fold(y, 5, random_state=4)
    assert all(np.array_equal(x, z) for x, z in zip(a, b))
    assert not all(np.array_equal(x, z) for x, z in zip(a, c))


def test_stratified_rejects_more_folds_than_a_class_has():
    y = np.array([0, 0, 1, 1, 1])
    with pytest.raises(ValueError):
        model_selection.stratified_k_fold(y, n_splits=3)


# Group k-fold tests
def test_groups_are_never_split_across_folds():
    groups = np.array(["a"] * 4 + ["b"] * 3 + ["c"] * 3 + ["d"] * 2 + ["e"] * 2 + ["f"])
    folds = model_selection.group_k_fold(groups, n_splits=3, random_state=0)

    assert_partition(folds, len(groups))
    fold_of = {}
    for k, fold in enumerate(folds):
        for g in groups[fold]:
            assert fold_of.setdefault(g, k) == k


def test_group_folds_are_balanced():
    groups = np.repeat(np.arange(20), 2)  # 20 groups of 2 samples
    sizes = [len(f) for f in model_selection.group_k_fold(groups, n_splits=4, random_state=0)]
    assert sizes == [10, 10, 10, 10]


def test_group_rejects_more_folds_than_groups():
    with pytest.raises(ValueError):
        model_selection.group_k_fold(np.array(["a", "a", "b"]), n_splits=3)


def test_train_indices_is_the_complement():
    np.testing.assert_array_equal(
        model_selection.train_indices(np.array([1, 3]), 5), [0, 2, 4]
    )
