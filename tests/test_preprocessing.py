from binary_classifier import preprocessing
import numpy as np
import pytest


# Standardize tests
def test_standardize_centers_and_scales():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=3.0, size=(200, 2))

    X_std, mean, std = preprocessing.standardize(X)

    np.testing.assert_allclose(X_std.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(X_std.std(axis=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(mean, X.mean(axis=0))
    np.testing.assert_allclose(std, X.std(axis=0))


def test_standardize_statistics_transform_new_data():
    # The returned mean/std are what lets a test set be scaled with the TRAIN
    # statistics, instead of leaking its own.
    X_train = np.array([[0.0], [2.0]])

    _, mean, std = preprocessing.standardize(X_train)
    X_test_std = (np.array([[4.0]]) - mean) / std

    np.testing.assert_allclose(X_test_std, [[3.0]])  # (4 - 1) / 1


def test_standardize_keeps_constant_features_finite():
    X = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])

    X_std, _, std = preprocessing.standardize(X)

    assert np.all(np.isfinite(X_std))
    np.testing.assert_allclose(X_std[:, 1], 0.0)
    assert std[1] == 1.0


# Train/test split tests
def test_train_test_split_sizes_and_partition():
    X = np.arange(100).reshape(100, 1).astype(float)
    y = np.arange(100).astype(float)

    X_train, X_test, y_train, y_test = preprocessing.train_test_split(
        X, y, 0.2, random_state=0
    )

    assert len(y_test) == 20
    assert len(y_train) == 80
    # every sample appears exactly once across the two splits
    assert sorted(np.concatenate([y_train, y_test])) == list(y)


def test_train_test_split_keeps_rows_aligned_with_labels():
    X = np.arange(50).reshape(50, 1).astype(float)
    y = np.arange(50).astype(float)

    X_train, X_test, y_train, y_test = preprocessing.train_test_split(
        X, y, 0.3, random_state=1
    )

    np.testing.assert_array_equal(X_train.ravel(), y_train)
    np.testing.assert_array_equal(X_test.ravel(), y_test)


def test_train_test_split_is_reproducible():
    X = np.arange(40).reshape(40, 1).astype(float)
    y = np.arange(40).astype(float)

    _, _, y_train1, _ = preprocessing.train_test_split(X, y, 0.25, random_state=42)
    _, _, y_train2, _ = preprocessing.train_test_split(X, y, 0.25, random_state=42)
    _, _, y_train3, _ = preprocessing.train_test_split(X, y, 0.25, random_state=7)

    np.testing.assert_array_equal(y_train1, y_train2)
    assert not np.array_equal(y_train1, y_train3)


def test_train_test_split_rejects_invalid_fraction():
    X = np.zeros((10, 1))
    y = np.zeros(10)
    with pytest.raises(ValueError):
        preprocessing.train_test_split(X, y, 1.0)
