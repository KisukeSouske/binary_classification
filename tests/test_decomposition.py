from binary_classifier.decomposition import PCA
import numpy as np
import pytest


def test_output_has_n_components_columns():
    X = np.random.default_rng(0).normal(size=(30, 12))
    assert PCA(4).fit_transform(X).shape == (30, 4)


def test_recovers_the_dominant_direction():
    # Points spread along (1, 1)/sqrt(2) with little noise: the first
    # component must be that direction (up to sign).
    rng = np.random.default_rng(1)
    t = rng.normal(scale=5.0, size=200)
    X = np.column_stack([t, t]) + rng.normal(scale=0.1, size=(200, 2))

    pca = PCA(1).fit(X)

    np.testing.assert_allclose(np.abs(pca.components[0]), [1 / np.sqrt(2)] * 2, atol=1e-2)
    assert pca.explained_variance_ratio[0] > 0.99


def test_components_are_orthonormal():
    X = np.random.default_rng(2).normal(size=(50, 8))
    C = PCA(5).fit(X).components
    np.testing.assert_allclose(C @ C.T, np.eye(5), atol=1e-10)


def test_projections_are_centered_and_ordered_by_variance():
    X = np.random.default_rng(3).normal(size=(100, 6)) * [5, 4, 3, 2, 1, 0.5]
    Z = PCA(6).fit_transform(X)
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-10)
    variances = Z.var(axis=0)
    assert all(variances[i] >= variances[i + 1] for i in range(5))


def test_transform_uses_the_training_statistics():
    # New data must be projected with the TRAIN mean, not its own
    X_train = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    pca = PCA(1).fit(X_train)
    z = pca.transform(np.array([[2.0, 0.0]]))
    np.testing.assert_allclose(z, [[0.0]], atol=1e-12)


def test_rejects_more_components_than_the_data_supports():
    X = np.zeros((5, 20))
    with pytest.raises(ValueError, match="more than the data supports"):
        PCA(5).fit(X)  # 5 samples -> at most 4 components


def test_transform_before_fit_raises():
    with pytest.raises(ValueError, match="not fitted"):
        PCA(2).transform(np.zeros((3, 4)))
