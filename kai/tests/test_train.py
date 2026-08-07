from binary_classifier import core, train
import numpy as np
import pytest


def test_initial_bias_matches_base_rate():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 3))
    y = np.zeros(n)
    y[:10] = 1.0  # 5% positive

    fit = train.fit_gradient_descent(
        X, y, learning_rate=0.0, epochs=1, batch_size=n, random_state=0
    )

    base_rate = y.mean()
    np.testing.assert_allclose(fit.bias, core.logit(base_rate))


def test_random_state_reproducibility():
    rng = np.random.default_rng(1)
    n = 200
    X = rng.normal(size=(n, 3))
    true_w = np.array([1.5, -2.0, 0.5])
    true_b = 0.3
    probs = core.sigmoid(X @ true_w + true_b)
    y = (rng.random(n) < probs).astype(float)

    fit1 = train.fit_gradient_descent(
        X, y, learning_rate=0.1, epochs=20, batch_size=32, random_state=42
    )
    fit2 = train.fit_gradient_descent(
        X, y, learning_rate=0.1, epochs=20, batch_size=32, random_state=42
    )
    np.testing.assert_allclose(fit1.weights, fit2.weights)
    assert fit1.bias == fit2.bias

    fit3 = train.fit_gradient_descent(
        X, y, learning_rate=0.1, epochs=20, batch_size=32, random_state=7
    )
    assert not np.allclose(fit1.weights, fit3.weights)


def test_recovers_true_parameters():
    rng = np.random.default_rng(2)
    n = 2000
    X = rng.normal(size=(n, 2))
    true_w = np.array([2.0, -1.5])
    true_b = 0.5
    probs = core.sigmoid(X @ true_w + true_b)
    y = (rng.random(n) < probs).astype(float)

    fit = train.fit_gradient_descent(
        X, y, learning_rate=0.5, epochs=500, batch_size=200, random_state=3
    )

    np.testing.assert_allclose(fit.weights, true_w, atol=0.3)
    np.testing.assert_allclose(fit.bias, true_b, atol=0.3)


def test_loss_decreases_over_training():
    rng = np.random.default_rng(4)
    n = 500
    X = rng.normal(size=(n, 3))
    true_w = np.array([1.0, -1.0, 0.5])
    probs = core.sigmoid(X @ true_w)
    y = (rng.random(n) < probs).astype(float)

    fit = train.fit_gradient_descent(
        X, y, learning_rate=0.3, epochs=200, batch_size=50, tolerance=0.0, random_state=5
    )

    loss_history = np.array(fit.loss_history)
    assert len(loss_history) > 10
    early_mean = loss_history[:10].mean()
    late_mean = loss_history[-10:].mean()
    assert late_mean < early_mean


def test_diverges_with_huge_learning_rate():
    # sigmoid/log_loss clip predictions, so the loss itself stays bounded
    # (~11-15) no matter how large a full-batch gradient step is: the
    # per-step error is always in [-1, 1], so weights grow at most linearly
    # in learning_rate per epoch and the isfinite() guard never trips, even
    # at learning_rate=1e300. With batch_size=1 the update is a single raw
    # X_i * error term multiplied straight through by learning_rate, so a
    # learning_rate near float64's max (~1.8e308) overflows the weights to
    # inf in one step, producing the nan loss this guard is meant to catch.
    rng = np.random.default_rng(6)
    n = 100
    X = rng.normal(size=(n, 2))
    y = np.array([0.0] * 50 + [1.0] * 50)
    rng.shuffle(y)

    with pytest.raises(ValueError):
        train.fit_gradient_descent(
            X, y, learning_rate=1e308, epochs=5, batch_size=1, random_state=6
        )


def test_stops_early_for_trivially_separable_data():
    X = np.array([[-1.0], [-1.0], [1.0], [1.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    fit = train.fit_gradient_descent(
        X, y, learning_rate=1.0, epochs=10_000, batch_size=4, tolerance=1e-2, random_state=7
    )

    assert fit.loss_history is not None
    assert len(fit.loss_history) < 10_000
