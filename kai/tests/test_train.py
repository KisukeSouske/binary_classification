from binary_classifier import core, train
import numpy as np
import pytest
import threading


@pytest.fixture
def separable_ish_data():
    """A signal-carrying problem with one near-irrelevant third feature."""
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 3))
    true_w = np.array([3.0, -2.0, 0.05])
    probs = core.sigmoid(X @ true_w + 0.2)
    y = (rng.random(n) < probs).astype(float)
    return X, y


def _fit(X, y, **kwargs):
    model = train.ClassifierFit(**kwargs)
    return model.fit(X, y, learning_rate=0.3, epochs=300, batch_size=100, random_state=1)


def test_initial_bias_matches_base_rate():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 3))
    y = np.zeros(n)
    y[:10] = 1.0  # 5% positive

    fit = train.ClassifierFit().fit(
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

    fit1 = train.ClassifierFit().fit(
        X, y, learning_rate=0.1, epochs=20, batch_size=32, random_state=42
    )
    fit2 = train.ClassifierFit().fit(
        X, y, learning_rate=0.1, epochs=20, batch_size=32, random_state=42
    )
    np.testing.assert_allclose(fit1.weights, fit2.weights)
    assert fit1.bias == fit2.bias

    fit3 = train.ClassifierFit().fit(
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

    fit = train.ClassifierFit().fit(
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

    fit = train.ClassifierFit().fit(
        X, y, learning_rate=0.3, epochs=200, batch_size=50, tolerance=0.0, random_state=5
    )

    loss_history = np.array(fit.loss_history)
    assert len(loss_history) > 10
    early_mean = loss_history[:10].mean()
    late_mean = loss_history[-10:].mean()
    assert late_mean < early_mean


@pytest.fixture
def divergent_data():
    rng = np.random.default_rng(6)
    n = 100
    X = rng.normal(size=(n, 2))
    y = np.array([0.0] * 50 + [1.0] * 50)
    rng.shuffle(y)
    return X, y


def test_diverges_with_huge_learning_rate(divergent_data):
    # sigmoid/log_loss clip predictions, so the loss itself stays bounded
    # (~11-15) no matter how large a full-batch gradient step is: the
    # per-step error is always in [-1, 1], so weights grow at most linearly
    # in learning_rate per epoch and the isfinite() guard never trips, even
    # at learning_rate=1e300. With batch_size=1 the update is a single raw
    # X_i * error term multiplied straight through by learning_rate, so a
    # learning_rate near float64's max (~1.8e308) overflows the weights to
    # inf in one step, producing the nan loss that guard is meant to catch.
    X, y = divergent_data
    with pytest.raises(ValueError):
        train.ClassifierFit().fit(
            X, y, learning_rate=1e308, epochs=5, batch_size=1, random_state=6
        )


def test_saturating_the_link_is_reported_as_divergence(divergent_data):
    # The case the isfinite() guard above cannot see: at learning_rate=50 the
    # weights run far enough that |eta| passes SIGMOID_EXP_CLIP, so sigmoid
    # and log_loss both clip and stop depending on the parameters. The loss
    # stays a perfectly finite ~15 while the run has stopped optimising
    # anything; family.saturated is what turns that into an error.
    X, y = divergent_data
    with pytest.raises(ValueError, match="saturated"):
        train.ClassifierFit().fit(
            X, y, learning_rate=50.0, epochs=5, batch_size=100, random_state=6
        )


def test_fit_can_be_cancelled(divergent_data):
    X, y = divergent_data
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(train.TrainingCancelled):
        train.ClassifierFit().fit(
            X, y, learning_rate=0.1, epochs=100, cancel_event=cancel_event
        )


def test_stops_early_for_trivially_separable_data():
    X = np.array([[-1.0], [-1.0], [1.0], [1.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])

    fit = train.ClassifierFit().fit(
        X, y, learning_rate=1.0, epochs=10_000, batch_size=4, tolerance=1e-2, random_state=7
    )

    assert fit.loss_history is not None
    assert len(fit.loss_history) < 10_000


# Configuration tests
def test_rejects_unknown_family_and_penalty():
    with pytest.raises(ValueError, match="Unsupported combination"):
        train.ClassifierFit(loss_function="mse", loss_function_link="identity")
    with pytest.raises(ValueError, match="Unknown penalty"):
        train.ClassifierFit(penalty="elastic_net")
    with pytest.raises(ValueError, match="non-negative"):
        train.ClassifierFit(penalty="ridge", delta=-0.1)


# Regularization tests
def test_delta_zero_is_identical_to_no_penalty(separable_ish_data):
    X, y = separable_ish_data

    unpenalized = _fit(X, y)
    ridge_zero = _fit(X, y, penalty="ridge", delta=0.0)

    np.testing.assert_array_equal(unpenalized.weights, ridge_zero.weights)
    assert unpenalized.bias == ridge_zero.bias


@pytest.mark.parametrize("penalty", ["ridge", "lasso"])
def test_larger_delta_shrinks_the_weights(separable_ish_data, penalty):
    X, y = separable_ish_data

    norms = [
        float(np.linalg.norm(_fit(X, y, penalty=penalty, delta=delta).weights))
        for delta in (0.0, 0.05, 0.5)
    ]

    assert norms[0] > norms[1] > norms[2]


def test_penalty_never_touches_the_bias_gradient(separable_ish_data):
    # ISLP p.241: the shrinkage penalty applies to the coefficients, not the
    # intercept. The weight gradient must change, the bias gradient must not.
    X, y = separable_ish_data
    model = _fit(X, y, penalty="ridge", delta=0.5)
    y_pred = model.predict(X)

    plain_dw, plain_db = core.log_loss_derivation(y, y_pred, X)
    penalized_dw, penalized_db = model.loss_gradient(y, y_pred, X)

    assert penalized_db == plain_db
    assert not np.allclose(penalized_dw, plain_dw)
    np.testing.assert_allclose(penalized_dw - plain_dw, 0.5 * 2.0 * model.weights)


def test_loss_reports_the_penalized_objective(separable_ish_data):
    X, y = separable_ish_data
    model = _fit(X, y, penalty="lasso", delta=0.1)
    y_pred = model.predict(X)

    expected = core.log_loss(y, y_pred) + 0.1 * core.lasso_penalty(model.weights)
    np.testing.assert_allclose(model.loss(y, y_pred), expected)
    np.testing.assert_allclose(model.penalty_value(), 0.1 * core.lasso_penalty(model.weights))


def test_lasso_shrinks_the_irrelevant_feature_hardest(separable_ish_data):
    # The third feature has a true coefficient of 0.05. ISLP 6.2.2: the l1
    # penalty pushes such coefficients towards zero far more aggressively than
    # the l2 penalty does at a comparable amount of total shrinkage.
    X, y = separable_ish_data

    unpenalized = _fit(X, y)
    lasso = _fit(X, y, penalty="lasso", delta=0.05)

    assert abs(lasso.weights[2]) < abs(unpenalized.weights[2])
    # the informative coefficients survive the same penalty
    assert abs(lasso.weights[0]) > 10 * abs(lasso.weights[2])


# Prediction tests
def test_predict_applies_the_inverse_link(separable_ish_data):
    X, y = separable_ish_data
    model = _fit(X, y)

    probabilities = model.predict(X)
    np.testing.assert_allclose(probabilities, core.sigmoid(model.linear_predictor(X)))
    assert np.all((probabilities > 0.0) & (probabilities < 1.0))


def test_predict_class_cuts_at_the_threshold(separable_ish_data):
    X, y = separable_ish_data
    model = _fit(X, y)
    probabilities = model.predict(X)

    np.testing.assert_array_equal(
        model.predict_class(X, threshold=0.8), (probabilities >= 0.8).astype(float)
    )
    assert set(np.unique(model.predict_class(X))) <= {0.0, 1.0}
    with pytest.raises(ValueError):
        model.predict_class(X, threshold=1.5)


def test_predicting_before_fitting_raises():
    with pytest.raises(ValueError, match="not trained"):
        train.ClassifierFit().predict(np.zeros((3, 2)))
