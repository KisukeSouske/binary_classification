from binary_classifier import core
import numpy as np

# Sigmoid tests
def test_sigmoid():
    # Test that the sigmoid function returns values between 0 and 1
    assert core.sigmoid(0) == 0.5
    assert core.sigmoid(-100) < 0.01
    assert core.sigmoid(100) > 0.99

    # Test that the sigmoid function is monotonically increasing
    x_values = [-10, -5, 0, 5, 10]
    sigmoid_values = [core.sigmoid(x) for x in x_values]
    assert all(sigmoid_values[i] < sigmoid_values[i + 1] for i in range(len(sigmoid_values) - 1))

def test_sigmoid_as_numpy_array():
    # Test that the sigmoid function can handle numpy arrays
    x = np.array([-1, 0, 1])
    expected = np.array([0.26894142, 0.5, 0.73105858])
    np.testing.assert_almost_equal(core.sigmoid(x), expected, decimal=6)

def test_sigmoid_clip():
    # Test that the sigmoid function clips large values to avoid overflow
    assert core.sigmoid(1000) < 1.0
    assert core.sigmoid(-1000) > 0.0

def test_sigmoid_saturates_at_clip_boundary():
    # Values beyond SIGMOID_EXP_CLIP should be clipped to the same result as
    # the boundary itself, and never fully saturate to 0.0 or 1.0.
    clip = core.SIGMOID_EXP_CLIP
    assert core.sigmoid(clip) < 1.0
    assert core.sigmoid(-clip) > 0.0
    assert core.sigmoid(clip + 1000) == core.sigmoid(clip)
    assert core.sigmoid(-clip - 1000) == core.sigmoid(-clip)

# Log loss tests
def test_log_loss_almost_zero():
    # Test that log loss is almost zero for perfect predictions
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.01, 0.99, 0.99, 0.01])
    np.testing.assert_almost_equal(core.log_loss(y_true, y_pred), 0.0, decimal=2)

def test_log_loss_high():
    # Test that log loss is high for completely wrong predictions
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.99, 0.01, 0.01, 0.99])
    assert core.log_loss(y_true, y_pred) > 0.9

def test_log_loss_with_clipping():
    # Test that log loss handles predictions that are exactly 0 or 1 by clipping
    y_true = np.array([0, 1])
    y_pred = np.array([0.0, 1.0])  # These would cause log(0) without clipping
    loss = core.log_loss(y_true, y_pred)
    assert loss < 1e-10  # Should be very small due to clipping

def test_log_loss_hand_calculated():
    # -ln(0.5) for a single sample with y_true=1, y_pred=0.5
    y_true = np.array([1])
    y_pred = np.array([0.5])
    np.testing.assert_allclose(core.log_loss(y_true, y_pred), 0.6931471805599453)

def test_log_loss_is_the_mean_over_samples():
    # Two samples with known individual losses; log_loss should average them.
    y_true = np.array([1, 0])
    y_pred = np.array([0.5, 0.25])
    expected = (-np.log(0.5) + -np.log(0.75)) / 2
    np.testing.assert_allclose(core.log_loss(y_true, y_pred), expected)

# Logit tests
def test_logit_is_inverse_of_sigmoid():
    x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    np.testing.assert_allclose(core.logit(core.sigmoid(x)), x, atol=1e-6)

def test_logit_edge_cases_are_finite():
    # p=0 and p=1 would be -inf/+inf without clipping; logit should clip
    # like log_loss does and stay finite.
    assert np.isfinite(core.logit(0.0))
    assert np.isfinite(core.logit(1.0))
    assert core.logit(0.0) < 0
    assert core.logit(1.0) > 0

# Gradient checking
def test_binary_cross_entropy_gradient_matches_numerical_gradient():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 3))
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    w = rng.normal(size=3)
    b = 0.2
    h = 1e-6

    def loss(w, b):
        y_pred = core.sigmoid(X @ w + b)
        return core.log_loss(y, y_pred)

    y_pred = core.sigmoid(X @ w + b)
    analytic_dw, analytic_db = core.binary_cross_entropy_gradient(y, y_pred, X)

    numeric_dw = np.zeros_like(w)
    for i in range(len(w)):
        w_plus, w_minus = w.copy(), w.copy()
        w_plus[i] += h
        w_minus[i] -= h
        numeric_dw[i] = (loss(w_plus, b) - loss(w_minus, b)) / (2 * h)
    numeric_db = (loss(w, b + h) - loss(w, b - h)) / (2 * h)

    np.testing.assert_allclose(analytic_dw, numeric_dw, atol=1e-6)
    np.testing.assert_allclose(analytic_db, numeric_db, atol=1e-6)
