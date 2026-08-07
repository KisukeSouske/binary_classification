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
