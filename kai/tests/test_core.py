from binary_classifier import core
import numpy as np

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