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

def test_binary_cross_entropy():
    # Perfect predictions give ~0 loss, a coin flip gives log(2)
    assert core.binary_cross_entropy([1, 0], [1.0, 0.0]) < 1e-6
    np.testing.assert_almost_equal(core.binary_cross_entropy([1, 0], [0.5, 0.5]), np.log(2))

def test_train_test_split_is_stratified():
    # Every class must show up in the test set with roughly the requested proportion
    X = np.arange(100).reshape(50, 2)
    y = np.array([0] * 20 + [1] * 30)
    X_train, X_test, y_train, y_test = core.train_test_split(X, y, test_size=0.2, seed=1)
    assert len(X_train) + len(X_test) == 50
    assert np.sum(y_test == 0) == 4
    assert np.sum(y_test == 1) == 6
    # No sample appears in both sets
    assert not set(X_train[:, 0]) & set(X_test[:, 0])

def test_standard_scaler():
    X = np.array([[1.0, 5.0], [3.0, 5.0]])
    Z = core.StandardScaler().fit_transform(X)
    np.testing.assert_almost_equal(Z[:, 0], [-1, 1])
    np.testing.assert_almost_equal(Z[:, 1], [0, 0])  # constant column doesn't become NaN

def test_logistic_regression_learns_separable_data():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 1, (50, 2)), rng.normal(2, 1, (50, 2))])
    y = np.array([0] * 50 + [1] * 50)
    model = core.LogisticRegression(learning_rate=0.1, n_iters=500, l2=0.0).fit(X, y)
    assert core.accuracy(y, model.predict(X)) > 0.95
    # Loss must go down during training
    assert model.loss_history[-1] < model.loss_history[0]

def test_confusion_matrix():
    cm = core.confusion_matrix([0, 0, 1, 1, 1], [0, 1, 1, 1, 0])
    np.testing.assert_array_equal(cm, [[1, 1], [1, 2]])