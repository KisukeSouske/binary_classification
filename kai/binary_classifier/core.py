import numpy as np

SIGMOID_EXP_CLIP = 30  # Clip values to avoid overflow in exp; also stays below the
                        # ~36.7 threshold where exp(-x) underflows to 0 in float64,
                        # which would saturate sigmoid to exactly 1.0
LOG_EPS = 1e-12  # Keeps log() away from 0 in the cross-entropy


def sigmoid(x):
    """
    Sigmoid activation function.
    Returns a value between 0 and 1.
    """
    x = np.clip(x, -SIGMOID_EXP_CLIP, SIGMOID_EXP_CLIP)
    return 1 / (1 + np.exp(-x))


def binary_cross_entropy(y_true, y_prob):
    """
    Mean binary cross-entropy between labels in {0, 1} and predicted probabilities.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), LOG_EPS, 1 - LOG_EPS)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))


def train_test_split(X, y, test_size=0.2, seed=0):
    """
    Stratified split: each class keeps the same test proportion, so a small
    dataset never ends up with a test set missing one of the classes.
    Returns X_train, X_test, y_train, y_test.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    test_idx = []
    for label in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == label))
        n_test = max(1, int(round(len(idx) * test_size)))
        test_idx.extend(idx[:n_test])
    test_mask = np.zeros(len(y), dtype=bool)
    test_mask[test_idx] = True
    return X[~test_mask], X[test_mask], y[~test_mask], y[test_mask]


class StandardScaler:
    """
    Standardizes features to zero mean and unit variance, using statistics from
    the training set only.
    """

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # constant features stay at 0 instead of NaN
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LogisticRegression:
    """
    Binary logistic regression trained with batch gradient descent and L2
    regularization (the bias is not regularized).
    """

    def __init__(self, learning_rate=0.1, n_iters=2000, l2=0.01):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.l2 = l2
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iters):
            y_prob = sigmoid(X @ self.weights + self.bias)
            error = y_prob - y
            grad_w = X.T @ error / n_samples + self.l2 * self.weights
            grad_b = error.mean()
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
            loss = binary_cross_entropy(y, y_prob) + 0.5 * self.l2 * np.sum(self.weights ** 2)
            self.loss_history.append(loss)
        return self

    def predict_proba(self, X):
        """Probability of the positive class (label 1)."""
        if self.weights is None:
            raise RuntimeError("Model is not trained. Call fit() first.")
        return sigmoid(np.asarray(X, dtype=float) @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred):
    """
    2x2 matrix [[TN, FP], [FN, TP]]: rows are the true class, columns the predicted one.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    matrix = np.zeros((2, 2), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix