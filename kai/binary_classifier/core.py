import numpy as np

SIGMOID_EXP_CLIP = 30  # Clip values to avoid overflow in exp; also stays below the
                        # ~36.7 threshold where exp(-x) underflows to 0 in float64,
                        # which would saturate sigmoid to exactly 1.0

def sigmoid(x):
    """
    Sigmoid activation function.
    Returns a value between 0 and 1.
    """
    x = np.clip(x, -SIGMOID_EXP_CLIP, SIGMOID_EXP_CLIP)
    return 1 / (1 + np.exp(-x))

def logit(p, eps=1e-12):
    """
    Inverse of the sigmoid function (log-odds).
    Maps a probability in (0, 1) to the real line.
    """
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def binary_cross_entropy_gradient(y_true, y_pred, X) -> tuple[np.ndarray, float]:
    """
    Computes the gradient of the binary cross-entropy loss with respect to weights and bias.
    y_true: true labels (0 or 1)
    y_pred: predicted probabilities (between 0 and 1)
    X: input features (numpy array of shape [n_samples, n_features])
    Returns: gradients with respect to weights and bias
    """
    n_samples = X.shape[0]
    error = y_pred - y_true
    dw = (1 / n_samples) * np.dot(X.T, error)
    db = (1 / n_samples) * np.sum(error)
    return dw, db

def confusion_matrix(y_true: np.ndarray, y_pred_class: np.ndarray) -> dict[str, int]:
    tp = int(np.sum((y_true == 1) & (y_pred_class == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred_class == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred_class == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred_class == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}