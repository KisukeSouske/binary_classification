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

def log_loss(y_true, y_pred, eps=1e-12) -> float:
    """
    Computes the log loss (binary cross-entropy loss) for binary classification.
    y_true: true labels (0 or 1)
    y_pred: predicted probabilities (between 0 and 1)
    """
    # Clip predictions to avoid log(0) which is undefined
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))