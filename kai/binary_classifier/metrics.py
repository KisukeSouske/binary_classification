import numpy as np

def log_loss(y_true, y_pred, eps=1e-12) -> float:
    """
    Computes the log loss (binary cross-entropy loss) for binary classification.
    y_true: true labels (0 or 1)
    y_pred: predicted probabilities (between 0 and 1)
    """
    # Clip predictions to avoid log(0) which is undefined
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def accuracy(y_true: np.ndarray, y_pred_class: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred_class))

def precision(y_true: np.ndarray, y_pred_class: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred_class == 1))
    fp = np.sum((y_true == 0) & (y_pred_class == 1))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

def recall(y_true: np.ndarray, y_pred_class: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred_class == 1))
    fn = np.sum((y_true == 1) & (y_pred_class == 0))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0