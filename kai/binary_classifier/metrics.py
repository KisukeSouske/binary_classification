import numpy as np

def confusion_matrix(y_true: np.ndarray, y_pred_class: np.ndarray) -> dict[str, int]:
    tp = int(np.sum((y_true == 1) & (y_pred_class == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred_class == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred_class == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred_class == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

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

def f1_score(y_true: np.ndarray, y_pred_class: np.ndarray) -> float:
    prec = precision(y_true, y_pred_class)
    rec = recall(y_true, y_pred_class)
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

def metrics_by_threshold(
    y_true: np.ndarray, y_pred_prob: np.ndarray, thresholds: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Computes accuracy, precision, recall and f1 for each threshold.
    y_true: true labels (0 or 1)
    y_pred_prob: predicted probabilities (between 0 and 1)
    thresholds: array of threshold values to evaluate the metrics at
    Returns: dict mapping metric name to an array aligned with `thresholds`
    """
    results = {"accuracy": [], "precision": [], "recall": [], "f1_score": []}
    for threshold in thresholds:
        y_pred_class = (y_pred_prob >= threshold).astype(float)
        results["accuracy"].append(accuracy(y_true, y_pred_class))
        results["precision"].append(precision(y_true, y_pred_class))
        results["recall"].append(recall(y_true, y_pred_class))
        results["f1_score"].append(f1_score(y_true, y_pred_class))
    return {name: np.array(values) for name, values in results.items()}