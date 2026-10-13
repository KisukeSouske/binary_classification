import numpy as np

def confusion_matrix(y_true: np.ndarray, y_pred_class: np.ndarray) -> dict[str, int]:
    tp = int(np.sum((y_true == 1) & (y_pred_class == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred_class == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred_class == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred_class == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

def accuracy(y_true: np.ndarray, y_pred_class: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred_class))

def wilson_interval(n_correct: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score confidence interval for a proportion such as accuracy.
    Unlike the normal approximation p +- z*sqrt(p(1-p)/n), it stays inside
    [0, 1] and does not collapse to a zero-width interval at p = 0 or p = 1,
    which is exactly the case of a "perfect" score on a small test set.
    n_correct: number of successes
    n_total: number of trials
    z: normal quantile (1.96 -> 95% interval)
    Returns: (lower, upper)
    """
    if n_total <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}.")
    if not 0 <= n_correct <= n_total:
        raise ValueError(f"n_correct must be in [0, {n_total}], got {n_correct}.")
    p = n_correct / n_total
    denominator = 1 + z ** 2 / n_total
    center = (p + z ** 2 / (2 * n_total)) / denominator
    half_width = z * np.sqrt(p * (1 - p) / n_total + z ** 2 / (4 * n_total ** 2)) / denominator
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))

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

def classification_report(
    y_true: np.ndarray, y_pred_class: np.ndarray
) -> dict[str, float | dict[str, int]]:
    """
    Computes every summary metric for one set of hard predictions.
    y_true: true labels (0 or 1)
    y_pred_class: predicted labels (0 or 1)
    Returns: dict with accuracy, precision, recall, f1_score and confusion_matrix
    """
    return {
        "accuracy": accuracy(y_true, y_pred_class),
        "precision": precision(y_true, y_pred_class),
        "recall": recall(y_true, y_pred_class),
        "f1_score": f1_score(y_true, y_pred_class),
        "confusion_matrix": confusion_matrix(y_true, y_pred_class),
    }

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