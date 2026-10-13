from binary_classifier import metrics
import numpy as np
import pytest


@pytest.fixture
def y_true_pred():
    # simple binary classification example
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])
    return y_true, y_pred


# Confusion matrix tests
def test_confusion_matrix_basic(y_true_pred):
    y_true, y_pred = y_true_pred
    cm = metrics.confusion_matrix(y_true, y_pred)
    assert cm == {"tp": 3, "tn": 3, "fp": 1, "fn": 1}


# Accuracy, precision, recall, f1 tests
def test_accuracy_precision_recall_f1(y_true_pred):
    y_true, y_pred = y_true_pred

    # accuracy = (tp+tn)/total = (3+3)/8 = 0.75
    assert pytest.approx(metrics.accuracy(y_true, y_pred), rel=1e-6) == 0.75

    # precision = tp / (tp+fp) = 3/(3+1) = 0.75
    assert pytest.approx(metrics.precision(y_true, y_pred), rel=1e-6) == 0.75

    # recall = tp / (tp+fn) = 3/(3+1) = 0.75
    assert pytest.approx(metrics.recall(y_true, y_pred), rel=1e-6) == 0.75

    # f1 = 2 * prec * rec / (prec + rec) = 0.75
    assert pytest.approx(metrics.f1_score(y_true, y_pred), rel=1e-6) == 0.75


def test_precision_with_no_positive_predictions_is_zero():
    y_true = np.array([1.0, 0.0, 1.0])
    y_pred_class = np.array([0.0, 0.0, 0.0])
    assert metrics.precision(y_true, y_pred_class) == 0.0


def test_recall_with_no_actual_positives_is_zero():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred_class = np.array([1.0, 0.0, 1.0])
    assert metrics.recall(y_true, y_pred_class) == 0.0


def test_f1_score_is_zero_when_precision_and_recall_are_zero():
    y_true = np.array([1.0, 1.0])
    y_pred_class = np.array([0.0, 0.0])
    assert metrics.f1_score(y_true, y_pred_class) == 0.0


def test_mismatched_length_raises():
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 0])
    with pytest.raises(ValueError):
        metrics.accuracy(y_true, y_pred)


# Classification report tests
def test_classification_report_matches_the_individual_metrics(y_true_pred):
    y_true, y_pred = y_true_pred
    report = metrics.classification_report(y_true, y_pred)

    assert report["accuracy"] == metrics.accuracy(y_true, y_pred)
    assert report["precision"] == metrics.precision(y_true, y_pred)
    assert report["recall"] == metrics.recall(y_true, y_pred)
    assert report["f1_score"] == metrics.f1_score(y_true, y_pred)
    assert report["confusion_matrix"] == metrics.confusion_matrix(y_true, y_pred)


# Metrics-by-threshold tests
def test_metrics_by_threshold_matches_pointwise_metrics():
    rng = np.random.default_rng(0)
    y_true = (rng.random(50) > 0.5).astype(float)
    y_pred_prob = rng.random(50)
    thresholds = np.array([0.2, 0.5, 0.8])

    curves = metrics.metrics_by_threshold(y_true, y_pred_prob, thresholds)

    for name in ("accuracy", "precision", "recall", "f1_score"):
        assert name in curves
        assert curves[name].shape == thresholds.shape

    for i, threshold in enumerate(thresholds):
        y_pred_class = (y_pred_prob >= threshold).astype(float)
        assert curves["accuracy"][i] == metrics.accuracy(y_true, y_pred_class)
        assert curves["precision"][i] == metrics.precision(y_true, y_pred_class)
        assert curves["recall"][i] == metrics.recall(y_true, y_pred_class)


def test_metrics_by_threshold_recall_decreases_as_threshold_increases():
    rng = np.random.default_rng(1)
    y_true = (rng.random(200) > 0.5).astype(float)
    y_pred_prob = np.clip(y_true * 0.6 + rng.normal(0, 0.25, 200) + 0.2, 0, 1)
    thresholds = np.linspace(0.0, 1.0, 11)

    curves = metrics.metrics_by_threshold(y_true, y_pred_prob, thresholds)

    recall = curves["recall"]
    assert all(recall[i] >= recall[i + 1] for i in range(len(recall) - 1))


# Wilson interval tests
def test_wilson_interval_perfect_score_is_not_certain():
    # 17/17 on a small test set: the upper end reaches 1, the lower end
    # must stay well below it
    low, high = metrics.wilson_interval(17, 17)
    assert high == pytest.approx(1.0)
    assert 0.75 < low < 0.85


def test_wilson_interval_contains_the_estimate_and_narrows_with_n():
    low_small, high_small = metrics.wilson_interval(8, 10)
    low_big, high_big = metrics.wilson_interval(800, 1000)
    assert low_small < 0.8 < high_small
    assert low_big < 0.8 < high_big
    assert (high_big - low_big) < (high_small - low_small)


def test_wilson_interval_known_value():
    # 50/100, z=1.96: center 0.5, half width 1.96*sqrt(0.25/100 + 1.96^2/40000)/(1+1.96^2/100)
    low, high = metrics.wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-4)
    assert high == pytest.approx(0.5962, abs=1e-4)


def test_wilson_interval_rejects_invalid_counts():
    with pytest.raises(ValueError):
        metrics.wilson_interval(5, 0)
    with pytest.raises(ValueError):
        metrics.wilson_interval(11, 10)