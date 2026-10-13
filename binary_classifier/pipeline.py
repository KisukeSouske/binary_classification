"""
Glue for the photo classifier:

    thumbnail -> PCA -> standardize -> ClassifierFit

plus cross-validated evaluation, saving and loading. PCA and the scaler are
fitted inside every training fold, so a test photo never influences how the
features it is judged on are built.
"""
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from binary_classifier import metrics, model_selection, preprocessing, train
from binary_classifier.decomposition import PCA
from binary_classifier.features import load_folders, session_groups, thumbnail


@dataclass(frozen=True)
class TrainingConfig:
    """Fixed up front, not tuned on the evaluation folds."""
    n_components: int = 10
    delta: float = 0.05
    learning_rate: float = 0.05
    epochs: int = 3000
    n_splits: int = 5
    random_state: int = 0


class PhotoClassifier:
    def __init__(self, class_names, pca: PCA, feature_mean, feature_std, model: train.ClassifierFit):
        self.class_names = list(class_names)  # index 0 -> label 0.0, index 1 -> label 1.0
        self.pca = pca
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.model = model

    def _features(self, X_raw) -> np.ndarray:
        return (self.pca.transform(X_raw) - self.feature_mean) / self.feature_std

    def predict_proba(self, X_raw) -> np.ndarray:
        """Probability of class_names[1] for each row of raw thumbnails."""
        return self.model.predict(self._features(X_raw))

    def predict_image(self, path) -> tuple[str, float]:
        """Returns (class_name, confidence in [0.5, 1])."""
        p_positive = float(self.predict_proba(thumbnail(path)[None, :])[0])
        label = int(p_positive >= 0.5)
        confidence = p_positive if label == 1 else 1 - p_positive
        return self.class_names[label], confidence

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            class_names=np.array(self.class_names),
            pca_mean=self.pca.mean,
            pca_components=self.pca.components,
            pca_explained_variance_ratio=self.pca.explained_variance_ratio,
            mean=self.feature_mean,
            std=self.feature_std,
            weights=self.model.weights,
            bias=np.array(self.model.bias),
        )

    @classmethod
    def load(cls, path):
        data = np.load(path)
        pca = PCA(n_components=data["pca_components"].shape[0])
        pca.mean = data["pca_mean"]
        pca.components = data["pca_components"]
        pca.explained_variance_ratio = data["pca_explained_variance_ratio"]
        model = train.ClassifierFit(weights=data["weights"], bias=float(data["bias"]))
        return cls([str(n) for n in data["class_names"]], pca, data["mean"], data["std"], model)


def fit_classifier(X_raw, y, class_names, config: TrainingConfig,
                   cancel_event: threading.Event | None = None) -> PhotoClassifier:
    """Fits PCA, scaler and model on exactly the rows it is given."""
    pca = PCA(config.n_components).fit(X_raw)
    Z, feature_mean, feature_std = preprocessing.standardize(pca.transform(X_raw))
    model = train.ClassifierFit(penalty="ridge", delta=config.delta)
    model.fit(
        Z,
        y,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        random_state=config.random_state,
        cancel_event=cancel_event,
    )
    return PhotoClassifier(class_names, pca, feature_mean, feature_std, model)


def _brightness_baseline(X_raw, y, train_idx, test_idx, config: TrainingConfig) -> np.ndarray:
    """One-feature logistic regression on mean brightness: the "obvious" model."""
    brightness = X_raw.mean(axis=1, keepdims=True)
    B_train, mean, std = preprocessing.standardize(brightness[train_idx])
    model = train.ClassifierFit().fit(
        B_train, y[train_idx], learning_rate=config.learning_rate,
        epochs=config.epochs, random_state=config.random_state,
    )
    return model.predict_class((brightness[test_idx] - mean) / std)


def cross_validate(X_raw, y, folds, class_names, config: TrainingConfig,
                   cancel_event: threading.Event | None = None) -> dict:
    """
    Out-of-fold evaluation: every sample is predicted exactly once, by a model
    (PCA + scaler + classifier) that never saw it during fitting.
    """
    n = len(y)
    y_pred = np.empty(n)
    y_prob = np.empty(n)
    y_baseline = np.empty(n)
    fold_accuracies = []
    for test_idx in folds:
        train_idx = model_selection.train_indices(test_idx, n)
        classifier = fit_classifier(X_raw[train_idx], y[train_idx], class_names, config, cancel_event)
        y_prob[test_idx] = classifier.predict_proba(X_raw[test_idx])
        y_pred[test_idx] = (y_prob[test_idx] >= 0.5).astype(float)
        y_baseline[test_idx] = _brightness_baseline(X_raw, y, train_idx, test_idx, config)
        fold_accuracies.append(metrics.accuracy(y[test_idx], y_pred[test_idx]))

    n_correct = int(np.sum(y_pred == y))
    n_baseline_correct = int(np.sum(y_baseline == y))
    report = metrics.classification_report(y, y_pred)
    report.update(
        n_samples=n,
        n_correct=n_correct,
        interval=metrics.wilson_interval(n_correct, n),
        fold_sizes=[len(f) for f in folds],
        fold_accuracies=fold_accuracies,
        baseline_accuracy=n_baseline_correct / n,
        baseline_interval=metrics.wilson_interval(n_baseline_correct, n),
        y_prob=y_prob,
    )
    return report


def train_on_folders(negative_folder, positive_folder, config: TrainingConfig = TrainingConfig(),
                     cancel_event: threading.Event | None = None):
    """
    Evaluates with two cross-validations - stratified folds and folds grouped
    by photo session - then fits the final model on every photo.
    Returns (final classifier, report dict).
    """
    negative_folder, positive_folder = Path(negative_folder), Path(positive_folder)
    class_names = [negative_folder.name, positive_folder.name]
    X_raw, y, paths = load_folders(negative_folder, positive_folder)
    groups = session_groups(paths)

    stratified = cross_validate(
        X_raw, y, model_selection.stratified_k_fold(y, config.n_splits, config.random_state),
        class_names, config, cancel_event,
    )
    by_session = cross_validate(
        X_raw, y, model_selection.group_k_fold(groups, config.n_splits, config.random_state),
        class_names, config, cancel_event,
    )

    classifier = fit_classifier(X_raw, y, class_names, config, cancel_event)

    # Errors and least confident photos, from the stratified out-of-fold predictions
    names = [p.name for p in paths]
    confidence = np.abs(stratified["y_prob"] - 0.5) + 0.5
    predicted = (stratified["y_prob"] >= 0.5).astype(float)
    report = {
        "config": config,
        "n_photos": len(y),
        "n_per_class": [int(np.sum(y == 0)), int(np.sum(y == 1))],
        "n_raw_features": X_raw.shape[1],
        "n_train_per_fold": len(y) - max(stratified["fold_sizes"]),
        "n_sessions": len(np.unique(groups)),
        "explained_variance": float(classifier.pca.explained_variance_ratio.sum()),
        "stratified": stratified,
        "by_session": by_session,
        "misclassified": [
            (names[i], class_names[int(predicted[i])], float(confidence[i]))
            for i in np.flatnonzero(predicted != y)
        ],
        "least_confident": [
            (names[i], class_names[int(predicted[i])], float(confidence[i]))
            for i in np.argsort(confidence)[:3]
        ],
    }
    return classifier, report


def format_report(report, class_names) -> str:
    config = report["config"]
    neg, pos = class_names

    def cv_lines(title, cv):
        low, high = cv["interval"]
        b_low, b_high = cv["baseline_interval"]
        folds = " ".join(f"{a:.0%}" for a in cv["fold_accuracies"])
        return [
            title,
            f"  Acurácia: {cv['n_correct']}/{cv['n_samples']} = {cv['accuracy']:.1%}"
            f"  (IC 95%: {low:.0%}–{high:.0%})",
            f"  Baseline brilho médio: {cv['baseline_accuracy']:.1%}  (IC 95%: {b_low:.0%}–{b_high:.0%})",
            f"  Por fold: {folds}",
        ]

    cm = report["stratified"]["confusion_matrix"]
    corner = "real \\ previsto"
    lines = [
        f"{report['n_photos']} fotos ({neg}: {report['n_per_class'][0]}, {pos}: {report['n_per_class'][1]})",
        f"Features: {report['n_raw_features']} pixels -> PCA {config.n_components} componentes",
        f"  ({report['explained_variance']:.0%} da variância; ~{report['n_train_per_fold']} fotos de treino por fold)",
        f"Ridge delta={config.delta} | {config.n_splits} folds",
        "",
        *cv_lines("Validação cruzada estratificada", report["stratified"]),
        "",
        *cv_lines(f"Validação agrupada por dia ({report['n_sessions']} grupos)", report["by_session"]),
        "",
        "Matriz de confusão (estratificada)",
        f"{corner:>16}{neg:>8}{pos:>8}",
        f"{neg:>16}{cm['tn']:>8}{cm['fp']:>8}",
        f"{pos:>16}{cm['fn']:>8}{cm['tp']:>8}",
        "",
    ]
    if report["misclassified"]:
        lines.append("Erros:")
        lines += [f"  {name} -> {pred} ({conf:.0%})" for name, pred, conf in report["misclassified"]]
    lines.append("Menor confiança:")
    lines += [f"  {name} -> {pred} ({conf:.0%})" for name, pred, conf in report["least_confident"]]
    lines += ["", "Modelo final treinado com todas as fotos."]
    return "\n".join(lines)
