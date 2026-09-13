"""
Glue for the photo classifier: features -> standardize -> ClassifierFit, plus
evaluation, saving and loading.
"""
import threading
from pathlib import Path

import numpy as np

from binary_classifier import metrics, preprocessing, train
from binary_classifier.features import extract_features, load_folders


class PhotoClassifier:
    def __init__(self, class_names, feature_mean, feature_std, model: train.ClassifierFit):
        self.class_names = list(class_names)  # index 0 -> label 0.0, index 1 -> label 1.0
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.model = model

    def predict_image(self, path) -> tuple[str, float]:
        """Returns (class_name, confidence in [0.5, 1])."""
        x = (extract_features(path)[None, :] - self.feature_mean) / self.feature_std
        p_positive = float(self.model.predict(x)[0])
        label = int(p_positive >= 0.5)
        confidence = p_positive if label == 1 else 1 - p_positive
        return self.class_names[label], confidence

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            class_names=np.array(self.class_names),
            mean=self.feature_mean,
            std=self.feature_std,
            weights=self.model.weights,
            bias=np.array(self.model.bias),
        )

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = train.ClassifierFit(weights=data["weights"], bias=float(data["bias"]))
        return cls([str(n) for n in data["class_names"]], data["mean"], data["std"], model)


def train_on_folders(
    negative_folder,
    positive_folder,
    test_fraction: float = 0.2,
    random_state: int = 0,
    learning_rate: float = 0.05,
    delta: float = 0.05,
    epochs: int = 3000,
    cancel_event: threading.Event | None = None,
):
    """
    Loads both folders, splits into train/test, trains a ridge-penalized
    logistic regression and evaluates it on the held-out photos.
    Returns (classifier, report dict).
    """
    negative_folder, positive_folder = Path(negative_folder), Path(positive_folder)
    X, y, paths = load_folders(negative_folder, positive_folder)

    # Split indices instead of X so the test photos can be named in the report
    indices = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = preprocessing.train_test_split(
        indices, y, test_fraction, random_state
    )
    X_train, feature_mean, feature_std = preprocessing.standardize(X[idx_train])
    X_test = (X[idx_test] - feature_mean) / feature_std

    model = train.ClassifierFit(penalty="ridge", delta=delta)
    model.fit(
        X_train,
        y_train,
        learning_rate=learning_rate,
        epochs=epochs,
        random_state=random_state,
        cancel_event=cancel_event,
    )
    classifier = PhotoClassifier(
        [negative_folder.name, positive_folder.name], feature_mean, feature_std, model
    )

    y_pred_test = model.predict_class(X_test)
    report = metrics.classification_report(y_test, y_pred_test)
    report.update(
        n_train=len(y_train),
        n_test=len(y_test),
        epochs=len(model.loss_history),
        final_loss=model.loss_history[-1],
        train_accuracy=metrics.accuracy(y_train, model.predict_class(X_train)),
        misclassified=[
            (paths[i].name, classifier.class_names[int(p)])
            for i, t, p in zip(idx_test, y_test, y_pred_test) if t != p
        ],
    )
    return classifier, report


def format_report(report, class_names) -> str:
    cm = report["confusion_matrix"]
    neg, pos = class_names
    lines = [
        f"Treino: {report['n_train']} fotos | Teste: {report['n_test']} fotos",
        f"Épocas: {report['epochs']} | loss final (treino): {report['final_loss']:.4f}",
        f"Acurácia treino: {report['train_accuracy']:.1%}",
        f"Acurácia teste:  {report['accuracy']:.1%}",
        f"Precisão ({pos}): {report['precision']:.1%} | Recall: {report['recall']:.1%} | F1: {report['f1_score']:.3f}",
        "",
        "Matriz de confusão (teste) — linha = real, coluna = previsto",
        f"{'':>10}{neg:>10}{pos:>10}",
        f"{neg:>10}{cm['tn']:>10}{cm['fp']:>10}",
        f"{pos:>10}{cm['fn']:>10}{cm['tp']:>10}",
    ]
    if report["misclassified"]:
        lines.append("")
        lines.append("Erros no teste:")
        lines += [f"  {name} -> previsto {pred}" for name, pred in report["misclassified"]]
    return "\n".join(lines)
