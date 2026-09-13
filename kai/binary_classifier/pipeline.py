"""
Glue between features, scaler and model: train, evaluate, save, load, predict.
"""
from pathlib import Path

import numpy as np

from . import core
from .features import extract_features, load_dataset, load_image


class CatClassifier:
    def __init__(self, class_names, scaler, model):
        self.class_names = list(class_names)  # index 0 -> label 0, index 1 -> label 1
        self.scaler = scaler
        self.model = model

    def predict_image(self, path_or_image):
        """Returns (class_name, confidence in [0.5, 1])."""
        img = load_image(path_or_image) if isinstance(path_or_image, (str, Path)) else path_or_image
        x = self.scaler.transform(extract_features(img)[None, :])
        p_positive = float(self.model.predict_proba(x)[0])
        label = int(p_positive >= 0.5)
        confidence = p_positive if label == 1 else 1 - p_positive
        return self.class_names[label], confidence

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            class_names=np.array(self.class_names),
            mean=self.scaler.mean_,
            std=self.scaler.std_,
            weights=self.model.weights,
            bias=np.array(self.model.bias),
        )

    @classmethod
    def load(cls, path):
        data = np.load(path)
        scaler = core.StandardScaler()
        scaler.mean_, scaler.std_ = data["mean"], data["std"]
        model = core.LogisticRegression()
        model.weights, model.bias = data["weights"], float(data["bias"])
        return cls([str(n) for n in data["class_names"]], scaler, model)


def train(negative_folder, positive_folder, test_size=0.2, seed=0,
          learning_rate=0.05, n_iters=3000, l2=0.05):
    """
    Loads both folders, splits into train/test, trains and evaluates.
    Returns (classifier, report dict).
    """
    negative_folder, positive_folder = Path(negative_folder), Path(positive_folder)
    X, y, paths = load_dataset(negative_folder, positive_folder)
    paths = np.array(paths, dtype=object)

    idx = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = core.train_test_split(idx, y, test_size, seed)

    scaler = core.StandardScaler()
    X_train = scaler.fit_transform(X[idx_train])
    X_test = scaler.transform(X[idx_test])

    model = core.LogisticRegression(learning_rate, n_iters, l2).fit(X_train, y_train)
    classifier = CatClassifier([negative_folder.name, positive_folder.name], scaler, model)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    report = {
        "n_train": len(y_train),
        "n_test": len(y_test),
        "train_accuracy": core.accuracy(y_train, y_pred_train),
        "test_accuracy": core.accuracy(y_test, y_pred_test),
        "confusion_matrix": core.confusion_matrix(y_test, y_pred_test),
        "misclassified": [
            (paths[i].name, classifier.class_names[p])
            for i, t, p in zip(idx_test, y_test, y_pred_test) if t != p
        ],
        "final_loss": model.loss_history[-1],
    }
    return classifier, report


def format_report(report, class_names):
    cm = report["confusion_matrix"]
    lines = [
        f"Treino: {report['n_train']} fotos | Teste: {report['n_test']} fotos",
        f"Loss final (treino): {report['final_loss']:.4f}",
        f"Acurácia treino: {report['train_accuracy']:.1%}",
        f"Acurácia teste:  {report['test_accuracy']:.1%}",
        "",
        "Matriz de confusão (teste) — linha = real, coluna = previsto",
        f"{'':>10}{class_names[0]:>10}{class_names[1]:>10}",
        f"{class_names[0]:>10}{cm[0, 0]:>10}{cm[0, 1]:>10}",
        f"{class_names[1]:>10}{cm[1, 0]:>10}{cm[1, 1]:>10}",
    ]
    if report["misclassified"]:
        lines.append("")
        lines.append("Erros no teste:")
        lines += [f"  {name} -> previsto {pred}" for name, pred in report["misclassified"]]
    return "\n".join(lines)
