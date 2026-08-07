import numpy as np
import pandas as pd

from binary_classifier import core, train

RANDOM_STATE = 42
TEST_FRACTION = 0.2
FEATURE_COLUMNS = [
    "Area",
    "Perimeter",
    "Major_Axis_Length",
    "Minor_Axis_Length",
    "Eccentricity",
    "Convex_Area",
    "Extent",
]
LABEL_COLUMN = "Class"
POSITIVE_CLASS = "Cammeo"


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(
        "https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv"
    )


def train_test_split(X: np.ndarray, y: np.ndarray, test_fraction: float, random_state: int):
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(y))
    n_test = int(len(y) * test_fraction)
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    return (X_train - mean) / std, (X_test - mean) / std


def accuracy(y_true: np.ndarray, y_pred_class: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred_class))


def confusion_matrix(y_true: np.ndarray, y_pred_class: np.ndarray) -> dict[str, int]:
    tp = int(np.sum((y_true == 1) & (y_pred_class == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred_class == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred_class == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred_class == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def main():
    rice_dataset_raw = load_dataset()
    print(f"Loaded {len(rice_dataset_raw)} rows")
    print(rice_dataset_raw.head())

    X = rice_dataset_raw[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = (rice_dataset_raw[LABEL_COLUMN] == POSITIVE_CLASS).to_numpy(dtype=float)
    print(f"Base rate ({POSITIVE_CLASS}): {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, TEST_FRACTION, RANDOM_STATE
    )
    X_train, X_test = standardize(X_train, X_test)

    fit = train.fit_gradient_descent(
        X_train,
        y_train,
        learning_rate=0.1,
        batch_size=64,
        epochs=500,
        tolerance=1e-5,
        random_state=RANDOM_STATE,
    )
    print(f"Trained for {len(fit.loss_history)} epochs, final train loss={fit.loss_history[-1]:.4f}")

    y_pred_prob = fit.sigmoid_predictor(X_test)
    y_pred_class = (y_pred_prob >= 0.5).astype(float)

    test_loss = core.log_loss(y_test, y_pred_prob)
    test_accuracy = accuracy(y_test, y_pred_class)
    matrix = confusion_matrix(y_test, y_pred_class)

    print(f"Test log loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Confusion matrix: {matrix}")


if __name__ == "__main__":
    main()
