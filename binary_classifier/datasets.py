"""Loaders for the datasets used in the experiments.

Each dataset owns its source URL and its column layout here, so the pipeline
scripts only deal with plain feature/label arrays.
"""
import numpy as np
import pandas as pd

RICE_CSV_URL = (
    "https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv"
)
RICE_FEATURE_COLUMNS = [
    "Area",
    "Perimeter",
    "Major_Axis_Length",
    "Minor_Axis_Length",
    "Eccentricity",
    "Convex_Area",
    "Extent",
]
RICE_LABEL_COLUMN = "Class"
RICE_POSITIVE_CLASS = "Cammeo"


def load_rice_dataframe() -> pd.DataFrame:
    """Download the rice dataset as a raw DataFrame."""
    return pd.read_csv(RICE_CSV_URL)


def rice_features_and_labels(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the modelling arrays from the raw rice DataFrame.
    Returns: X of shape [n_samples, n_features] and y of 0.0/1.0 labels,
    where 1.0 means the positive class (RICE_POSITIVE_CLASS).
    """
    X = frame[RICE_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = (frame[RICE_LABEL_COLUMN] == RICE_POSITIVE_CLASS).to_numpy(dtype=float)
    return X, y
