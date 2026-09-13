"""
Turns photos into fixed-length feature vectors for the classifier.

Each image becomes:
  - a small grayscale thumbnail (coarse shape / where the dark and light fur is)
  - HSV color histograms (fur color and eye color, regardless of position)
"""
from pathlib import Path

import numpy as np
from PIL import Image

from binary_classifier import images

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
THUMB_SIZE = 32   # thumbnail side, in pixels
BLOCK = 4         # the image is loaded at THUMB_SIZE * BLOCK and averaged down
HIST_BINS = 16    # bins per HSV channel
LUMA = np.array([0.299, 0.587, 0.114])  # same weights Pillow uses for RGB -> L


def extract_features(path) -> np.ndarray:
    """Feature vector for one image file."""
    side = THUMB_SIZE * BLOCK
    rgb = images.load_image(path, size=(side, side), color_mode="rgb")

    # Block-average the luma so the thumbnail comes from the same single load
    gray = rgb @ LUMA
    thumb = gray.reshape(THUMB_SIZE, BLOCK, THUMB_SIZE, BLOCK).mean(axis=(1, 3))

    hsv = np.asarray(Image.fromarray(np.round(rgb * 255).astype(np.uint8)).convert("HSV"))
    hists = [
        np.histogram(hsv[..., channel], bins=HIST_BINS, range=(0, 256))[0]
        for channel in range(3)
    ]
    hists = np.concatenate(hists).astype(float) / (side * side)

    return np.concatenate([thumb.ravel(), hists])


def list_images(folder) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_folders(negative_folder, positive_folder) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """
    Loads every image in both folders.
    Returns X, y (0.0 = negative_folder, 1.0 = positive_folder) and the file
    paths in the same order.
    """
    X, y, paths = [], [], []
    for label, folder in enumerate([negative_folder, positive_folder]):
        for path in list_images(folder):
            X.append(extract_features(path))
            y.append(float(label))
            paths.append(path)
    if not X:
        raise ValueError("No images found in the given folders.")
    return np.array(X), np.array(y), paths
