"""
Turns photos into fixed-length feature vectors for the logistic regression.

Each image becomes:
  - a small grayscale thumbnail (coarse shape / where the dark and light fur is)
  - HSV color histograms (fur color and eye color, regardless of position)
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
THUMB_SIZE = 32   # thumbnail side, in pixels
HIST_BINS = 16    # bins per HSV channel


def load_image(path):
    """Opens an image as RGB, applying the phone's EXIF rotation."""
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def extract_features(img):
    """Feature vector for one PIL image (RGB)."""
    # Center crop to a square so every photo is compared on the same framing
    square = ImageOps.fit(img, (THUMB_SIZE * 4, THUMB_SIZE * 4))

    thumb = np.asarray(square.convert("L").resize((THUMB_SIZE, THUMB_SIZE)), dtype=float) / 255.0

    hsv = np.asarray(square.convert("HSV"))
    hists = [
        np.histogram(hsv[..., channel], bins=HIST_BINS, range=(0, 256))[0]
        for channel in range(3)
    ]
    hists = np.concatenate(hists).astype(float) / hsv[..., 0].size

    return np.concatenate([thumb.ravel(), hists])


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_dataset(negative_folder, positive_folder):
    """
    Loads every image in both folders.
    Label 0 = negative_folder, label 1 = positive_folder.
    Returns X, y and the list of file paths (same order).
    """
    X, y, paths = [], [], []
    for label, folder in enumerate([negative_folder, positive_folder]):
        for path in list_images(folder):
            X.append(extract_features(load_image(path)))
            y.append(label)
            paths.append(path)
    if not X:
        raise ValueError("No images found in the given folders.")
    return np.array(X), np.array(y), paths
