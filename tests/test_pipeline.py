from binary_classifier import pipeline
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def two_folders(tmp_path):
    """Two classes of noisy synthetic photos: dark left half vs dark right half."""
    rng = np.random.default_rng(0)
    folders = []
    for name, dark_side in [("Left", slice(0, 20)), ("Right", slice(20, 40))]:
        folder = tmp_path / name
        folder.mkdir()
        for i in range(12):
            pixels = rng.normal(180, 30, size=(40, 40))
            pixels[:, dark_side] -= 120
            Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).save(folder / f"{i:02d}.png")
        folders.append(folder)
    return folders


CONFIG = pipeline.TrainingConfig(n_components=3, n_splits=3, epochs=300, learning_rate=0.1)


def test_features_given_to_the_model_are_fewer_than_training_samples(two_folders):
    classifier, report = pipeline.train_on_folders(*two_folders, CONFIG)

    assert classifier.model.weights.shape == (CONFIG.n_components,)
    assert CONFIG.n_components < report["n_train_per_fold"]


def test_every_photo_is_predicted_once_out_of_fold(two_folders):
    _, report = pipeline.train_on_folders(*two_folders, CONFIG)

    for cv in (report["stratified"], report["by_session"]):
        assert cv["n_samples"] == 24
        assert sum(cv["fold_sizes"]) == 24
        assert len(cv["fold_accuracies"]) == CONFIG.n_splits
        low, high = cv["interval"]
        assert low <= cv["accuracy"] <= high


def test_learns_an_easy_problem_and_round_trips_through_save(two_folders, tmp_path):
    classifier, report = pipeline.train_on_folders(*two_folders, CONFIG)
    assert report["stratified"]["accuracy"] > 0.9

    path = tmp_path / "model.npz"
    classifier.save(path)
    loaded = pipeline.PhotoClassifier.load(path)

    photo = two_folders[1] / "00.png"
    assert loaded.predict_image(photo) == pytest.approx(classifier.predict_image(photo))
    assert loaded.predict_image(photo)[0] == "Right"
