import matplotlib.pyplot as plt

from binary_classifier import core, datasets, train, preprocessing, metrics, plotting

RANDOM_STATE = 42
TEST_FRACTION = 0.2


def main():
    rice_dataset_raw = datasets.load_rice_dataframe()
    print(f"Loaded {len(rice_dataset_raw)} rows")
    print(rice_dataset_raw.head())

    X, y = datasets.rice_features_and_labels(rice_dataset_raw)
    print(f"Base rate ({datasets.RICE_POSITIVE_CLASS}): {y.mean():.3f}")

    X_train, X_test, y_train, y_test = preprocessing.train_test_split(
        X, y, TEST_FRACTION, RANDOM_STATE
    )
    X_train, feature_mean, feature_std = preprocessing.standardize(X_train)
    X_test = (X_test - feature_mean) / feature_std

    model = train.ClassifierFit()
    model.fit(
        X_train,
        y_train,
        learning_rate=0.1,
        batch_size=64,
        epochs=40000,
        random_state=RANDOM_STATE,
    )
    print(f"Trained for {len(model.loss_history)} epochs, final train loss={model.loss_history[-1]:.4f}")

    y_pred_prob = model.predict(X_test)
    y_pred_class = model.predict_class(X_test)

    report = metrics.classification_report(y_test, y_pred_class)
    print(f"Test loss: {core.log_loss(y_test, y_pred_prob):.4f}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall: {report['recall']:.4f}")
    print(f"F1 score: {report['f1_score']:.4f}")
    print(f"Confusion matrix: {report['confusion_matrix']}")

    plotting.plot_metrics_vs_threshold(y_test, y_pred_prob)
    plt.show()


if __name__ == "__main__":
    main()
