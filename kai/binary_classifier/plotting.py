import numpy as np
import matplotlib.pyplot as plt

from binary_classifier import metrics

METRIC_COLORS = {
    "accuracy": "#4C78A8",
    "precision": "#F58518",
    "recall": "#54A24B",
    "f1_score": "#B279A2",
}


def plot_metrics_vs_threshold(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
    initial_threshold: float = 0.5,
):
    """
    Plots accuracy/precision/recall/f1 curves against the decision threshold.

    A draggable vertical dashed line lets the user click-and-drag along the
    x-axis to inspect the metric values at a given threshold; the legend
    labels update live with the values under the line.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 201)

    curves = metrics.metrics_by_threshold(y_true, y_pred_prob, thresholds)

    fig, ax = plt.subplots(figsize=(9, 6))
    lines = {}
    for name, values in curves.items():
        (line,) = ax.plot(
            thresholds, values, label=name, color=METRIC_COLORS.get(name), linewidth=2
        )
        lines[name] = (line, values)

    vline = ax.axvline(initial_threshold, color="gray", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("Metrics vs. threshold (drag the dashed line)")
    ax.set_xlim(thresholds.min(), thresholds.max())
    ax.set_ylim(0.0, 1.05)
    legend = ax.legend(loc="lower left")

    def values_at(threshold: float) -> dict[str, float]:
        idx = int(np.argmin(np.abs(thresholds - threshold)))
        return {name: values[idx] for name, (_, values) in lines.items()}

    def refresh_legend(threshold: float):
        current = values_at(threshold)
        for text, name in zip(legend.get_texts(), lines.keys()):
            text.set_text(f"{name}: {current[name]:.3f}")
        ax.set_title(f"Metrics vs. threshold (threshold = {threshold:.3f})")

    refresh_legend(initial_threshold)

    dragging = {"active": False}

    def on_press(event):
        if event.inaxes != ax:
            return
        if abs(event.xdata - vline.get_xdata()[0]) < 0.03 * (thresholds.max() - thresholds.min()):
            dragging["active"] = True

    def on_release(event):
        dragging["active"] = False

    def on_motion(event):
        if not dragging["active"] or event.inaxes != ax or event.xdata is None:
            return
        threshold = float(np.clip(event.xdata, thresholds.min(), thresholds.max()))
        vline.set_xdata([threshold, threshold])
        refresh_legend(threshold)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)

    return fig, ax
