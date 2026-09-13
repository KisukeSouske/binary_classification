"""Binary classification solver, decoupled from data loading and plotting.

Mirrors the structure of the linear regression library (`kai/regression.py`)
so the two can be merged later: the model configuration is composed out of a
`_Family` - the (loss, link) pair plus everything that depends on it - and
regularization is a second, orthogonal registry of penalties.

Logistic regression is just another GLM family here: log-loss with a sigmoid
inverse link, and an intercept initialized at the log-odds of the base rate.
"""
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from binary_classifier import core


class TrainingCancelled(Exception):
    """Raised by `ClassifierFit.fit` when `cancel_event` is set mid-run."""


@dataclass(frozen=True)
class _Family:
    inverse_link: Callable[[np.ndarray], np.ndarray]
    loss: Callable[[np.ndarray, np.ndarray], float]
    gradient: Callable[..., tuple[np.ndarray, float]]
    init_bias: Callable[[np.ndarray], float]
    # Optional check for "the linear predictor has left the range this family
    # can represent". A clipped inverse link keeps the loss finite no matter
    # how far the parameters run away, which silently defeats the usual
    # `not isfinite(loss)` divergence guard.
    saturated: Callable[[np.ndarray], bool] | None = None


_FAMILIES = {
    ("log_loss", "sigmoid"): _Family(
        inverse_link=core.sigmoid,
        loss=core.log_loss,
        gradient=core.log_loss_derivation,
        # log-odds of the base rate: the best constant prediction, so the
        # first step starts from the intercept-only model rather than p=0.5.
        init_bias=lambda y: float(core.logit(np.mean(y))),
        # Past the clip, sigmoid(eta) no longer depends on the parameters and
        # log_loss clips too, so both the reported loss and the gradient stop
        # describing the model being fitted.
        saturated=lambda eta: bool(np.any(np.abs(eta) >= core.SIGMOID_EXP_CLIP)),
    ),
}


@dataclass(frozen=True)
class _Penalty:
    """A shrinkage penalty and its derivative w.r.t. the weights.

    Orthogonal to `_Family`: any family can be fitted with any penalty, so the
    two are composed at fit time rather than enumerated together.
    """

    value: Callable[[np.ndarray], float]
    gradient: Callable[[np.ndarray], np.ndarray]


_PENALTIES = {
    "ridge": _Penalty(value=core.ridge_penalty, gradient=core.ridge_penalty_derivation),
    "lasso": _Penalty(value=core.lasso_penalty, gradient=core.lasso_penalty_derivation),
}


def _as_design_inputs(X, y) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to float arrays, promote 1-D X to a single column, check shapes."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D, got {X.ndim} dimensions.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got {y.ndim} dimensions.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples "
            f"(got {X.shape[0]} and {y.shape[0]})."
        )
    return X, y


def gradient_norm(y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray,
                  gradient=None) -> float:
    """Euclidean norm of the full gradient (weights and bias together).

    `gradient` selects which objective's gradient to measure, defaulting to the
    unpenalized log loss. The convergence test has to measure the objective
    actually being optimised: with a penalty in play the data-term gradient is
    a different function with different zeros, so stopping on it would stop at
    the wrong place.
    """
    if gradient is None:
        gradient = core.log_loss_derivation
    weight_slope, bias_slope = gradient(y_true, y_pred, X)
    return float(np.sqrt(np.sum(weight_slope ** 2) + bias_slope ** 2))


@dataclass
class ClassifierFit:
    """A binary classifier: created empty, trained with `fit`, and keeping the
    learned parameters as its own internal state.

    `loss_function`/`loss_function_link` select the family; `penalty`/`delta`
    add a shrinkage term (`None`/0.0 = plain maximum likelihood).

    >>> model = ClassifierFit(penalty="ridge", delta=0.01)
    >>> model.fit(X_train, y_train, learning_rate=0.1)
    >>> probabilities = model.predict(X_test)
    """

    weights: np.ndarray | None = None
    bias: float | None = None
    loss_history: tuple[float, ...] | None = None
    loss_function: str = "log_loss"
    loss_function_link: str = "sigmoid"
    penalty: str | None = None
    delta: float = 0.0

    def __post_init__(self):
        if (self.loss_function, self.loss_function_link) not in _FAMILIES:
            raise ValueError(
                f"Unsupported combination: loss_function={self.loss_function!r}, "
                f"loss_function_link={self.loss_function_link!r}. "
                f"Available: {sorted(_FAMILIES)}"
            )
        if self.penalty is not None and self.penalty not in _PENALTIES:
            raise ValueError(
                f"Unknown penalty {self.penalty!r}; "
                f"available: {sorted(_PENALTIES)} or None."
            )
        if self.delta < 0.0:
            raise ValueError(f"delta must be non-negative (got {self.delta}).")

    @property
    def family(self) -> _Family:
        return _FAMILIES[(self.loss_function, self.loss_function_link)]

    @property
    def is_fitted(self) -> bool:
        return self.weights is not None and self.bias is not None

    # ------------------------------------------------------------------ #
    # Objective: family loss + shrinkage penalty
    # ------------------------------------------------------------------ #
    def penalty_value(self) -> float:
        """The shrinkage term delta * P(weights), or 0.0 when unregularized.

        Only the weights enter it. The intercept is deliberately excluded
        (ISLP p.241): shrinking it would shrink the base rate of the response
        rather than the association of any predictor with it.
        """
        if self.penalty is None or self.delta == 0.0:
            return 0.0
        return float(self.delta * _PENALTIES[self.penalty].value(self.weights))

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """The penalized objective being minimized: family loss + penalty.

        ISLP eq. (6.5) for ridge and (6.7) for the lasso, with the family's
        loss standing in for the RSS.
        """
        return float(self.family.loss(y_true, y_pred) + self.penalty_value())

    def loss_gradient(
        self, y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Gradient of `loss` w.r.t. the weights and the bias.

        This is what actually regularizes the fit: adding the penalty to the
        reported loss alone would change the number in `loss_history` without
        moving a single weight, since the updates descend this gradient.
        """
        weight_slope, bias_slope = self.family.gradient(y_true, y_pred, X)
        if self.penalty is not None and self.delta != 0.0:
            penalty_slope = _PENALTIES[self.penalty].gradient(self.weights)
            weight_slope = weight_slope + self.delta * penalty_slope
        return weight_slope, bias_slope  # bias_slope untouched: see penalty_value

    def gradient_norm(self, y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray) -> float:
        """Norm of the penalized objective's gradient, for the stopping rule."""
        return gradient_norm(y_true, y_pred, X, self.loss_gradient)

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def linear_predictor(self, X: np.ndarray) -> np.ndarray:
        """The linear part, eta = bias + X @ weights, BEFORE the inverse link.

        This is the scale the coefficients are additive on (log-odds / logit
        scale) - not a probability.
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet; call fit(...) first.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1 and self.weights.shape[0] == 1:
            X = X.reshape(-1, 1)
        return self.bias + X @ self.weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted probabilities of the positive class, shape [n_samples].

        The family's inverse link is applied here, so this is mu and not eta.
        """
        return self.family.inverse_link(self.linear_predictor(X))

    def predict_class(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Hard 0.0/1.0 labels, from `predict` cut at `threshold`."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1] (got {threshold}).")
        return (self.predict(X) >= threshold).astype(float)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X,
        y,
        learning_rate: float,
        batch_size: int = 100,
        epochs: int = 10_000,
        tolerance: float = 1e-4,
        random_state: int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> "ClassifierFit":
        """Fit by mini-batch gradient descent, filling this instance's state.

        Parameters:
        X (array-like): Features, shape (n_samples,) or (n_samples, n_features).
            Already in whatever space you want to train in - this method does no
            scaling of its own. Standardize first when using a penalty: ridge and
            the lasso are not scale-equivariant (ISLP eq. 6.6).
        y (array-like): Binary labels (0.0/1.0), shape (n_samples,).
        learning_rate (float): Step size for each mini-batch update.
        batch_size (int): Samples per gradient step. Values >= n_samples reduce
            this to plain full-batch gradient descent.
        epochs (int): Maximum passes over the data; a safety ceiling, not a target.
        tolerance (float): Relative convergence threshold. Stops when
            ||grad|| <= tolerance * ||grad_initial||, which makes the criterion
            independent of the scale of the features.
        random_state (int | None): Seed for the per-epoch shuffle. Pass an int
            for reproducible runs.
        cancel_event (threading.Event | None): Checked once per epoch; when set,
            training stops early by raising `TrainingCancelled`.

        Returns:
        ClassifierFit: this same object, now carrying weights, bias and the
            per-epoch loss history.

        Raises:
        ValueError: If y is single-class, or if training diverged.
        TrainingCancelled: If `cancel_event` was set before training finished.
        """
        X, y = _as_design_inputs(X, y)
        n_samples, n_features = X.shape
        family = self.family

        base_rate = float(np.mean(y))
        if not 0.0 < base_rate < 1.0:
            raise ValueError(
                "y must contain both classes to initialize the bias; got a "
                f"single class (base rate={base_rate})."
            )

        self.weights = np.zeros(n_features)
        self.bias = float(family.init_bias(y))
        loss_history: list[float] = []
        self.loss_history = ()
        rng = np.random.default_rng(random_state)

        # Reference magnitude for the convergence test, measured at the STARTING
        # parameters (before any update). Taking it after the first epoch would be
        # self-defeating: a run that converges immediately would compare its
        # already-tiny gradient against itself and never satisfy the threshold.
        initial_gradient_norm = self.gradient_norm(y, self.predict(X), X) or 1.0

        for epoch in range(epochs):
            if cancel_event is not None and cancel_event.is_set():
                raise TrainingCancelled("Training was stopped by the user.")
            indices = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch = indices[start:start + batch_size]
                x_batch, y_batch = X[batch], y[batch]
                mu_batch = family.inverse_link(self.bias + x_batch @ self.weights)
                weight_slope, bias_slope = self.loss_gradient(y_batch, mu_batch, x_batch)
                self.weights -= learning_rate * weight_slope
                self.bias -= learning_rate * bias_slope

            eta = self.bias + X @ self.weights
            y_pred = family.inverse_link(eta)
            epoch_loss = self.loss(y, y_pred)
            if not np.isfinite(epoch_loss):
                raise ValueError(
                    f"Training diverged at epoch {epoch}: loss became {epoch_loss}. "
                    f"Try a smaller learning_rate (current={learning_rate}) or "
                    f"standardize the features first."
                )
            if family.saturated is not None and family.saturated(eta):
                raise ValueError(
                    f"Training diverged at epoch {epoch}: the linear predictor "
                    f"saturated the link's safe range, so the loss stopped "
                    f"tracking the parameters. Try a smaller learning_rate "
                    f"(current={learning_rate}) or standardize the features first."
                )
            loss_history.append(epoch_loss)
            self.loss_history = tuple(loss_history)

            # Converged when the full-dataset gradient has shrunk to a small
            # FRACTION of where it started. A flat loss delta is not enough on
            # its own: the loss can stall while the parameters are still far
            # from the optimum.
            if self.gradient_norm(y, y_pred, X) <= tolerance * initial_gradient_norm:
                break

        return self
