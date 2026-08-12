from dataclasses import dataclass
from binary_classifier import core
import numpy as np

# Registries the `loss_function` / `loss_function_link` fields dispatch on.
LINK_FUNCTIONS = {"sigmoid": core.sigmoid}
LOSS_FUNCTIONS = {"log_loss": core.log_loss}
LOSS_GRADIENTS = {"log_loss": core.binary_cross_entropy_gradient}


@dataclass
class ClassifierFit:
    """A binary classifier that is created empty, trained with `fit`, and
    keeps the learned parameters as its own internal state.

    >>> model = ClassifierFit()
    >>> model.fit(X_train, y_train, learning_rate=0.1)
    >>> probs = model.sigmoid_predictor(X_test)
    """

    weights: np.ndarray | None = None
    bias: float | None = None
    loss_history: tuple[float, ...] | None = None
    loss_function: str = "log_loss"
    loss_function_link: str = "sigmoid"

    def __post_init__(self):
        if self.loss_function not in LOSS_FUNCTIONS:
            raise ValueError(
                f"Unknown loss_function {self.loss_function!r}; "
                f"available: {sorted(LOSS_FUNCTIONS)}."
            )
        if self.loss_function_link not in LINK_FUNCTIONS:
            raise ValueError(
                f"Unknown loss_function_link {self.loss_function_link!r}; "
                f"available: {sorted(LINK_FUNCTIONS)}."
            )

    @property
    def is_fitted(self) -> bool:
        return self.weights is not None and self.bias is not None

    def link(self, eta: np.ndarray) -> np.ndarray:
        """Apply the configured link (`loss_function_link`) to the linear predictor."""
        return LINK_FUNCTIONS[self.loss_function_link](eta)

    def loss(self, y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
        """Evaluate the configured loss (`loss_function`)."""
        return LOSS_FUNCTIONS[self.loss_function](y_true, y_pred_prob)

    def loss_gradient(
        self, y_true: np.ndarray, y_pred_prob: np.ndarray, X: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Gradient of the configured loss w.r.t. the weights and the bias."""
        return LOSS_GRADIENTS[self.loss_function](y_true, y_pred_prob, X)

    def gradient_norm(
        self, y_true: np.ndarray, y_pred_prob: np.ndarray, X: np.ndarray
    ) -> float:
        """Calcula a norma L2 do gradiente em classificação binária."""
        weight_slope, bias_slope = self.loss_gradient(y_true, y_pred_prob, X)
        return float(np.sqrt(np.sum(weight_slope**2) + bias_slope**2))

    def linear_predictor(self, X: np.ndarray) -> np.ndarray:
        """The linear part, eta = bias + X @ weights.

        This is the scale the coefficients are additive on (log-odds / logit
        scale). Apply the sigmoid to convert it to a probability.
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet; call fit(...) first.")
        if X.ndim == 1 and self.weights.shape[0] == 1:
            X = X.reshape(-1, 1)
        return self.bias + X @ self.weights

    def sigmoid_predictor(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using the link function.
        X: input features (numpy array of shape [n_samples, n_features])
        Returns: predicted probabilities (numpy array of shape [n_samples])
        """
        X = np.asarray(X, dtype=float)
        return self.link(self.linear_predictor(X))

    def fit(
        self,
        X,
        y,
        learning_rate: float,
        batch_size: int = 100,
        epochs: int = 10_000,
        tolerance: float = 1e-4,
        random_state: int | None = None,
    ) -> "ClassifierFit":
        """
        Fit this classifier using gradient descent, storing the learned
        weights, bias and loss history on the instance. Returns self.
        X: input features (numpy array of shape [n_samples, n_features])
        y: true labels (numpy array of shape [n_samples])
        learning_rate: step size for gradient descent
        epochs: number of iterations to run gradient descent
        random_state: seed for the batch-shuffling RNG; None uses non-deterministic entropy
        """
        n_samples, n_features = X.shape
        base_rate = np.mean(y)
        if base_rate <= 0.0 or base_rate >= 1.0:
            raise ValueError(
                "y must contain both classes to initialize the bias; got a "
                f"single class (base rate={base_rate})."
            )
        self.weights = np.zeros(n_features)
        # Initialize bias to the log-odds of the base rate for better convergence
        self.bias = core.logit(base_rate)
        self.loss_history = ()
        loss_history: list[float] = []
        rng = np.random.default_rng(random_state)

        initial_gradient_norm = self.gradient_norm(y, self.sigmoid_predictor(X), X)
        if initial_gradient_norm < tolerance:
            return self
        for epoch in range(epochs):
            indices = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                batch = indices[start:start + batch_size]
                X_batch = X[batch]
                y_batch = y[batch]
                mu_batch = self.sigmoid_predictor(X_batch)
                weight_slope, bias_slope = self.loss_gradient(y_batch, mu_batch, X_batch)
                self.weights -= learning_rate * weight_slope
                self.bias -= learning_rate * bias_slope

            y_pred = self.sigmoid_predictor(X)
            epoch_loss = self.loss(y, y_pred)
            if not np.isfinite(epoch_loss):
                raise ValueError(
                    f"Training diverged at epoch {epoch}: loss became {epoch_loss}. "
                    f"Try a smaller learning_rate (current={learning_rate}) or "
                    f"standardize the features first."
                )
            loss_history.append(epoch_loss)
            self.loss_history = tuple(loss_history)

            if self.gradient_norm(y, y_pred, X) < tolerance:
                break

        return self
