from dataclasses import dataclass
from binary_classifier import core, metrics
import numpy as np

@dataclass(frozen=True)
class ClassifierFit:
    weights: np.ndarray
    bias: float
    loss_history: tuple[float, ...] | None = None
    loss_function: str = "log_loss"
    loss_function_link: str = "sigmoid"

    def linear_predictor(self, X: np.ndarray) -> np.ndarray:
        """The linear part, eta = bias + X @ weights.

        This is the scale the coefficients are additive on (log-odds / logit
        scale). Apply the sigmoid to convert it to a probability.
        """
        if X.ndim == 1 and self.weights.shape[0] == 1:
            X = X.reshape(-1, 1)
        return self.bias + X @ self.weights
    
    def sigmoid_predictor(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using the sigmoid function.
        X: input features (numpy array of shape [n_samples, n_features])
        Returns: predicted probabilities (numpy array of shape [n_samples])
        """
        X = np.asarray(X, dtype=float)
        X = self.linear_predictor(X)
        return core.sigmoid(X)

def fit_gradient_descent(X,
                         y,
                         learning_rate: float,
                         batch_size: int = 100,
                         epochs: int=10_000,
                         tolerance: float=1e-4,
                         random_state: int | None = None) -> ClassifierFit:
    """
    Fit a binary classifier using gradient descent.
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
    weights = np.zeros(n_features)
    bias = core.logit(base_rate)  # Initialize bias to the log-odds of the base rate for better convergence
    loss_history: list[float] = []
    rng = np.random.default_rng(random_state)

    initial_gradient_norm = gradient_norm(y, core.sigmoid(np.dot(X, weights) + bias), X)
    if initial_gradient_norm < tolerance:
        return ClassifierFit(weights=weights, bias=bias, loss_history=tuple(loss_history))
    for epoch in range(epochs):
        indices = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            batch = indices[start:start + batch_size]
            X_batch = X[batch]
            y_batch = y[batch]
            mu_batch = core.sigmoid(np.dot(X_batch, weights) + bias)
            weight_slope, bias_slope = core.binary_cross_entropy_gradient(y_batch, mu_batch, X_batch)
            weights -= learning_rate * weight_slope
            bias -= learning_rate * bias_slope

        eta = bias + X @ weights
        y_pred = core.sigmoid(eta)
        epoch_loss = metrics.log_loss(y, y_pred)
        if not np.isfinite(epoch_loss):
            raise ValueError(
                f"Training diverged at epoch {epoch}: loss became {epoch_loss}. "
                f"Try a smaller learning_rate (current={learning_rate}) or "
                f"standardize the features first."
            )
        loss_history.append(epoch_loss)

        if gradient_norm(y, y_pred, X) < tolerance:
            break

    return ClassifierFit(weights=weights, bias=bias, loss_history=tuple(loss_history))

def gradient_norm(
    y_true: np.ndarray, y_pred_prob: np.ndarray, X: np.ndarray
) -> float:
    """Calcula a norma L2 do gradiente em classificação binária."""
    weight_slope, bias_slope = core.binary_cross_entropy_gradient(
        y_true, y_pred_prob, X
    )
    return float(np.sqrt(np.sum(weight_slope**2) + bias_slope**2))
