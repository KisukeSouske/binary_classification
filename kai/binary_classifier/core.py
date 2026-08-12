import numpy as np

SIGMOID_EXP_CLIP = 30  # Clip values to avoid overflow in exp; also stays below the
                        # ~36.7 threshold where exp(-x) underflows to 0 in float64,
                        # which would saturate sigmoid to exactly 1.0

def sigmoid(x):
    """
    Sigmoid activation function.
    Returns a value between 0 and 1.
    """
    x = np.clip(x, -SIGMOID_EXP_CLIP, SIGMOID_EXP_CLIP)
    return 1 / (1 + np.exp(-x))

def logit(p, eps=1e-12):
    """
    Inverse of the sigmoid function (log-odds).
    Maps a probability in (0, 1) to the real line.
    """
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def log_loss_derivation(y_true, y_pred, x_true) -> tuple[np.ndarray, float]:
    """
    Computes the gradient of the log loss (binary cross-entropy) with respect
    to the weights and the bias.
    y_true: true labels (0 or 1)
    y_pred: predicted probabilities (between 0 and 1)
    x_true: input features (numpy array of shape [n_samples, n_features])
    Returns: the (weight_derivation, bias_derivation) gradients
    """
    n_samples = x_true.shape[0]
    error = y_pred - y_true
    dw = (1 / n_samples) * np.dot(x_true.T, error)
    db = (1 / n_samples) * np.sum(error)
    return dw, db

def log_loss(y_true, y_pred, eps=1e-12) -> float:
    """
    Computes the log loss (binary cross-entropy loss) for binary classification.
    y_true: true labels (0 or 1)
    y_pred: predicted probabilities (between 0 and 1)
    """
    # Clip predictions to avoid log(0) which is undefined
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def ridge_penalty(weights):
    """
    Computes the ridge (L2) penalty for regularization: sum of squared weights
    (ISLP eq. 6.5, without the tuning parameter, which the caller multiplies in).
    weights: model weights (numpy array)
    Returns: ridge penalty value
    """
    return np.sum(weights ** 2)

def ridge_penalty_derivation(weights) -> np.ndarray:
    """
    Gradient of the ridge penalty with respect to the weights: d(sum w^2)/dw = 2w.
    weights: model weights (numpy array)
    Returns: gradient, same shape as weights
    """
    return 2.0 * np.asarray(weights, dtype=float)

def lasso_penalty(weights):
    """
    Computes the lasso (L1) penalty for regularization: sum of absolute weights
    (ISLP eq. 6.7, without the tuning parameter, which the caller multiplies in).
    weights: model weights (numpy array)
    Returns: lasso penalty value
    """
    return np.sum(np.abs(weights))

def lasso_penalty_derivation(weights) -> np.ndarray:
    """
    Subgradient of the lasso penalty with respect to the weights: sign(w),
    taking 0 at w=0 where the penalty is not differentiable.

    Being a subgradient rather than a gradient, plain descent on it shrinks
    coefficients but does not drive them to exactly zero; the variable
    selection ISLP describes (eq. 6.15) comes from soft-thresholding instead.
    weights: model weights (numpy array)
    Returns: subgradient, same shape as weights
    """
    return np.sign(np.asarray(weights, dtype=float))

