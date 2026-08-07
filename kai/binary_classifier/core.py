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