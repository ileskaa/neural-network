"""Loss function used to optimizize model parameters"""

import numpy as np
from nn_utils import verify_one_hot_encoding

def cross_entropy(y_true, y_pred):
    """Cross-entropy loss function.

    This loss function gives a score based on how different
    two probability distributions are.
    A higher cross-entropy means more difference between the distrbutions.
    That's why softmax is great when paired with cross-entropy,
    since it transforms a vector of raw values into a vector of probabilities.
    By minimizing the cross entropy we thus minimize the difference
    between predicted values and true values.

    True labels must be one-hot encoded before computing cross-entropy.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred should have the same shape")

    verify_one_hot_encoding(y_true)

    epsilon = 1e-15
    # We want to avoid log(0),
    # which would equal -infinity.
    lower_bound = epsilon
    # We also want to avoid log(1), since that would yield 0.
    # This would cause the gradient to stay stuck
    # and stop updating the parameters.
    upper_bound = 1 - 1e-15
    # We clip the values so that they are never quite 0 and never quite 1
    y_pred = np.clip(y_pred, lower_bound, upper_bound)

    if y_true.ndim > 1:
        # For batches, we mean across samples to get a single value
        return -np.mean(
            np.sum(y_true * np.log(y_pred),
            axis=1)
        )
    return -np.sum(y_true * np.log(y_pred))

def cross_entropy_gradient(y_pred, y_true):
    """Gradient of loss function w.r.t. pre-softmax values.

    Due to softmax, the gradient for a single input simplifies to
    ∂L/∂z = y_pred - y_true
    where y_pred is the softmax output and y_true refers
    to the one-hot encoded correct digit.

    When working with batches we divide by the number of samples
    to get averaged values. This matches the behavior of
    the loss function, which calculates the mean across all batch samples.
    """
    verify_one_hot_encoding(y_true)

    # Number of samples
    n = 1 if y_true.ndim == 1 else y_true.shape[0]

    return (y_pred - y_true) / n
