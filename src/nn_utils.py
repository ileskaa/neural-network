"""Utility functions used by the neural network"""

import numpy as np

def he_initalization(fan_in, fan_out):
    """He initialization is useful when working with ReLU activations.
    It maintains the variance of the activations,
    which helps prevent the gradient from becoming too small (vanishing)
    or too large (exploding) during backpropagation.

    With ReLU, only about half the neurons are activated during the forward pass.
    He initialization accounts for this by scaling the variance up.
    """
    std = np.sqrt(2 / fan_in)
    return np.random.normal(0, std, size=(fan_in, fan_out))

def one_hot_encode(y):
    """One hot encodes digits"""
    class_count = 10
    return np.eye(class_count)[y]

def cross_entropy(y_true, y_pred):
    """Cross-entropy loss function.

    This loss function gives a score based on how different
    two probability distributions are.
    A higher cross-entropy means more differene between the distrbutions.
    That's why softmax is great when paired with cross-entropy,
    since it transforms a vector of raw values into a vector of probabilities.
    By minimizing the cross entropy we thus minimize the difference
    between predicted values and true values.

    True labels must be one-hot encoded before computing cross-entropy.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred should have the same shape")

    if y_true.ndim > 1:
        sums = np.sum(y_true, axis=1)
        if not np.allclose(sums, 1):
            raise ValueError("All values should be one-hot encoded")
    else:
        mysum = np.sum(y_true)
        if not np.isclose(mysum, 1):
            raise ValueError("Not a one-hot encoded value")

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
        return -np.mean(
            np.sum(y_true * np.log(y_pred), axis=1)
        )

    return -np.mean(
        np.sum(y_true * np.log(y_pred))
    )
