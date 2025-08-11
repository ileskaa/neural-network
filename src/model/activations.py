"""Activation functions used in the neural network"""

import numpy as np


def relu(x):
    """Rectified Linear Unit (ReLU).
    Common activation function for classification tasks.

    This function uses np.maximum instead of np.max
    since we want the element-wise maximum.
    """
    return np.maximum(x, 0)


def relu_derivative(x):
    """Derivative of the ReLU activation function.

    Returns 1 if x is greater than 0. Otherwise returns 0.
    """
    return np.where(x > 0, 1, 0)


def softmax(z):
    """Softmax activation function.

    This function will be used in the last layer of the network.
    Softmax turns a vector of raw values into a vector of probabilites,
    which makes it great when working with a cross-entropy loss function.

    Since softmax relies on the exponential function,
    it can cause numerical overflow if its input is too large.
    We therefore substract the maximum value of the vector from each element,
    thus ensuring that the largest input value will be 0.
    As a consequence, the largest exponent will be exp(0) = 1.
    And all other exponents will be bounded between 0 and 1.
    This works because softmax is not affected by constant shifts.

    This function handles both 1- and 2-dimensional arrays.
    """
    if z.ndim == 2:
        shifted_z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shifted_z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    shifted_z = z - np.max(z, keepdims=True)
    exp_z = np.exp(shifted_z)
    return exp_z / np.sum(exp_z, keepdims=True)
