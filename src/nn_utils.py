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
