"""Utility functions used by the neural network"""

import numpy as np

# Enable dependency injection for the RNG
def he_initalization(fan_in, fan_out, rng=np.random.default_rng()):
    """He initialization is useful when working with ReLU activations.
    It maintains the variance of the activations,
    which helps prevent the gradient from becoming too small (vanishing)
    or too large (exploding) during backpropagation.

    With ReLU, only about half the neurons are activated during the forward pass.
    He initialization accounts for this by scaling the variance up.
    """
    std = np.sqrt(2 / fan_in)
    return rng.normal(0, std, size=(fan_in, fan_out))

def one_hot_encode(y):
    """One hot encodes digits"""
    class_count = 10
    return np.eye(class_count)[y]

def verify_one_hot_encoding(y):
    """Verify that the given input is one-hot encoded"""
    if not hasattr(y, 'ndim'):
        raise ValueError("Not a one-hot encoded value")
    if y.ndim > 1:
        sums = np.sum(y, axis=1)
        if not np.allclose(sums, 1):
            raise ValueError("All values should be one-hot encoded")
    else:
        single_sum = np.sum(y)
        if not np.isclose(single_sum, 1):
            raise ValueError("Not a one-hot encoded value")

def normalize_image_data(x):
    """Normalize pixel values to scale them down to [0, 1].
    This prevents oscillation and overshooting during gradient descent.
    Without normalization, finding an appropriate learning rate
    would be much harder.
    """
    return x / 255

def load_layer(source):
    """Load layer parameters from an .npz file"""
    data = np.load(source)
    return (data['weights'], data['biases'])

def load_parameters(source_dir='src/web/parameters/', layers=3):
    """Load model parameters from .npz files.
    Enables the web app to access the parameters optimized during training.
    """
    weights = []
    biases = []
    for layer in range(1, layers+1):
        filename = source_dir + f'layer{layer}.npz'
        weights_arr, biases_arr = load_layer(filename)
        weights.append(weights_arr)
        biases.append(biases_arr)
    return (weights, biases)
