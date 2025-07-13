"""Implementation of the multilayer perceptron"""

import numpy as np
from activations import relu, softmax
from nn_utils import he_initalization


class MultiLayerPerceptron:
    """Neural network with the goal of classifying handwritten digits."""
    def __init__(self, layer_sizes) -> None:
        """Initialize weights and biases"""
        self.weights = [
            he_initalization(input_size, output_size) for input_size, output_size in zip(
                layer_sizes[:-1],
                layer_sizes[1:]
            )
        ]
        self.biases = [np.zeros(layer_size) for layer_size in layer_sizes[1:]]
        # The following lists will be used during forward pass and backpropagation
        self.z_vectors = []
        self.activations = []

    def forward(self, network_input):
        """The forward pass of the neural network.
        Will return a numpy array of size 10.
        """
        n = len(self.weights)
        # Reset activations and z vectors at each forward pass
        self.activations = [network_input] # store input and activations
        self.z_vectors = [] # store pre-activation values

        for layer in range(n):
            w = self.weights[layer]
            b = self.biases[layer]
            x = self.activations[-1]
            z = np.matmul(x, w) + b
            self.z_vectors.append(z)
            if layer == n-1:
                output = softmax(z)
            else:
                output = relu(z)
            self.activations.append(output)
        final_output = self.activations[-1]
        return final_output
