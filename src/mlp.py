"""Implementation of the multilayer perceptron"""

import numpy as np
from activations import relu, relu_derivative, softmax
from loss import cross_entropy_gradient
from nn_utils import he_initalization, one_hot_encode


class MultiLayerPerceptron:
    """Neural network with the goal of classifying handwritten digits."""
    def __init__(self, layer_sizes, rng=np.random.default_rng()) -> None:
        """Initialize weights and biases"""
        self.weights = [
            he_initalization(
                input_size,
                output_size,
                rng
            ) for input_size, output_size in zip(
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

    def backprop(self, y_true, learning_rate=0.01):
        """Backpropagation method.
        
        Updates all weights and biases in a way that minimizes the loss function.
        """
        y_hot_encoded = one_hot_encode(y_true)
        y_pred = self.activations[-1]
        # cross_entropy_gradient() returns a gradient normalized by batch size
        dL_dz = cross_entropy_gradient(y_pred, y_hot_encoded)

        num_layers = len(self.weights) # excluding input layer
        for layer in reversed(range(num_layers)):
            dz_dW = self.activations[layer]
            dL_dW = np.matmul(dz_dW.T, dL_dz)
            dz_db = 1
            dL_db = dz_db * np.sum(dL_dz, axis=0)
            # Update weights and biases. Remember learning rate
            self.weights[layer] -= dL_dW * learning_rate
            self.biases[layer] -= dL_db * learning_rate

            # Compute gradient for preceding layer, if it's not the input layer
            if layer > 0:
                dz_da = self.weights[layer]
                dL_da = np.matmul(dL_dz, dz_da.T) # (batch_size, prev_layer_size)
                z = self.z_vectors[layer-1]
                da_dz = relu_derivative(z)
                if dL_da.shape != da_dz.shape:
                    raise ValueError('dL/da and da/dz should have the same shape')
                dL_dz = dL_da * da_dz
