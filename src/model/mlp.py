"""Implementation of the multilayer perceptron"""

import time
import numpy as np
from .activations import relu, relu_derivative, softmax
from .loss import cross_entropy, cross_entropy_gradient
from .nn_utils import he_initalization, one_hot_encode


class MultiLayerPerceptron:
    """Neural network with the goal of classifying handwritten digits."""

    def __init__(
        self,
        layer_sizes=None,
        rng=np.random.default_rng(),
        weights=None,
        biases=None
    ) -> None:
        """Initialize weights and biases.

        The model can be initialized with pre-defined weights and biases.
        If not provided, a new set of weights and biases will be provided.
        """

        if layer_sizes is None:
            layer_sizes = [784, 256, 128, 10]

        self.weights = weights or [
            he_initalization(
                input_size,
                output_size,
                rng
            ) for input_size, output_size in zip(
                layer_sizes[:-1],
                layer_sizes[1:]
            )
        ]
        self.biases = biases or [np.zeros(layer_size) for layer_size in layer_sizes[1:]]
        # The following lists will be used during forward pass and backpropagation
        self.z_vectors = []
        self.activations = []

    def forward(self, network_input):
        """The forward pass of the neural network.
        Will return a numpy array of size 10.
        """
        n = len(self.weights)
        # Reset activations and z vectors at each forward pass
        self.activations = [network_input]  # store input and activations
        self.z_vectors = []  # store pre-activation values

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

        num_layers = len(self.weights)  # excluding input layer
        for layer in reversed(range(num_layers)):
            dz_dW = self.activations[layer]
            # Can overflow if the learning rate is too high
            dL_dW = np.matmul(dz_dW.T, dL_dz)
            dz_db = 1
            dL_db = dz_db * np.sum(dL_dz, axis=0)
            # Update weights and biases. Remember learning rate
            self.weights[layer] -= dL_dW * learning_rate
            self.biases[layer] -= dL_db * learning_rate

            # Compute gradient for preceding layer, if it's not the input layer
            if layer > 0:
                # Can overflow if the learning rate is too high
                dz_da = self.weights[layer]
                dL_da = np.matmul(dL_dz, dz_da.T)  # (batch_size, prev_layer_size)
                z = self.z_vectors[layer-1]
                da_dz = relu_derivative(z)
                dL_dz = dL_da * da_dz

    def predict(self, x):
        """Takes in an image or a vector of images in pixel values
        and predicts which digit it is based on model parameters.

        This network's forward method returns an array
        of discrete probabilities for each digit.
        Using numpy's argmax method, we'll get the index of the highest probability,
        since that will correspond to our predicted digit.
        """
        y_pred = self.forward(x)
        if y_pred.ndim > 1:
            # argmax returns the indices of the maximum values
            return np.argmax(y_pred, axis=1)
        return np.argmax(y_pred)

    def train(
        self,
        x_train,
        y_train,
        epochs=15,
        batch_size=128,
        learning_rate=0.01
    ):
        """Train the model using stochastic gradient descent (SGD).

        Splitting the training data into batches will allow us to introduce
        stochasticity (AKA randomness) into the training process,
        which should speed it up.

        Image data should be normalized since it helps
        the network converge faster.
        """
        start_time = time.time()

        if np.mean(x_train) > 1:
            raise ValueError('Pixel values should be normalized')

        num_samples = x_train.shape[0]

        for epoch in range(epochs):
            permutated_indexes = np.random.permutation(num_samples)
            x_shuffled = x_train[permutated_indexes]
            y_shuffled = y_train[permutated_indexes]

            epoch_loss = 0

            for start in range(0, num_samples, batch_size):
                # Avoid exceeding the max range
                end = min(start+batch_size, num_samples)
                x_batch = x_shuffled[start:end]
                y_pred = self.forward(x_batch)
                y_batch_true = y_shuffled[start:end]
                self.backprop(y_batch_true, learning_rate)

                y_true_enc = one_hot_encode(y_batch_true)
                ce_score = cross_entropy(y_true_enc, y_pred)
                size = end - start
                # Scale cross-entropy score based on size
                epoch_loss += ce_score * size

            # Normalize loss based on total samples
            epoch_loss /= num_samples
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.3f}")
        end_time = time.time()
        diff = end_time - start_time
        print(f"Elapsed time: {diff:.2f} seconds")

    def measure_accuracy(self, x_test, y_test):
        """Use test data to check the accuracy of the model"""
        n = len(x_test)
        predictions = self.predict(x_test)
        comparison = (predictions == y_test).astype(int)
        accuracy = sum(comparison)/n
        percents = accuracy * 100
        print(f"Accuracy on test data: {percents:.2f}%")

    def save_parameters(self, destination_dir='src/web/parameters/'):
        """Save model parameters into a file.
        This will allow the Flask application to access those parameters once deployed.
        """
        n = len(self.weights)
        for i in range(n):
            filename = 'layer' + str(i+1)
            np.savez(
                destination_dir+filename,
                weights=self.weights[i],
                biases=self.biases[i]
            )
