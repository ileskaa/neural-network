"""Implementation of the multilayer perceptron"""

import time
import json
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
        biases=None,
        x_train=None,
        y_train=None
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
        self.x_train = x_train
        self.y_train = y_train
        # The following lists will be used during forward pass and backpropagation
        self.z_vectors = []
        self.activations = []

    def forward(self, network_input):
        """The forward pass of the neural network.
        For each digit, will return an array of size 10.
        Hence the output will have shape (num_digits, 10)
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

        Explanation of some variables:
        - dL_dz: loss function gradient with respect to pre-activation vector (AKA z-vector)
        - dz_dW: z-vector gradient w.r.t. weights
        - dz_db: z-vector gradient w.r.t. biases
        - dL_db: loss function gradient w.r.t. biases
        - dz_da: z-vector gradient w.r.t. previous layer's activations,
          i.e., activation function outputs
        - dL_da: loss function gradient w.r.t. activations
        - da_dz: activation vector gradient w.r.t. z-vector
        """
        y_hot_encoded = one_hot_encode(y_true)
        y_pred = self.activations[-1]
        # Initialize lists to save the gradients
        W_gradients = []
        b_gradients = []

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
            W_gradients.append(dL_dW)
            b_gradients.append(dL_db)

            # Compute loss gradient w.r.t. the pre-activation vector, for preceding layer.
            # Except if that preceding layer is the input layer
            if layer > 0:
                # Can overflow if the learning rate is too high
                dz_da = self.weights[layer]
                dL_da = np.matmul(dL_dz, dz_da.T)  # (batch_size, prev_layer_size)
                z = self.z_vectors[layer-1]
                da_dz = relu_derivative(z)
                dL_dz = dL_da * da_dz

        # Returning the gradients allows us to use them in the Adam optimizer
        return (W_gradients, b_gradients)

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

    def shuffle_data(self, x, y):
        """Introduce stochasticity by shuffling your data"""
        num_samples = x.shape[0]
        permutated_indexes = np.random.permutation(num_samples)
        x_shuffled = x[permutated_indexes]
        y_shuffled = y[permutated_indexes]
        return (x_shuffled, y_shuffled)

    def adam(self, x, y, f, theta, alpha=0.001, beta1=0.9, beta2=0.999, batch_size=64):
        """Implementaion of the adaptive moment estimation (Adam) optimization algorithm.

        Explanation of parameters:
        - alpha: stepsize
        - beta 1 and beta 2: exponential decay rates for the moment estimates.
          These are defined within a range of [0, 1)
        - f: stochastic objective function which takes a parameter vector theta as argument
        - theta: initial parameter vector

        Explanation of some terms:
        - first moment: expected value, or mean, of a distribution
        - second raw moment: mean of the squared values of a random variable.
          Measures how spread out a distribution is
        """
        # First moment vector
        m = 0
        # Second moment vector
        v = 0

        # initial timestep
        t = 0

        # Once we get below this, we can consider the parameters have converged
        loss_goal = 0.05
        # Initialize loss with a high value
        loss = 1

        epsilon = 10e-8

        num_layers = len(self.weights)
        num_samples = x.shape[0]
        while loss > loss_goal:
            t += 1
            x_shuffled, y_shuffled = self.shuffle_data(x, y)
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                x_batch = x_shuffled[start:end]
                y_batch = x_shuffled[start:end]

                y_pred = self.forward(x_batch)
                gradients_w, gradients_b = self.backprop(y)

                # TODO Implement loop to handle both weight and bias gradients
                # Perform for each parameter, meaning for each weight matrix and vias vector
                for layer in range(num_layers):
                    gradient = gradients_w[layer]
                    gradient_b = gradients_b[layer]
                    # Update biased first moment estimates
                    m = beta1 * m + (1-beta1) * gradient
                    # Update biased second raw moment estimates
                    v = beta2 * v + (1-beta2) * gradient**2
                    # Compute bias corrected 1st moment estimates
                    m = m / (1 - beta1**t)
                    # Compute bias corrected 2nd raw moment estimates
                    v = v / (1 - beta2**t)
                    # Update parameters
                    theta = theta - alpha*m / (np.sqrt(v)+epsilon)

            # Get gradients with respect to the stochastic objective at timestep t
            # gt =

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
            x_shuffled, y_shuffled = self.shuffle_data(x_train, y_train)

            epoch_loss = 0

            for start in range(0, num_samples, batch_size):
                # Avoid exceeding the max range
                end = start + batch_size
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

    def measure_accuracy(self, x_test, y_test, save_result=True):
        """Use test data to check the accuracy of the model.

        Saves the measure accuracy in a JSON file, unless specified otherwise.
        """
        n = len(x_test)
        predictions = self.predict(x_test)
        comparison = (predictions == y_test).astype(int)
        accuracy = sum(comparison)/n
        percents = accuracy * 100
        print(f"Accuracy on test data: {percents:.2f}%")
        to_be_saved = {"accuracy": percents}
        if save_result:
            with open("src/web/parameters/accuracy.json", "w", encoding="utf-8") as file:
                json.dump(to_be_saved, file)

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
