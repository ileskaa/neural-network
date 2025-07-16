"""Tests for the MultiLayerPerceptron class"""

import unittest
import numpy as np
import loss
from mlp import MultiLayerPerceptron
from nn_utils import one_hot_encode, normalize_image_data


class TestMLP(unittest.TestCase):
    """Unit tests for the neural network"""
    def setUp(self) -> None:
        """Create a model to be used during testing.

        The input layer should have size 784 since the digits
        were initially drawn on a 28x28 pixel grid.
        This means a total of 784 pixels.

        The output layer should have size 10 since this is
        a classification problem with 10 possible outcomes.

        The size and amount of hidden layers is arbitrary.
        """
        layer_sizes = [784, 512, 256, 10]
        self.rng = np.random.default_rng(84)
        self.model = MultiLayerPerceptron(layer_sizes, self.rng)

    def test_initialization(self):
        """Testing weight and bias initialization.
        
        The model defined in the setUp function has 2 hidden layers
        and 1 output layer.
        """
        weights = self.model.weights
        biases = self.model.biases
        self.assertTrue(len(weights) == len(biases) == 3)
        # Test weight shapes
        self.assertEqual(weights[0].shape, (784, 512))
        self.assertEqual(weights[-1].shape, (256, 10))
        # Test bias shapes
        self.assertEqual(biases[0].shape, (512,))
        self.assertEqual(biases[-1].shape, (10,))
        # Initally all biases should be set to 0
        bias_sum = sum(np.sum(bias_vector) for bias_vector in biases)
        self.assertEqual(bias_sum, 0)

    def test_forward(self):
        """Testing the forward pass method of the network.
        
        We'll generate 784 integers to simulate a grayscale digit.
        Then forwards them through the network.
        At the end, we should get a discrete probability distribution
        of 10 values.
        """
        x = np.random.randint(0, 255, 784)
        output = self.model.forward(x)
        self.assertEqual(output.shape, (10,))
        summed_probabilities = np.sum(output)
        self.assertAlmostEqual(summed_probabilities, 1)

        # Activations and z vectors should be stored in the model
        self.assertEqual(len(self.model.z_vectors), 3)
        # The activation list should contain 4 values
        # since it also stores the network input
        self.assertEqual(len(self.model.activations), 4)

    def test_backprop(self):
        """Backpropagation tests.

        Weights and biases should be updated after backpropagation.

        We'll be using a batch of 100 samples for these tests.
        """
        y_true = self.rng.integers(0, 10, size=100)
        with self.assertRaises(IndexError):
            # Should raise an error because we haven't done any forward pass yet
            self.model.backprop(y_true)
        x = self.rng.integers(0, 255, (100, 784))
        x = normalize_image_data(x)
        output = self.model.forward(x)

        y_true_encoded = one_hot_encode(y_true)
        cross_entropy = loss.cross_entropy(y_true_encoded, output)
        inital_weights = [w.copy() for w in self.model.weights]
        initial_biases = self.model.biases
        self.model.backprop(y_true)

        weights_after = self.model.weights
        biases_after = self.model.biases
        n = len(inital_weights)
        for layer in range(n):
            np.not_equal(inital_weights[layer], weights_after[layer])
            np.not_equal(initial_biases[layer], biases_after[layer])

        # The loss function score should decrease
        output_after_backprop = self.model.forward(x)
        cross_entropy_after_backprop = loss.cross_entropy(
            y_true_encoded,
            output_after_backprop
        )
        self.assertTrue(cross_entropy_after_backprop < cross_entropy)

    def test_predict(self):
        """Tests for the networks digit prediction method.

        The predict method should return a single integer or an array of integers,
        depending on the shape of the input.
        """
        single_image = np.random.randint(0, 255, 784)
        y_pred = self.model.predict(single_image)
        self.assertIsInstance(y_pred, np.int64)

        # Test array inputs
        image_vector = self.rng.integers(0, 255, (100, 784))
        y_pred = self.model.predict(image_vector)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertTrue(np.issubdtype(y_pred.dtype, np.int64))
