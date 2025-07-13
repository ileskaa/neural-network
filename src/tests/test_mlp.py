"""Tests for the MultiLayerPerceptron class"""

import unittest
import numpy as np
from mlp import MultiLayerPerceptron


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
        self.model = MultiLayerPerceptron(layer_sizes)

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

