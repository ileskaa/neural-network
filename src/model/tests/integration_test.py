"""Integration tests for the neural network"""

import unittest
import numpy as np
from model.mlp import MultiLayerPerceptron


class TestModel(unittest.TestCase):
    """Integration tests for the neural network"""

    def setUp(self) -> None:
        """Create a model to be used during testing"""
        layer_sizes = [784, 256, 128, 10]
        self.rng = np.random.default_rng(63)
        self.model = MultiLayerPerceptron(layer_sizes, self.rng)

    def test_network_overfits_small_dataset(self):
        """On a small dataset, the network should be able to achieve
        a classification accuracy of 100%"""
        l_bound = 0
        u_bound_exclusive = 256
        sample_size = 20
        num_pixels = 784
        x_sample = self.rng.integers(
            l_bound,
            u_bound_exclusive,
            size=(sample_size, num_pixels)
        )
        # Normalize pixel values
        x_sample = x_sample / 255
        y_sample = self.rng.integers(0, 10, sample_size)
        self.model.train(
            x_sample, y_sample, epochs=15, batch_size=2,
            learning_rate=0.02
        )
        accuracy = self.model.measure_accuracy(x_sample, y_sample)
        # The model should fit the data perfectly
        self.assertEqual(accuracy, 100)
