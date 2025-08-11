"""Integration tests for the neural network"""

import copy
import unittest
import numpy as np
from model.mlp import MultiLayerPerceptron


class TestModel(unittest.TestCase):
    """Integration tests for the neural network"""

    def setUp(self) -> None:
        """Create a model to be used during testing"""
        layer_sizes = [784, 256, 128, 10]
        rng = np.random.default_rng(seed=63)
        # Intialize 2 models to test different training methods
        self.model = MultiLayerPerceptron(layer_sizes, rng)
        self.model2 = MultiLayerPerceptron([784, 384, 128, 10], rng)
        l_bound = 0
        u_bound_exclusive = 256
        sample_size = 16
        num_pixels = 784
        x_sample = rng.integers(
            l_bound, u_bound_exclusive, size=(sample_size, num_pixels)
        )
        # Normalize pixel values
        self.x_sample = x_sample / 255
        self.y_sample = rng.integers(0, 10, sample_size)

    def test_network_overfits_small_dataset(self):
        """On a small dataset, the network should be able to achieve
        a classification accuracy of 100%"""
        self.model.train(
            self.x_sample, self.y_sample, epochs=16, batch_size=2, learning_rate=0.02
        )
        accuracy = self.model.measure_accuracy(self.x_sample, self.y_sample)
        # The model should fit the data perfectly
        self.assertEqual(accuracy, 100)

        # Run tests for the Adam optimizer
        self.model2.adam(
            self.x_sample,
            self.y_sample,
            epochs=18,
            alpha=0.001,
            beta2=0.97,
            batch_size=16,
        )
        accuracy = self.model2.measure_accuracy(self.x_sample, self.y_sample)
        self.assertEqual(accuracy, 100)

    def test_all_layers_change(self):
        """Ensure all layers get updated during a training cycle"""
        # Create deep copy of parameter lists
        initial_weights = copy.deepcopy(self.model.weights)
        initial_biases = copy.deepcopy(self.model.biases)
        self.model.train(self.x_sample, self.y_sample, epochs=1, batch_size=2)
        weight_after = self.model.weights
        biases_after = self.model.biases
        num_layers = len(initial_weights)
        for layer in range(num_layers):
            are_weights_equal = np.array_equal(
                initial_weights[layer], weight_after[layer]
            )
            are_biases_equal = np.array_equal(
                initial_biases[layer], biases_after[layer]
            )
            self.assertFalse(are_weights_equal)
            self.assertFalse(are_biases_equal)

    def test_single_vs_batch(self):
        """Ensure the output is the same regardless of whether the input was
        given as part of a minibatch or as a single sample.
        """
        rng = np.random.default_rng(21)
        model = MultiLayerPerceptron(rng=rng)
        y_pred = model.forward(self.x_sample)
        y_pred2 = []
        for x in self.x_sample:
            pred = model.forward(x)
            y_pred2.append(pred)
        y_pred2 = np.array(y_pred2)
        self.assertEqual(y_pred.shape, y_pred2.shape)
        np.testing.assert_array_almost_equal(y_pred, y_pred2)
