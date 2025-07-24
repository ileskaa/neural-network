"""Tests for data loading functions"""

import tempfile
import unittest
import numpy as np
from model.mnist_loader import load_data
from model.mlp import MultiLayerPerceptron
from model.nn_utils import load_layer, load_parameters


class TestLoader(unittest.TestCase):
    """Unit tests for the MNIST dataset loader"""
    def test_loading(self):
        """Verify that the loaded objects have the expected dimensions"""
        (x_train, y_train), (x_test, y_test) = load_data()
        self.assertEqual(x_train.shape, (6e4, 784))
        self.assertEqual(y_train.shape, (6e4,))
        self.assertEqual(x_test.shape, (1e4, 784))
        self.assertEqual(y_test.shape, (1e4,))

    def test_parameter_load(self):
        """Verify that the model is able to load parameters from files"""
        layer_sizes = [784, 512, 256, 10]
        model = MultiLayerPerceptron(layer_sizes)

        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
            weights0 = model.weights[0]
            biases0 = model.biases[0]
            np.savez(tmp.name, weights=weights0, biases=biases0)

            weights_from_file, biases_from_file = load_layer(tmp.name)
            np.testing.assert_array_equal(weights0, weights_from_file)
            np.testing.assert_array_equal(biases0, biases_from_file)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_parameters(destination_dir=tmpdir)
            weights, biases = load_parameters(tmpdir)

        self.assertTrue(len(weights) == len(biases) == 3)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(weights[2], model.weights[1])
        np.testing.assert_array_equal(weights[2], model.weights[2])
        np.testing.assert_array_equal(biases[2], model.biases[2])
