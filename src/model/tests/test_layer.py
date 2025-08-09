"""Module for testing the layer class"""

import unittest
import numpy as np
from model.layer import Layer
from model.nn_utils import he_initalization


class TestLayer(unittest.TestCase):
    """Unit tests for the Layer class"""

    def setUp(self) -> None:
        """Create a layer to be used during testing"""
        self.input_size = 32
        self.hidden_size = 16
        weights = he_initalization(self.input_size, self.hidden_size)
        biases = np.zeros(self.hidden_size)
        self.layer = Layer(weights, biases)
        self.rng = np.random.default_rng(21)

    def test_initialization(self):
        """Verify that the layer is initialized with the correct shapes"""
        self.assertEqual(self.layer.weights.shape, (self.input_size, self.hidden_size))
        self.assertEqual(self.layer.biases.shape, (self.hidden_size,))

    def test_compute_estimates(self):
        """Verify that first and second moment estimates are computed properly"""

        grad_w = self.rng.random((self.input_size, self.hidden_size))
        m, v = self.layer.compute_estimates(0, 0, 1, grad_w)
        self.assertTrue(grad_w.shape == m.shape == v.shape)

        grad_b = self.rng.random(self.hidden_size)
        m, v = self.layer.compute_estimates(0, 0, 1, grad_b)
        self.assertTrue(grad_b.shape == m.shape == v.shape)

    def test_weights_update(self):
        """Ensure weights and their moment estimates are getting updated"""
        weights_before = self.layer.weights.copy()
        grad_w = self.rng.random((self.input_size, self.hidden_size))
        self.layer.update_weights(grad_w)
        # Weights should now be different
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            weights_before,
            self.layer.weights,
        )
        # The weights matrix should still have the same shape
        self.assertEqual(weights_before.shape, self.layer.weights.shape)
        # First moment estimates should have taken the shape of the weight matrix
        if isinstance(self.layer.m_w, np.ndarray):
            self.assertEqual(self.layer.weights.shape, self.layer.m_w.shape)
        else:
            raise TypeError("Expected a numpy array")
        # Same for second moment estimates
        if isinstance(self.layer.v_w, np.ndarray):
            self.assertEqual(self.layer.weights.shape, self.layer.v_w.shape)
        else:
            raise TypeError("Expected a numpy array")

    def test_biases_update(self):
        """Ensure biases and their moment estimates are getting updated"""
        biases_before = self.layer.biases.copy()
        grad_b = self.rng.random(self.hidden_size)
        self.layer.update_biases(grad_b)
        # Weights should now be different
        parameters = self.layer.biases
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            biases_before,
            parameters,
        )
        # First moment estimates should have taken the shape of the bias vector
        if isinstance(self.layer.m_b, np.ndarray):
            self.assertEqual(parameters.shape, self.layer.m_b.shape)
        else:
            raise TypeError("Expected a numpy array")
        # Same for second moment estimates
        if isinstance(self.layer.v_b, np.ndarray):
            self.assertEqual(parameters.shape, self.layer.v_b.shape)
        else:
            raise TypeError("Expected a numpy array")

    def test_layer_update(self):
        """Ensure that both weights and biases get updated during a training cycle"""
        grad_w = self.rng.random((self.input_size, self.hidden_size))
        grad_b = self.rng.random(self.hidden_size)
        weights_before = self.layer.weights.copy()
        biases_before = self.layer.biases.copy()

        self.layer.update_layer(grad_w, grad_b)
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            weights_before,
            self.layer.weights,
        )
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            biases_before,
            self.layer.biases,
        )

        self.assertEqual(weights_before.shape, self.layer.weights.shape)
        self.assertEqual(biases_before.shape, self.layer.biases.shape)
