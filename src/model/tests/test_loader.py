"""Tests for the MNIST dataset loader"""

import unittest
from model.mnist_loader import load_data


class TestLoader(unittest.TestCase):
    """Unit tests for the dataset loader"""
    def test_loading(self):
        """Verify that the loaded objects have the expected dimensions"""
        (x_train, y_train), (x_test, y_test) = load_data()
        self.assertEqual(x_train.shape, (6e4, 784))
        self.assertEqual(y_train.shape, (6e4,))
        self.assertEqual(x_test.shape, (1e4, 784))
        self.assertEqual(y_test.shape, (1e4,))
