"""Tests for activation functions"""

import unittest
import numpy as np
from activations import relu, relu_derivative, softmax


class ActivationTest(unittest.TestCase):
    """Following tests will be subclassed from this class.

    Instructions within setUp() are executed before each test method.
    """

    def setUp(self):
        # Matrix with values picked from a standard normal distribution
        self.x = np.random.normal(0, 1, (400, 784))
        self.sample = self.x[0]

    def verify_shape(self, f):
        """Verify that the function preserves the shape of the input"""

        sample_output_shape = f(self.sample).shape
        self.assertEqual(self.sample.shape, sample_output_shape)

        input_shape = self.x.shape
        output = f(self.x)
        output_shape = output.shape
        self.assertEqual(input_shape, output_shape)


class TestRelu(ActivationTest):
    """Testing the ReLU activation"""

    def test_shape(self):
        """Verify that ReLU preserves the shape of the input"""
        self.verify_shape(relu)

    def test_zeroing(self):
        """Verify that roughly half the values equal zero after ReLU"""
        shape = self.x.shape
        expected_zero_count = shape[0] * shape[1] / 2
        actual_zero_count = np.sum(relu(self.x) == 0)
        self.assertTrue(0.99 < expected_zero_count / actual_zero_count < 1.01)


class TestReluDerivative(ActivationTest):
    """Testing the ReLU derivative"""

    def test_shape(self):
        """Verify that ReLU's derivative preserves the shape of the input"""
        self.verify_shape(relu_derivative)

    def test_values(self):
        """The output should contain only 0s and 1s"""
        shape = self.x.shape
        total_values = shape[0] * shape[1]
        derivatives = relu_derivative(self.x)
        zeroes = np.sum(derivatives == 0)
        ones = np.sum(derivatives == 1)
        self.assertEqual(total_values, zeroes + ones)


class TestSoftmax(ActivationTest):
    """Testing the softmax activation function"""

    def test_shape(self):
        """Verify that softmax activation preserves the shape of the input"""
        self.verify_shape(softmax)

    def test_sum(self):
        """Since sofmax converts values into probabilities,
        the sum of the probabilities within a vector should be 1.
        """
        self.assertEqual(self.sample.shape, (784,))

        sample_output = softmax(self.sample)
        sample_sum = np.sum(sample_output)
        self.assertAlmostEqual(sample_sum, 1)

        full_output = softmax(self.x)
        full_sum = np.sum(full_output)
        row_count = self.x.shape[0]
        self.assertAlmostEqual(full_sum, row_count)
