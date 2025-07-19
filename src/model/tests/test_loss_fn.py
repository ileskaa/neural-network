"""Tests for the cross-entropy loss function and its gradient"""

import unittest
import numpy as np
from model.activations import softmax
from model.loss import cross_entropy, cross_entropy_gradient
from model.nn_utils import one_hot_encode


class TestCrossEntropy(unittest.TestCase):
    """Tests for the cross-entropy loss function"""

    def setUp(self):
        """Generate an array with values in the range 0-9"""
        self.y_true = np.random.randint(0, 10, size=400)

    def test_one_hot_encoding(self):
        """Ensure one-hot encoding works as expecped.
        It will be required by the loss function.
        
        Since one-hot encoding turns every digit into a vector
        composed of one 1 and nine 0s, the sum of the matrix
        of one-hot encoded digits should equal the number of rows. 
        """
        size = self.y_true.size
        hot_encoded = one_hot_encode(self.y_true)
        possible_digits = 10
        self.assertEqual(hot_encoded.shape, (size, possible_digits))
        self.assertEqual(np.sum(hot_encoded), size)

    def test_loss_function(self):
        """Tests for the cross-entropy loss function.

        Cross entropy should be lower
        when the probability distributions are more alike.

        The cross-entropy score should always be >= 0.
        """
        hot_encoded_5 = one_hot_encode(5)
        raw_scores = np.array([2,1,1,1,1,1,1,1,1,1])
        normalized_scores = softmax(raw_scores)
        self.assertEqual(hot_encoded_5.shape, normalized_scores.shape)

        cross_entropy_score = cross_entropy(hot_encoded_5, normalized_scores)

        better_raw_scores = np.array([1,1,1,1,1,2,1,1,1,1])
        better_normalized_scores = softmax(better_raw_scores)
        lower_cross_entropy = cross_entropy(
            hot_encoded_5, better_normalized_scores
        )
        self.assertTrue(lower_cross_entropy > 0 and cross_entropy_score > 0)
        self.assertTrue(lower_cross_entropy < cross_entropy_score)

        # Should raise an error if the shapes do not match
        with self.assertRaises(ValueError):
            too_long = np.array([1,1,1,1,1,2,1,1,1,1,1])
            cross_entropy(hot_encoded_5, too_long)

        # The loss function should also work with 2D arrays
        two_d_array = np.array([[2,1,1,1,1,1,1,1,1,1],
                                [1,1,1,1,1,2,1,1,1,1]])
        normalized_2d_arr = softmax(two_d_array)
        hot_encoded_arr = one_hot_encode([3, 2])
        cross_entropy_score = cross_entropy(hot_encoded_arr, normalized_2d_arr)
        self.assertTrue(cross_entropy_score > 0)

        # Should raise an error if the first loss function argument
        # is not one-hot encoded
        with self.assertRaises(ValueError):
            zeros = np.zeros(10)
            cross_entropy(zeros, normalized_scores)
        with self.assertRaises(ValueError):
            two_d_zeros = np.zeros((2, 10))
            cross_entropy(two_d_zeros, normalized_2d_arr)

    def test_loss_func_gradient(self):
        """Tests for the gradient of the cross-entropy loss
        w.r.t. pre-softmax values.

        For a single input, the gradient should be
        ∂L/∂z = y_pred - y_true
        """
        true_value = 5
        raw_scores = np.array([2,1,1,1,1,1,1,1,1,1])
        y_pred = softmax(raw_scores)
        # Should raise an error if the correct value
        # is not one-hot encoded
        with self.assertRaises(ValueError):
            cross_entropy_gradient(y_pred, true_value)

        y_true = one_hot_encode(true_value)
        self.assertEqual(y_pred.shape, y_true.shape)

        expected_gradient = y_pred - y_true
        returned_gradient = cross_entropy_gradient(y_pred, y_true)
        # We use assert_array_almost_equal because of
        # potential floating point precision issues
        np.testing.assert_array_almost_equal(expected_gradient, returned_gradient)

        # Verify the gradient is correct when working with batches
        sample_size = 400
        num_classes = 10
        array_size = (sample_size, num_classes)
        batch = np.random.normal(size=array_size)
        normalized_batch = softmax(batch)
        y_true_array = one_hot_encode(self.y_true)
        self.assertEqual(batch.shape, y_true_array.shape)
        expected_gradient = (normalized_batch - y_true_array) / sample_size
        returned_gradient = cross_entropy_gradient(normalized_batch, y_true_array)
        np.testing.assert_array_almost_equal(expected_gradient, returned_gradient)
