"""Tests for the MultiLayerPerceptron class"""

import io
import unittest
import unittest.mock
import numpy as np
from model import loss
from model.mlp import MultiLayerPerceptron
from model.nn_utils import one_hot_encode, normalize_image_data


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

    def gen_x_sample(self, out_type='single', num_images=100):
        """Generate a random sample of pixel values to simulate image data.
        
        This method prevents repetition since samples are used in a bunch of tests.

        Can return either a single image or an array of images.
        """
        low = 0
        high = 256 # Upper bound is exclusive
        num_pixels = 784
        if out_type == 'single':
            return self.rng.integers(low, high, num_pixels)
        return self.rng.integers(low, high, size=(num_images, num_pixels))

    def gen_y_sample(self):
        """Generate a random sample of digits"""
        num_digits = 100
        low = 0
        high = 10
        return self.rng.integers(low, high, size=num_digits)

    def test_forward(self):
        """Testing the forward pass method of the network.
        
        We'll generate 784 integers to simulate a grayscale digit.
        Then forwards them through the network.
        At the end, we should get a discrete probability distribution
        of 10 values.
        """
        x = self.gen_x_sample('single')
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

        The backprop method is expected to raise an error if it is called
        before a forward pass is run.
        The forward pass will initialize lists of activations and z vectors,
        which are required by the forward pass.

        We'll be using a batch of 100 samples for these tests.
        """
        y_true = self.gen_y_sample()
        # Backprop should raise an error because we have yet to do a forward pass
        with self.assertRaises(IndexError):
            self.model.backprop(y_true)

        # Raise an error if x and y are based on a different number of images
        x = self.gen_x_sample('array', num_images=99)
        output = self.model.forward(x)
        with self.assertRaises(ValueError):
            self.model.backprop(y_true)

        x = self.gen_x_sample('array')
        x = normalize_image_data(x)
        output = self.model.forward(x)
        y_true_encoded = one_hot_encode(y_true)
        cross_entropy = loss.cross_entropy(y_true_encoded, output)

        # Weights and biases should be updated after backpropagation
        inital_weights = [w.copy() for w in self.model.weights]
        initial_biases = [b.copy() for b in self.model.biases]
        self.model.backprop(y_true)
        weights_after = self.model.weights
        biases_after = self.model.biases
        n = len(inital_weights)
        for layer in range(n):
            equal_weights = np.array_equal(inital_weights[layer], weights_after[layer])
            equal_biases = np.array_equal(initial_biases[layer], biases_after[layer])
            self.assertFalse(equal_weights)
            self.assertFalse(equal_biases)

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
        single_image = self.gen_x_sample()
        y_pred = self.model.predict(single_image)
        self.assertIsInstance(y_pred, np.int64)

        # Test array inputs
        image_vector = self.gen_x_sample('array')
        y_pred = self.model.predict(image_vector)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertTrue(np.issubdtype(y_pred.dtype, np.int64))

    # The patch decorator patches a target with a new object.
    # For this test, we'll patch the standard ouput with the StringIO method,
    # which will allow us to test if the training method prints the expected strings.
    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_train(self, mock_stdout):
        """Tests for the network's training method.

        The training method should update model parameters
        and print data about the advancement of the training process.

        The method should raise an error if pixel data is not normalized.
        """
        x = self.gen_x_sample('array')
        y = self.gen_y_sample()
        # Should raise error if data is not normalized
        with self.assertRaises(ValueError):
            self.model.train(x, y, epochs=1)

        # Cross-entropy should be lower after training
        x = normalize_image_data(x)
        output = self.model.forward(x)
        y_hot_enc = one_hot_encode(y)
        ce_score = loss.cross_entropy(output, y_hot_enc)
        self.model.train(x, y, epochs=1)
        new_output = self.model.forward(x)
        new_ce_score = loss.cross_entropy(new_output, y_hot_enc)
        self.assertTrue(new_ce_score < ce_score)

        # The method should print training-related information
        standard_output = mock_stdout.getvalue()
        self.assertIn("Loss", standard_output)
        self.assertIn("Elapsed time", standard_output)

    @unittest.mock.patch('sys.stdout', new_callable=io.StringIO)
    def test_accuracy_(self, mock_stdout):
        """Verify that the accuracy measurement method prints to the standard output"""
        x = self.gen_x_sample('array')
        y = self.gen_y_sample()
        self.model.measure_accuracy(x,y)
        std_output = mock_stdout.getvalue()
        self.assertIn("Accuracy", std_output)
        self.assertIn("%", std_output)
