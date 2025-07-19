"""Tests for the nn_utils module"""

import unittest
import math
import numpy as np
from nn_utils import he_initalization

def normal_distrib_pdf(x, mean = 0, variance = 1):
    """Probability density function for the normal distribution.
    Used for testing He initialization.
    """
    return (
        1 / np.sqrt(2*math.pi*variance)
    ) * np.exp(- (x-mean)**2 / (2*variance))

def trapezoidal(f, lower_limit = 0, upper_limit = 10, n = 10**4):
    """Estimate the integral of f using the trapezoidal rule with n partitions.
    Used for testing He initialization.
    """
    # Need to compute x, y and h
    x = np.linspace(lower_limit, upper_limit, n+1)
    y = f(x)
    partition_width = (upper_limit - lower_limit) / n
    integral_estimate = partition_width * (
        (y[0] + y[-1]) / 2 + np.sum(y[1:-1])
    )
    return integral_estimate


class TestHeInitialization(unittest.TestCase):
    """Testing weight initialization"""
    def setUp(self):
        self.weights = [
            he_initalization(784, 512),
            he_initalization(512, 256),
        ]
        # Generate a matrix whose values are picked
        # from a standard normal distribution
        self.x = np.random.normal(0, 1, (400, 784))

    def test_shape(self):
        """Verify that the generated array has the expected dimensions"""
        weights = self.weights[0]
        self.assertEqual(weights.shape, (784, 512))

    def test_std(self):
        """The standard deviation should roughly equal sqrt(2/input_size)"""
        for weights in self.weights:
            input_size = weights.shape[0]
            expected_std = math.sqrt(2 / input_size)
            std = np.std(weights)
            self.assertTrue(
                expected_std*0.95 < std < expected_std*1.05
            )

    def test_variance_doubling(self):
        """Thanks to He initialization, the weights should roughly double
        the variance of the matrix they are multiplied with.
        """
        weights = self.weights[0]
        z = np.matmul(self.x, weights)
        self.assertTrue(1.95 < np.var(z)/np.var(self.x) < 2.05)

    def test_first_moment(self):
        """The first moment (expected value) of ReLU(X) should evaluate to
        1/sqrt(2*pi) when the input X is a standard normal variable.
        We'll verify this by approximating the integral.

        To get the first moment, we compute the integral of x * f(x)
        where f(x) is the probability density function of X.
        The lower limit of the integral is 0
        since ReLU transforms all negative values to 0.
        The integral is hence computed over the interval [0, infinity].
        """
        lower_lim = 0
        # Technically the upper limit of the integral should be infinity
        # but the PDF of the standard normal distribution gets
        # so thin at the edges that an upper limit of 10
        # is enough to give a good approximation
        upper_lim = 10
        partitions = 10*4
        # Using the trapezoidal rule, we get a good estimate of the integral
        estimated_integral = trapezoidal(
            lambda x: x * normal_distrib_pdf(x),
            lower_lim,
            upper_lim,
            partitions
        )
        expected_integral = 1/math.sqrt(2*math.pi)
        self.assertTrue(0.99 < estimated_integral / expected_integral < 1.01)

    def test_second_moment(self):
        """E[X^2] is the 2nd moment of the random variable X.
        If X follows a standard normal distribution,
        the second moment of ReLU(X) should equal 1/2.
        To show this, we compute the integral of x^2 * f(x)
        where f(x) is the probability density function of X.
        As with the first moment the lower limit of the integral is 0
        since ReLU transforms all negative values to 0.
        The integral is hence computed over the interval [0, infinity].
        """
        estimated_integral = trapezoidal(
            lambda x: x**2 * normal_distrib_pdf(x),
        )
        expected_integral = 1/2
        self.assertTrue(0.99 < estimated_integral / expected_integral < 1.01)

    def test_output_variance(self):
        """Test the variance after ReLU activation.

        For a random variable X, the variance is
        Var(X) = E[X^2] - E[X]^2
        In other words, the variance is the second moment
        minus the square of the first moment of that random variable.

        We'll be estimating the first and second moments of ReLU(Z)
        where Z is the input multiplied by the weights.
        We'll obtain these estimates using the trapezoidal rule.

        Due to He initialization, Z will have a variance very close to 2.
        """
        first_moment_approx = trapezoidal(
            lambda z: z * normal_distrib_pdf(z, variance=2),
        )

        second_moment_approx = trapezoidal(
            lambda z: z**2 * normal_distrib_pdf(z, variance=2),
        )

        expected_variance = second_moment_approx - first_moment_approx**2

        weights = self.weights[0]
        z = np.matmul(self.x, weights)

        # We verify that the expected variance is roughly equal
        # to the variance of the ReLU output
        self.assertTrue(
            0.98 < expected_variance / np.var(np.maximum(z, 0)) < 1.02
        )
