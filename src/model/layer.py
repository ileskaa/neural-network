"""Neural network layer class"""

import numpy as np


class Layer:
    """Implement a neural network layer.
    Useful when working with the Adam optimization algorithm.
    """

    def __init__(self, weights: np.ndarray, biases: np.ndarray) -> None:
        self.weights = weights
        self.biases = biases
        # First moment vectors
        self.m_w: int | np.ndarray = 0
        self.m_b: int | np.ndarray = 0
        # Second moment vectors
        self.v_w = self.v_b = 0
        self.alpha = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.t = 0

    def compute_estimates(
        self,
        m: np.ndarray | int,
        v: np.ndarray | int,
        t: int,
        gradient: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute first and second moment estimates for the given gradient"""
        # Update biased first moment estimates
        m = self.beta1 * m + (1 - self.beta1) * gradient
        # Update biased second raw moment estimates
        # ISSUE: Values within v can grow to infinity
        v = self.beta2 * v + (1 - self.beta2) * gradient**2
        v = np.clip(v, 1e-4, 1e4)
        max_v = np.max(v)
        # print("max v", max_v)
        if np.isinf(max_v):
            raise ValueError("Some value in v reached infinity")
        # Compute bias-corrected 1st moment estimates
        m_hat = m / (1 - self.beta1**t)
        # Compute bias-corrected 2nd raw moment estimates
        beta2_t = self.beta2**t
        # ISSUE: If 1 - beta2_t is very small, v_hat might grow like crazy
        # Might be useful to add espilon within the parentheses
        v_hat = v / (1 - beta2_t)
        return m_hat, v_hat

    def update_weights(self, grad_w: np.ndarray, t: int = 1):
        """Update layer weights and their moment estimates"""
        m, v = self.compute_estimates(self.m_w, self.v_w, t, grad_w)
        self.m_w = m
        self.v_w = v
        theta = self.weights
        epsilon = 1e-8
        self.weights = theta - self.alpha * m / (np.sqrt(v) + epsilon)

    def update_biases(self, grad_b: np.ndarray, t: int = 1):
        """Update layer biases and their moment estimates"""
        m, v = self.compute_estimates(self.m_b, self.v_b, t, grad_b)
        self.m_b = m
        self.v_b = v
        theta = self.biases
        epsilon = 1e-8
        self.biases = theta - self.alpha * m / (np.sqrt(v) + epsilon)

    def update_layer(self, grad_w, grad_b):
        """Update layer weights and biases based on adaptive moment estimtion"""
        self.t += 1
        self.update_weights(grad_w, self.t)
        self.update_biases(grad_b, self.t)
