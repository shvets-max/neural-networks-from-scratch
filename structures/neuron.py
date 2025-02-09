import numpy as np
from structures.activation_functions import ActivationFunction


class Neuron:
    weights: np.ndarray
    activation_function: ActivationFunction

    def activate(self, inputs: np.ndarray):
        if inputs.shape != self.weights.shape:
            raise ValueError("inputs shape is not the same as weights shape.")
        # Add bias to inputs
        _inputs = np.insert(inputs, 0, 1)

        # Calculate dot product
        dot_product = np.dot(_inputs, self.weights)

        # Apply activation function
        return self.activation_function(dot_product)

    def calculate_error(self, weights, errors):
        return np.dot(weights, errors)

    def update_weights(self, inp, out, delta, lr: float = .01):
        # Calculate deltas for weights
        delta_weights = np.array(
            [lr * delta * self.activation_function.derivative(out) * _inp for _inp in inp]
        )

        # Update neuron's input weights
        self.weights = self.weights + delta_weights
