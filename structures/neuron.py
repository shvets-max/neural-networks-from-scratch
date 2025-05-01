import numpy as np
from structures.activation_functions import ActivationFunction


class Neuron:
    def __init__(self, input_shape: int, activation_function: ActivationFunction):
        self.input_shape = input_shape
        self.weights = np.random.randn(input_shape + 1) * 0.01  # +1 for bias

        self.activation_function = activation_function
        self.last_input = None
        self.last_z = None
        self.last_output = None

    def activate(self, inputs: np.ndarray):
        bias_input = np.append(1, inputs)
        self.last_input = bias_input
        z = np.dot(self.weights, bias_input)
        self.last_z = z
        self.last_output = self.activation_function(z)
        return self.last_output

    def update_weights(self, delta, lr: float = 0.01):
        grad = lr * delta * self.last_input
        self.weights -= grad