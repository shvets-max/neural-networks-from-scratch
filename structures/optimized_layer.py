import numpy as np


class OptimizedLayer:
    def __init__(self, input_dim, output_dim, activation_func, activation_deriv):
        self.weights = np.random.randn(output_dim, input_dim + 1) * 0.01
        self.activation = activation_func
        self.activation_deriv = activation_deriv
        self.last_input = None
        self.last_z = None
        self.last_output = None

    def activate(self, inputs):
        bias_input = np.concatenate(([1], inputs))
        self.last_input = bias_input
        z = np.dot(self.weights, bias_input)
        self.last_z = z
        self.last_output = self.activation(z)
        return self.last_output

    def compute_deltas(self, next_weights, next_deltas):
        # next_weights: shape (next_layer_neurons, current_layer_neurons + 1)
        W = next_weights[:, 1:]  # skip bias
        delta = self.activation_deriv(self.last_z) * np.dot(W.T, next_deltas)
        return delta

    def backward(self, deltas, lr):
        grad = np.outer(deltas, self.last_input)
        self.weights -= lr * grad