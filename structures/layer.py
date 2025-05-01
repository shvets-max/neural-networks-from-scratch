from typing import List

import numpy as np
from structures.neuron import Neuron

class Layer:
    def __init__(self, neurons: List[Neuron]):
        self.neurons = neurons

    def activate(self, inputs: np.ndarray):
        return np.array([neuron.activate(inputs) for neuron in self.neurons])

    def backward(self, deltas, lr: float = 0.01):
        for neuron, delta in zip(self.neurons, deltas):
            neuron.update_weights(delta, lr)

    def compute_deltas(self, next_weights: np.ndarray, next_deltas: np.ndarray):
        deltas = []
        for i, neuron in enumerate(self.neurons):
            errors = np.dot(next_weights[:, i + 1], next_deltas)  # skip bias
            delta = errors * neuron.activation_function.derivative(neuron.last_z)
            deltas.append(delta)
        return np.array(deltas)

    def get_outputs(self):
        return np.array([neuron.last_output for neuron in self.neurons])

    def get_weights_matrix(self):
        return np.array([neuron.weights for neuron in self.neurons])

