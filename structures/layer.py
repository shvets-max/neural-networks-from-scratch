from typing import List

import numpy as np
from structures.neuron import Neuron
from structures.activation_functions import Sigmoid


class Layer:
    def __init__(self, neurons: List[Neuron]):
        self.neurons = neurons

    def activate(self, inputs: np.ndarray):
       return np.array([n.activate(inputs) for n in self.neurons])

    def calculate_error(self, weights, errors):
        return np.array([n.calculate_error(w, errors) for n, w in zip(self.neurons, weights)])

    def update_weights(self, neuron_inputs, neuron_outputs, errors):
        for neuron, inp, out in zip(self.neurons, neuron_inputs, neuron_outputs):
            neuron.update_weights(inp, out, errors)

    @property
    def weights(self):
        return np.asarray([neuron.weights for neuron in self.neurons])


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    # Generate dataset
    X, y = make_classification(n_samples=25, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, random_state=42)
    l0 = Layer(neurons=[
        Neuron(
            input_shape=2,
            activation_function=Sigmoid
        ) for _ in range(10)]
    )
    l1 = Layer(neurons=[
        Neuron(
            input_shape=10,
            activation_function=Sigmoid
        ) for _ in range(20)]
    )
    l2 = Layer(neurons=[
        Neuron(
            input_shape=20,
            activation_function=Sigmoid
        )]
    )

    # Epoch 1.

    # Forward propagation
    l0_outputs = l0.activate(X)
    l1_outputs = l1.activate(l0_outputs.transpose())
    l2_outputs = l2.activate(l1_outputs.transpose())

    errors = np.mean(l2_outputs - y)

    # Backward propagation
    l2_errors = l2.calculate_error(l2.weights, errors)
    l1_errors = l1.calculate_error(l1.weights, l2_errors)
    l0_errors = l0.calculate_error(l1_errors, l0.weights)

    # Update weights
    l0.update_weights(X, l0_outputs, l0_errors)
    l1.update_weights(l0_outputs, l1_outputs, l1_errors)
    l2.update_weights(l1_outputs, l2_outputs, l2_errors)

    # Epoch 1 end.

