from typing import List

import numpy as np
from structures.neuron import Neuron
from structures.activation_functions import Sigmoid


class Layer:
    neurons: List[Neuron]

    def activate(self, inputs: np.ndarray):
       return [n.activate(inputs) for n in self.neurons]

    def calculate_error(self, weights, errors):
        return [n.calculate_error(w, errors) for n, w in zip(self.neurons, weights)]

    def update_weights(self, neuron_inputs, neuron_outputs, errors):
        for neuron, inp, out in zip(self.neurons, neuron_inputs, neuron_outputs):
            neuron.update_weights(inp, out, errors)

    @property
    def weights


# if __name__ == "__main__":
#     l0 = Layer(neurons=[
#         Neuron(
#             weights=np.random.random(5, ),
#             activation_function=Sigmoid
#         ) for _ in range(10)]
#     )
#     l1 = Layer(neurons=[
#         Neuron(
#             weights=np.random.random(10, ),
#             activation_function=Sigmoid
#         ) for _ in range(20)]
#     )
#     l2 = Layer(neurons=[
#         Neuron(
#             weights=np.random.random(20, ),
#             activation_function=Sigmoid
#         ) for _ in range(3)]
#     )
#
#     l0_outputs = l0.activate(inputs)
#     l1_outputs = l1.activate(l0_outputs)
#     l2_outputs = l2.activate(l1_outputs)
#
#     errors = np.mean(l2_outputs - outputs, axis=1)
#
#     # Weights: Number of neurons in current layer + bias times number of neurons in next layer
#     weights = None  # Matrix of shape [(N_curr + 1), N_next]
#
#     l2_errors = l2.calculate_error(weights, errors)