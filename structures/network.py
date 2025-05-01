import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class VectorizedNetwork:
    def __init__(self, layer_sizes, activation_func, activation_deriv):
        self.weights = [np.random.randn(n_out, n_in + 1) * 0.01
                        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.activation = activation_func
        self.activation_deriv = activation_deriv

    def forward(self, x):
        a = x
        activations = [a]
        zs = []
        for w in self.weights:
            bias_input = np.empty(a.shape[0] + 1)
            bias_input[0] = 1.0
            bias_input[1:] = a
            z = w @ bias_input
            zs.append(z)
            a = self.activation(z)
            activations.append(a)
        return activations, zs

    def backward(self, x, y, lr):
        activations, zs = self.forward(x)
        y = float(y)
        delta = (activations[-1] - y) * self.activation_deriv(zs[-1])
        deltas = [delta]

        for i in reversed(range(len(self.weights) - 1)):
            W = self.weights[i + 1][:, 1:]
            z = zs[i]
            delta = self.activation_deriv(z) * (W.T @ deltas[0])
            deltas.insert(0, delta)

        for i in range(len(self.weights)):
            bias_input = np.empty(activations[i].shape[0] + 1)
            bias_input[0] = 1.0
            bias_input[1:] = activations[i]
            self.weights[i] -= lr * np.outer(deltas[i], bias_input)

    def predict(self, x):
        a = x
        for w in self.weights:
            bias_input = np.empty(a.shape[0] + 1)
            bias_input[0] = 1.0
            bias_input[1:] = a
            a = self.activation(w @ bias_input)
        return 1 if a[0] >= 0.5 else 0

    def evaluate(self, X, y):
        correct = 0
        for xi, yi in zip(X, y):
            correct += self.predict(xi) == yi[0]
        return correct / len(X)

# Utility functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

if __name__ == "__main__":

    # Data setup
    X, y = make_classification(n_samples=500, n_features=5, n_informative=5,
                               n_redundant=0, n_clusters_per_class=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y.reshape(-1, 1), test_size=0.2, random_state=42)

    # Network parameters
    layer_sizes = [5, 10, 20, 1]
    epochs = 200
    lr = 0.1

    # Instantiate networks
    vector_net = VectorizedNetwork(layer_sizes, sigmoid, sigmoid_deriv)

    # Training vectorized network
    for epoch in range(epochs):
        loss = 0
        for xi, yi in zip(X_train, y_train):
            vector_net.backward(xi, yi, lr)
            pred = vector_net.forward(xi)[0][-1][0]
            loss += 0.5 * (pred - yi[0]) ** 2

    vectorized_accuracy = vector_net.evaluate(X_test, y_test)
    print(vectorized_accuracy)
