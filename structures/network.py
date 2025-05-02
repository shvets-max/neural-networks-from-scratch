import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from structures.base import (
    softmax,
    sigmoid,
    sigmoid_deriv,
    mse_deriv,
    identity,
    relu,
    relu_deriv,
    identity_deriv,
    accuracy, f1_score_macro, cross_entropy
)
from structures.optimized_layer import OptimizedLayer

class OOPNetwork:
    def __init__(self, layer_sizes, task="classification", activation="relu"):
        act_map = {"relu": (relu, relu_deriv), "sigmoid": (sigmoid, sigmoid_deriv)}
        self.activation, self.activation_deriv = act_map[activation]
        self.task = task
        self.is_multiclass = layer_sizes[-1] > 1 and task == "classification"

        self.layers = []
        for i in range(len(layer_sizes) - 1):
            is_last = (i == len(layer_sizes) - 2)
            act_func = (
                identity if task == "regression" and is_last else
                softmax if self.is_multiclass and is_last else
                self.activation
            )
            act_deriv = (
                identity_deriv if task == "regression" and is_last else
                lambda z: np.ones_like(z) if self.is_multiclass and is_last else
                self.activation_deriv
            )
            self.layers.append(OptimizedLayer(layer_sizes[i], layer_sizes[i + 1], act_func, act_deriv))

    def forward(self, x):
        a = x
        for layer in self.layers:
            a = layer.activate(a)
        return a

    def backward(self, x, y, lr):
        pred = self.forward(x)
        y = y.flatten()
        if self.task == "regression":
            delta = mse_deriv(pred, y)
        elif self.is_multiclass:
            one_hot = np.zeros_like(pred)
            one_hot[int(y)] = 1
            delta = pred - one_hot
        else:
            delta = (pred - y) * sigmoid_deriv(self.layers[-1].last_z)

        deltas = [delta]
        for i in reversed(range(len(self.layers) - 1)):
            layer = self.layers[i]
            next_weights = self.layers[i + 1].weights
            next_deltas = deltas[0]
            deltas.insert(0, layer.compute_deltas(next_weights, next_deltas))

        for layer, delta in zip(self.layers, deltas):
            layer.backward(delta, lr)

    def predict(self, x):
        out = self.forward(x)
        if self.task == "regression":
            return out
        elif self.is_multiclass:
            return np.argmax(out)
        else:
            return int(out[0] >= 0.5)

class VectorizedNetwork:
    def __init__(self, layer_sizes, task="classification", activation="sigmoid"):
        act_map = {"relu": (relu, relu_deriv), "sigmoid": (sigmoid, sigmoid_deriv)}
        self.activation, self.activation_deriv = act_map[activation]

        self.weights = [np.random.randn(n_out, n_in + 1) * 0.01
                        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.task = task
        self.is_multiclass = layer_sizes[-1] > 1 and task == "classification"

    def forward(self, x):
        a = x
        activations = [a]
        zs = []
        for i, w in enumerate(self.weights):
            bias_input = np.insert(a, 0, 1.0)  # Add bias term
            z = w @ bias_input
            zs.append(z)
            if i == len(self.weights) - 1:
                if self.task == "regression":
                    a = identity(z)
                elif self.is_multiclass:
                    a = softmax(z)
                else:
                    a = sigmoid(z)
            else:
                a = self.activation(z)
            activations.append(a)
        return activations, zs

    def backward(self, x, y, lr):
        activations, zs = self.forward(x)
        pred = activations[-1]

        # Compute output error
        if self.task == "regression":
            delta = mse_deriv(pred, y)
        elif self.is_multiclass:
            one_hot = np.zeros_like(pred)
            one_hot[int(y)] = 1
            delta = pred - one_hot
        else:
            delta = (pred - y) * sigmoid_deriv(zs[-1])

        deltas = [delta]

        for i in reversed(range(len(self.weights) - 1)):
            z = zs[i]
            W = self.weights[i + 1][:, 1:]  # Remove bias column
            delta = self.activation_deriv(z) * (W.T @ deltas[0])
            deltas.insert(0, delta)

        bias_inputs = [np.insert(a, 0, 1.0) for a in activations[:-1]]
        grads = [lr * np.outer(d, b) for d, b in zip(deltas, bias_inputs)]
        self.weights = [w - g for w, g in zip(self.weights, grads)]


    def predict(self, x):
        a = x
        for i, w in enumerate(self.weights):
            a = np.insert(a, 0, 1.0)
            z = w @ a
            if i == len(self.weights) - 1:
                if self.task == "regression":
                    return z
                elif self.is_multiclass:
                    return np.argmax(softmax(z))
                else:
                    return int(sigmoid(z[0]) >= 0.5)
            else:
                a = self.activation(z)
        return None

if __name__ == "__main__":

    # Data setup
    n_classes = 3
    last_layer_size = 1 if n_classes == 2 else n_classes
    X, y = make_classification(n_samples=500, n_features=5, n_informative=5, n_classes=n_classes,
                               n_redundant=0, n_clusters_per_class=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y.reshape(-1, 1), test_size=0.2, random_state=42)

    # Network parameters
    layer_sizes = [5, 16, 64, last_layer_size]
    epochs = 200
    lr = 0.1

    # Instantiate networks
    network = OOPNetwork(layer_sizes, task="classification", activation="sigmoid")

    # Training vectorized network
    # Training network
    for epoch in range(epochs):
        loss = 0
        for xi, yi in zip(X_train, y_train):
            network.backward(xi, yi, lr)
            pred_vec = network.forward(xi)  # full softmax vector
            # One-hot encode target
            one_hot = np.zeros_like(pred_vec)
            one_hot[int(yi)] = 1
            # Cross-entropy loss
            loss += -np.sum(one_hot * np.log(pred_vec + 1e-9))  # add epsilon for numerical stability

        if epoch % 20 == 0 or epoch == epochs - 1:
            predicted = np.array([network.predict(x) for x in X_test])
            acc = accuracy(y_test, predicted)
            f1 = f1_score_macro(y_test, predicted)
            print(f"loss: {loss}, accuracy: {acc}, f1: {f1}.")
