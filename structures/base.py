import numpy as np

# --- Activation functions ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(x):
    return np.exp(x) / np.exp(x).sum()

def identity(x):
    return x

def identity_deriv(x):
    return np.ones_like(x)


# --- Loss functions ---
def cross_entropy(pred, target):
    return -np.sum(target * np.log(pred + 1e-9))

def mse(pred, target):
    return 0.5 * np.sum((pred - target)**2)

def cross_entropy_deriv(pred, target):
    return pred - target

def mse_deriv(pred, target):
    return pred - target

# --- Metrics ---
def accuracy(actual: np.ndarray, predicted: np.ndarray):
    actual = actual.flatten()
    predicted = predicted.flatten()
    return np.mean(actual == predicted)

def precision(actual: np.ndarray, predicted: np.ndarray, label: int):
    actual = actual.flatten()
    predicted = predicted.flatten()
    predicted_positive = (predicted == label)
    tp = np.sum((actual == label) & predicted_positive)
    return tp / np.sum(predicted_positive) if np.sum(predicted_positive) > 0 else 0.0

def recall(actual: np.ndarray, predicted: np.ndarray, label: int):
    actual = actual.flatten()
    predicted = predicted.flatten()
    actual_positive = (actual == label)
    tp = np.sum((predicted == label) & actual_positive)
    return tp / np.sum(actual_positive) if np.sum(actual_positive) > 0 else 0.0

def f1_score(actual, predicted, label):
    prec = precision(actual, predicted, label)
    rec = recall(actual, predicted, label)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

def f1_score_macro(actual, predicted):
    actual = actual.flatten()
    predicted = predicted.flatten()
    return np.mean([f1_score(actual, predicted, label) for label in np.unique(actual)])
