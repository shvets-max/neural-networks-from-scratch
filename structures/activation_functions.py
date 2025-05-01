import numpy as np


class ActivationFunction:
    def __init__(self, name, lbd=1):
        self._lambda = lbd
        self._name = name

    def derivative(self, *args):
        pass

    def __call__(self, *args, **kwargs):
        pass


class Sigmoid(ActivationFunction):
    def __init__(self, name="sigmoid", lbd=1):
        super().__init__(name, lbd)

    def func(self, x):
        return 1 / (1 + np.exp(-self._lambda * x))

    def derivative(self, x):
        fx = self.func(x)
        return self._lambda * fx * (1 - fx)

    def __call__(self, x):
        return self.func(x)