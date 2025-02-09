import numpy as np


class ActivationFunction:
    _lambda: float
    _name: str

    def derivative(self, *args):
        pass

    def __call__(self, *args, **kwargs):
        pass


class Sigmoid(ActivationFunction):
    def __init__(self, name="sigmoid", lbd=1):
        super().__init__(name, lbd)

    def func(self, x):
        try:
            return 1 / (1 + np.exp(-self._lambda * x))
        except OverflowError as e:
            return 1 / (1 + np.exp(-self._lambda * x / 10))

    def derivative(self, x):
        return self._lambda * self.func(x) * (1 - self.func(x))