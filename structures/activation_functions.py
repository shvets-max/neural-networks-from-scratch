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
        try:
            return 1 / (1 + np.exp(-self._lambda * x))
        except OverflowError as e:
            return 1 / (1 + np.exp(-self._lambda * x / 10))

    def derivative(self, x):
        return self._lambda * self.func(x) * (1 - self.func(x))

    def __call__(self, *args, **kwargs):
        arg = args[0]
        if type(arg) in (int, float):
            return self.func(arg)
        elif type(arg) == np.ndarray:
            return np.vectorize(self.func)(arg)