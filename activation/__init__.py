from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm


class Base(ABC):
    def intr(self, X):
        I = self._intr(X, self.params)

        return I

    def func(self, X, I=None, full_output=False):
        if I is None:
            I = self._intr(X, self.params)

        F = self._func(I, self.params)

        if full_output:
            return F, I
        else:
            return F

    def grad(self, X, I=None, full_output=False):
        if I is None:
            I = self._intr(X, self.params)

        G = self._grad(I, self.params)

        if full_output:
            return G, I
        else:
            return G

    @staticmethod
    @abstractmethod
    def _intr(X, params):
        pass

    @staticmethod
    @abstractmethod
    def _func(I, params):
        pass

    @staticmethod
    @abstractmethod
    def _grad(I, params):
        pass


class ReLU(Base):
    def __init__(self, offset=0):
        self.params = {
            'offset': offset,
        }

    @staticmethod
    def _intr(X, params):
        offset = params['offset']

        A = X + offset
        return (A,)

    @staticmethod
    def _func(I, _):
        A = I[0]
        F = np.maximum(A, 0)
        return F

    @staticmethod
    def _grad(I, _):
        A = I[0]
        G = (A > 0).astype(float)
        return G


class LeakyReLU(Base):
    def __init__(self, offset=0, alpha=1e-2):
        self.params = {
            'offset': offset,
            'alpha': alpha,
        }

    @staticmethod
    def _intr(X, params):
        offset = params['offset']

        A = X + offset
        return (A,)

    @staticmethod
    def _func(I, params):
        alpha = params['alpha']

        A = I[0]
        F = np.where(A > 0, A, alpha * A)
        return F

    @staticmethod
    def _grad(I, params):
        alpha = params['alpha']

        A = I[0]
        G = np.where(A > 0, 1, alpha)
        return G


class SmoothReLU1(Base):
    def __init__(self, offset=0):
        self.params = {
            'offset': offset,
        }

    @staticmethod
    def _intr(X, params):
        offset = params['offset']

        A = X + offset
        B = A + 0.5
        return (A, B)

    @staticmethod
    def _func(I, _):
        A = I[0]
        B = I[1]
        F = np.where(A > 0.5, A,
                     np.where(A > -0.5, np.square(B) / 2, 0))
        return F

    @staticmethod
    def _grad(I, _):
        A = I[0]
        B = I[1]
        G = np.where(A > 0.5, 1,
                     np.where(A > -0.5, B, 0))
        return G


class LeakySmoothReLU1(Base):
    def __init__(self, offset=0, alpha=1e-2):
        assert alpha <= 1

        self.params = {
            'offset': offset,
            'alpha': alpha,
            'a': alpha - 0.5,
            'b': 0.5 * (alpha - alpha ** 2),
        }

    @staticmethod
    def _intr(X, params):
        offset = params['offset']

        A = X + offset
        B = A + 0.5
        return (A, B)

    @staticmethod
    def _func(I, params):
        alpha = params['alpha']
        a = params['a']
        b = params['b']

        A = I[0]
        B = I[1]
        F = np.where(A > 0.5, A,
                     np.where(A > a, np.square(B) / 2, alpha * A + b))
        return F

    @staticmethod
    def _grad(I, params):
        alpha = params['alpha']
        a = params['a']

        A = I[0]
        B = I[1]
        G = np.where(A > 0.5, 1,
                     np.where(A > a, B, alpha))
        return G


class SmoothReLU2(Base):
    def __init__(self, offset=0):
        self.params = {
            'offset': offset,
        }

    @staticmethod
    def _intr(X, params):
        offset = params['offset']

        A = X + offset
        B = np.square(A)
        return (A, B)

    @staticmethod
    def _func(I, _):
        A = I[0]
        B = I[1]
        C = 2 / 3 * B * A
        D = B + A / 2 + 1 / 12
        F = np.where(A > 0.5, A,
                     np.where(A > 0, D - C,
                              np.where(A > -0.5, D + C, 0)))
        return F

    @staticmethod
    def _grad(I, _):
        A = I[0]
        B = I[1]
        C = 2 * B
        D = 2 * A + 1 / 2
        G = np.where(A > 0.5, 1,
                     np.where(A > 0, D - C,
                              np.where(A > -0.5, D + C, 0)))
        return G


class LeakySmoothReLU2(Base):
    def __init__(self, offset=0, alpha=1e-2):
        assert 0 <= alpha <= 0.5

        self.params = {
            'offset': offset,
            'alpha': alpha,
            'a': np.sqrt(0.5 * alpha) - 0.5,
            'b': 0.5 * alpha - np.sqrt(2) / 3 * alpha ** (3 / 2),
        }

    @staticmethod
    def _intr(X, params):
        offset = params['offset']

        A = X + offset
        B = np.square(A)
        return (A, B)

    @staticmethod
    def _func(I, params):
        alpha = params['alpha']
        a = params['a']
        b = params['b']

        A = I[0]
        B = I[1]
        C = 2 / 3 * B * A
        D = B + A / 2 + 1 / 12
        F = np.where(A > 0.5, A,
                     np.where(A > 0, D - C,
                              np.where(A > a, D + C, alpha * A + b)))
        return F

    @staticmethod
    def _grad(I, params):
        alpha = params['alpha']
        a = params['a']

        A = I[0]
        B = I[1]
        C = 2 * B
        D = 2 * A + 1 / 2
        G = np.where(A > 0.5, 1,
                     np.where(A > 0, D - C,
                              np.where(A > a, D + C, alpha)))
        return G


class SmoothReLU3(Base):
    def __init__(self, offset=0):
        self.params = {
            'offset': offset,
        }

    @staticmethod
    def _intr(X, params):
        offset = params['offset']

        A = X + offset + 1
        B = 1 + np.exp(4 * (A - 1))
        C = np.square(A)
        return (A, B, C)

    @staticmethod
    def _func(I, _):
        A = I[0]
        B = I[1]
        C = I[2]
        F = np.where(A > 1, np.log(B / 2) / 4 + 1 / 6,
                     np.where(A > 0, C * A / 6, 0))
        return F

    @staticmethod
    def _grad(I, _):
        A = I[0]
        B = I[1]
        C = I[2]
        G = np.where(A > 1, 1 - 1 / B,
                     np.where(A > 0, C / 2, 0))
        return G


class LeakySmoothReLU3:
    def __init__(self, offset=0, alpha=1e-2):
        assert 0 <= alpha <= 0.5

        self.offset = offset
        self.alpha = alpha
        self.a = np.sqrt(2 * alpha) - 1
        self.b = alpha - 2 / 3 * np.sqrt(2) * alpha ** (3 / 2)

    def func(self, X):
        Xo = X + self.offset
        return np.where(Xo > 0, np.log((1 + np.exp(4 * Xo)) / 2) / 4 + 1 / 6,
                        np.where(Xo > self.a, np.power(Xo + 1, 3) / 6, self.alpha * Xo + self.b))

    def grad(self, X):
        Xo = X + self.offset
        return np.where(Xo > 0, 1 - 1 / (1 + np.exp(4 * Xo)),
                        np.where(Xo > self.a, np.square(Xo + 1) / 2, self.alpha))


class Logistic:
    def __init__(self, offset=0):
        self.offset = offset

    def func(self, X):
        Xo = 4 * (X + self.offset)
        Y = np.exp(Xo)
        return 0.25 * np.where(np.isposinf(Y), Xo, np.log1p(Y))

    def grad(self, X):
        Xo = 4 * (X + self.offset)
        Y = np.exp(Xo)
        return np.where(np.isposinf(Y), 1, 1 - 1 / (1 + Y))


class LeakyLogistic:
    def __init__(self, offset=0, alpha=1e-2):
        assert 0 <= alpha < 1

        self.offset = offset
        self.alpha = alpha
        self.a = -np.inf if alpha == 0 else np.log(alpha / (1 - alpha))
        self.b = 0 if alpha == 0 else np.log(1 / (1 - alpha)) - alpha * np.log(alpha / (1 - alpha))

    def func(self, X):
        Xo = 4 * (X + self.offset)
        Y = np.exp(Xo)
        return 0.25 * np.where(Xo > self.a,
                               np.where(np.isposinf(Y), Xo, np.log1p(Y)), self.alpha * Xo + self.b)

    def grad(self, X):
        Xo = 4 * (X + self.offset)
        Y = np.exp(Xo)
        return np.where(Xo > self.a,
                        np.where(np.isposinf(Y), 1, 1 - 1 / (1 + Y)), self.alpha)


class Softplus:
    def __init__(self, offset=0):
        self.offset = offset

    def func(self, X):
        Xo = X + self.offset
        Y = np.exp(Xo)
        return np.where(np.isposinf(Y), Xo, np.log1p(Y))

    def grad(self, X):
        Xo = X + self.offset
        Y = np.exp(Xo)
        return np.where(np.isposinf(Y), 1, 1 - 1 / (1 + Y))


class LeakySoftplus:
    def __init__(self, offset=0, alpha=1e-2):
        assert 0 <= alpha < 1

        self.offset = offset
        self.alpha = alpha
        self.a = -np.inf if alpha == 0 else np.log(alpha / (1 - alpha))
        self.b = 0 if alpha == 0 else np.log(1 / (1 - alpha)) - alpha * np.log(alpha / (1 - alpha))

    def func(self, X):
        Xo = X + self.offset
        Y = np.exp(Xo)
        return np.where(Xo > self.a,
                        np.where(np.isposinf(Y), Xo, np.log1p(Y)), self.alpha * Xo + self.b)

    def grad(self, X):
        Xo = X + self.offset
        Y = np.exp(Xo)
        return np.where(Xo > self.a,
                        np.where(np.isposinf(Y), 1, 1 - 1 / (1 + Y)), self.alpha)


class SELU:
    def __init__(self, offset=0):
        self.offset = offset

    def func(self, X):
        Xo = X + self.offset
        return np.where(Xo > 1, Xo, np.exp(Xo - 1))

    def grad(self, X):
        Xo = X + self.offset
        return np.where(Xo > 1, 1, np.exp(Xo - 1))


class LeakySELU:
    def __init__(self, offset=0, alpha=1e-2):
        assert 0 <= alpha <= 1

        self.offset = offset
        self.alpha = alpha
        self.a = -np.inf if alpha == 0 else np.log(alpha) + 1
        self.b = 0 if alpha == 0 else -alpha * np.log(alpha)

    def func(self, X):
        Xo = X + self.offset
        return np.where(Xo > 1, Xo,
                        np.where(Xo > self.a, np.exp(Xo - 1), self.alpha * Xo + self.b))

    def grad(self, X):
        Xo = X + self.offset
        return np.where(Xo > 1, 1,
                        np.where(Xo > self.a, np.exp(Xo - 1), self.alpha))


class Quadratic:
    def __init__(self, offset=0):
        self.offset = offset

    def func(self, X):
        Xo = X + self.offset
        return np.where(Xo > 0, np.square(Xo), 0)

    def grad(self, X):
        Xo = X + self.offset
        return np.where(Xo > 0, 2 * Xo, 0)


class LeakyQuadratic:
    def __init__(self, offset=0, alpha=1e-2):
        self.offset = offset
        self.alpha = alpha
        self.a = 0.5 * alpha
        self.b = -((0.5 * alpha) ** 2)

    def func(self, X):
        Xo = X + self.offset
        return np.where(Xo > self.a, np.square(Xo), self.alpha * Xo + self.b)

    def grad(self, X):
        Xo = X + self.offset
        return np.where(Xo > self.a, 2 * Xo, self.alpha)


class Sigmoid:
    def __init__(self, offset=0):
        self.offset = offset

    def func(self, X):
        return 1 / (1 + np.exp(-4 * (X + self.offset)))

    def grad(self, X):
        Y = 1 / (1 + np.exp(-4 * (X + self.offset)))
        return 4 * Y * (1 - Y)


class LeakySigmoid:
    def __init__(self, offset=0, alpha=1e-2, beta=1e-2):
        assert 0 <= alpha <= 1
        assert 0 <= beta <= 1

        self.offset = offset
        self.alpha = alpha
        self.beta = beta
        self.a = -np.inf if alpha == 0 else np.log((-alpha - 2 * np.sqrt(1 - alpha) + 2) / alpha)
        self.b = 0 if alpha == 0 else 1 - (1 / (1 - alpha / (alpha - 2 * np.sqrt(1 - alpha) - 2)) -
                                           0.25 * alpha * np.log((-alpha + 2 * np.sqrt(1 - alpha) + 2) / alpha))
        self.c = np.inf if beta == 0 else np.log((-beta + 2 * np.sqrt(1 - beta) + 2) / beta)
        self.d = 1 if beta == 0 else (1 / (1 - beta / (beta - 2 * np.sqrt(1 - beta) - 2)) -
                                      0.25 * beta * np.log((-beta + 2 * np.sqrt(1 - beta) + 2) / beta))

    def func(self, X):
        Xo = 4 * (X + self.offset)
        return np.where(Xo > self.c, 0.25 * self.beta * Xo + self.d,
                        np.where(Xo > self.a, 1 / (1 + np.exp(-Xo)), 0.25 * self.alpha * Xo + self.b))

    def grad(self, X):
        Xo = 4 * (X + self.offset)
        Y = 1 / (1 + np.exp(-Xo))
        return np.where(Xo > self.c, self.beta,
                        np.where(Xo > self.a, 4 * Y * (1 - Y), self.alpha))


class SiLU:
    def __init__(self, offset=0):
        self.offset = offset

    def func(self, X):
        Xo = X + self.offset
        return Xo / (1 + np.exp(-Xo))

    def grad(self, X):
        Xo = X + self.offset
        Y = np.exp(-Xo)
        Z = 1 + Y
        return Xo * Y / np.square(Z) + 1 / Z


class GELU:
    def __init__(self, offset=0):
        self.offset = offset

    def func(self, X):
        Xo = X + self.offset
        return Xo * norm.cdf(Xo)

    def grad(self, X):
        Xo = X + self.offset
        return Xo * norm.pdf(Xo) + norm.cdf(Xo)


def get_activation_function(type='smooth_relu2', offset=1, slope=1e-2):
    match type:
        case 'relu':
            return ReLU(offset)
        case 'leaky_relu':
            return LeakyReLU(offset, slope)
        case 'smooth_relu1':
            return SmoothReLU1(offset)
        case 'leaky_smooth_relu1':
            return LeakySmoothReLU1(offset, slope)
        case 'smooth_relu2':
            return SmoothReLU2(offset)
        case 'leaky_smooth_relu2':
            return LeakySmoothReLU2(offset, slope)
        case 'smooth_relu3':
            return SmoothReLU3(offset)
        case 'leaky_smooth_relu3':
            return LeakySmoothReLU3(offset, slope)
        case 'logisic':
            return Logistic(offset)
        case 'leaky_logistic':
            return LeakyLogistic(offset, slope)
        case 'softplus':
            return Softplus(offset)
        case 'leaky_softplus':
            return LeakySoftplus(offset, slope)
        case 'selu':
            return SELU(offset)
        case 'leaky_selu':
            return LeakySELU(offset, slope)
        case 'quadratic':
            return Quadratic(offset)
        case 'leaky_quadratic':
            return LeakyQuadratic(offset, slope)
        case 'sigmoid':
            return Sigmoid(offset)
        case 'leaky_sigmoid':
            return LeakySigmoid(offset, slope, slope)
        case 'silu':
            return SiLU(offset)
        case 'gelu':
            return GELU(offset)
        case _:
            raise ValueError(f"{type} is not a supported activation function")
