import numpy as np
from scipy.stats import norm


class ReLU:
    def __init__(self, offset=0):
        self.offset = offset

    def func(self, X):
        return np.maximum(X + self.offset, 0)

    def grad(self, X):
        return np.where(X + self.offset > 0, 1, 0)


class LeakyReLU:
    def __init__(self, offset=0, alpha=1e-2):
        self.offset = offset
        self.alpha = alpha

    def func(self, X):
        Xo = X + self.offset
        return np.where(Xo > 0, Xo, self.alpha * Xo)

    def grad(self, X):
        return np.where(X + self.offset > 0, 1, self.alpha)


class SmoothReLU1:
    def __init__(self, offset=0):
        self.offset = offset

    def func(self, X):
        Xo = X + self.offset
        return np.where(Xo > 0.5, Xo,
                        np.where(Xo > -0.5, np.square(Xo + 0.5) / 2, 0))

    def grad(self, X):
        Xo = X + self.offset
        return np.where(Xo > 0.5, 1,
                        np.where(Xo > -0.5, Xo + 0.5, 0))


class LeakySmoothReLU1:
    def __init__(self, offset=0, alpha=1e-2):
        assert alpha <= 1

        self.offset = offset
        self.alpha = alpha
        self.a = alpha - 0.5
        self.b = 0.5 * (alpha - alpha ** 2)

    def func(self, X):
        Xo = X + self.offset
        return np.where(Xo > 0.5, Xo,
                        np.where(Xo > self.a, np.square(Xo + 0.5) / 2, self.alpha * Xo + self.b))

    def grad(self, X):
        Xo = X + self.offset
        return np.where(Xo > 0.5, 1,
                        np.where(Xo > self.a, Xo + 0.5, self.alpha))


class SmoothReLU2:
    def __init__(self, offset=0):
        self.offset = offset

    def func(self, X):
        Xo = X + self.offset
        return np.where(Xo > 0.5, Xo,
                        np.where(Xo > 0, -2 / 3 * np.power(Xo + 0.5, 3) + 2 * np.square(Xo + 0.5) - Xo - 1 / 3,
                                 np.where(Xo > -0.5, 2 / 3 * np.power(Xo + 0.5, 3), 0)))

    def grad(self, X):
        Xo = X + self.offset
        return np.where(Xo > 0.5, 1,
                        np.where(Xo > 0, 1 - 2 * np.square(Xo - 0.5),
                                 np.where(Xo > -0.5, 2 * np.square(Xo + 0.5), 0)))


class LeakySmoothReLU2:
    def __init__(self, offset=0, alpha=1e-2):
        assert 0 <= alpha <= 0.5

        self.offset = offset
        self.alpha = alpha
        self.a = np.sqrt(0.5 * alpha) - 0.5
        self.b = 0.5 * alpha - np.sqrt(2) / 3 * alpha ** (3 / 2)

    def func(self, X):
        Xo = X + self.offset
        return np.where(Xo > 0.5, Xo,
                        np.where(Xo > 0, -2 / 3 * np.power(Xo + 0.5, 3) + 2 * np.square(Xo + 0.5) - Xo - 1 / 3,
                                 np.where(Xo > self.a, 2 / 3 * np.power(Xo + 0.5, 3), self.alpha * Xo + self.b)))

    def grad(self, X):
        Xo = X + self.offset
        return np.where(Xo > 0.5, 1,
                        np.where(Xo > 0, 1 - 2 * np.square(Xo - 0.5),
                                 np.where(Xo > self.a, 2 * np.square(Xo + 0.5), self.alpha)))


class SmoothReLU3:
    def __init__(self, offset=0):
        self.offset = offset

    def func(self, X):
        Xo = X + self.offset
        return np.where(Xo > 0, np.log((1 + np.exp(4 * Xo)) / 2) / 4 + 1 / 6,
                        np.where(Xo > -1, np.power(Xo + 1, 3) / 6, 0))

    def grad(self, X):
        Xo = X + self.offset
        return np.where(Xo > 0, 1 - 1 / (1 + np.exp(4 * Xo)),
                        np.where(Xo > -1, np.square(Xo + 1) / 2, 0))


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
        case 'logistic':
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
