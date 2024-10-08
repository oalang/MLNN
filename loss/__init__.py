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


class SmoothReLU:
    def __init__(self, offset=0):
        self.offset = offset

    def func(self, X):
        Xo = X + self.offset
        return np.where(Xo > .5, Xo, np.where(Xo > -.5, .5 * (Xo + .5) ** 2, 0))

    def grad(self, X):
        Xo = X + self.offset
        return np.where(Xo > .5, 1, np.where(Xo > -.5, Xo + .5, 0))


class LeakySmoothReLU:
    def __init__(self, offset=0, alpha=1e-2):
        assert alpha <= 1

        self.offset = offset
        self.alpha = alpha
        self.a = alpha - .5
        self.b = .5 * (alpha - alpha ** 2)

    def func(self, X):
        Xo = X + self.offset
        return np.where(Xo > .5, Xo, np.where(Xo > self.a, .5 * (Xo + .5) ** 2, self.alpha * Xo + self.b))

    def grad(self, X):
        Xo = X + self.offset
        return np.where(Xo > .5, 1, np.where(Xo > self.a, Xo + .5, self.alpha))


class SmoothReLU2:
    def __init__(self, offset=0):
        self.offset = offset

    def func(self, X):
        Xo = X + self.offset
        return np.where(Xo > .5, Xo, np.where(Xo > 0, -2 / 3 * (Xo + .5) ** 3 + 2 * (Xo + .5) ** 2 - Xo - 1 / 3,
                                              np.where(Xo > -.5, 2 / 3 * (Xo + .5) ** 3, 0)))

    def grad(self, X):
        Xo = X + self.offset
        return np.where(Xo > .5, 1, np.where(Xo > 0, 1 - 2 * (Xo - .5) ** 2,
                                             np.where(Xo > -.5, 2 * (Xo + .5) ** 2, 0)))


class LeakySmoothReLU2:
    def __init__(self, offset=0, alpha=1e-2):
        assert 0 <= alpha <= .5

        self.offset = offset
        self.alpha = alpha
        self.a = (.5 * alpha) ** .5 - .5
        self.b = .5 * alpha - 2 ** .5 / 3 * alpha ** 1.5

    def func(self, X):
        Xo = X + self.offset
        return np.where(Xo > .5, Xo, np.where(Xo > 0, -2 / 3 * (Xo + .5) ** 3 + 2 * (Xo + .5) ** 2 - Xo - 1 / 3,
                                              np.where(Xo > self.a, 2 / 3 * (Xo + .5) ** 3, self.alpha * Xo + self.b)))

    def grad(self, X):
        Xo = X + self.offset
        return np.where(Xo > .5, 1, np.where(Xo > 0, 1 - 2 * (Xo - .5) ** 2,
                                             np.where(Xo > self.a, 2 * (Xo + .5) ** 2, self.alpha)))


class SmoothReLU3:
    def __init__(self, offset=0):
        self.offset = offset

    def func(self, X):
        Xo = X + self.offset
        return np.where(Xo > 0, np.log((1 + np.exp(4 * Xo)) / 2) / 4 + 1 / 6, np.where(Xo > -1, (Xo + 1) ** 3 / 6, 0))

    def grad(self, X):
        Xo = X + self.offset
        return np.where(Xo > 0, 1 - 1 / (1 + np.exp(4 * Xo)), np.where(Xo > -1, (Xo + 1) ** 2 / 2, 0))


class LeakySmoothReLU3:
    def __init__(self, offset=0, alpha=1e-2):
        assert 0 <= alpha <= .5

        self.offset = offset
        self.alpha = alpha
        self.a = (2 * alpha) ** .5 - 1
        self.b = alpha - 2 / 3 * 2 ** .5 * alpha ** 1.5

    def func(self, X):
        Xo = X + self.offset
        return np.where(Xo > 0, np.log((1 + np.exp(4 * Xo)) / 2) / 4 + 1 / 6,
                        np.where(Xo > self.a, (Xo + 1) ** 3 / 6, self.alpha * Xo + self.b))

    def grad(self, X):
        Xo = X + self.offset
        return np.where(Xo > 0, 1 - 1 / (1 + np.exp(4 * Xo)),
                        np.where(Xo > self.a, (Xo + 1) ** 2 / 2, self.alpha))


class Quadratic:
    def __init__(self, offset=0):
        self.offset = offset

    def func(self, X):
        Xo = X + self.offset
        return np.where(Xo > 0, Xo ** 2, 0)

    def grad(self, X):
        Xo = X + self.offset
        return np.where(Xo > 0, 2 * Xo, 0)


class LeakyQuadratic:
    def __init__(self, offset=0, alpha=1e-2):
        self.offset = offset
        self.alpha = alpha
        self.a = .5 * alpha
        self.b = -((.5 * alpha) ** 2)

    def func(self, X):
        Xo = X + self.offset
        return np.where(Xo > self.a, Xo ** 2, self.alpha * Xo + self.b)

    def grad(self, X):
        Xo = X + self.offset
        return np.where(Xo > self.a, 2 * Xo, self.alpha)


class Logistic:
    def __init__(self, offset=0):
        self.offset = offset

    def func(self, X):
        Xo = 4 * (X + self.offset)
        Y = np.exp(Xo)
        return .25 * np.where(np.isposinf(Y), Xo, np.log1p(Y))

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
        self.b = 0 if alpha == 0 else .25 * (np.log(1 / (1 - alpha)) - alpha * np.log(alpha / (1 - alpha)))

    def func(self, X):
        Xo = 4 * (X + self.offset)
        Y = np.exp(Xo)
        return np.where(Xo > self.a, .25 * np.where(np.isposinf(Y), Xo, np.log1p(Y)), .25 * self.alpha * Xo + self.b)

    def grad(self, X):
        Xo = 4 * (X + self.offset)
        Y = np.exp(Xo)
        return np.where(Xo > self.a, np.where(np.isposinf(Y), 1, 1 - 1 / (1 + Y)), self.alpha)


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
        self.a = -np.inf if alpha == 0 else np.log((-alpha - 2 * (1 - alpha) ** .5 + 2) / alpha)
        self.b = 0 if alpha == 0 else 1 - (1 / (1 - alpha / (alpha - 2 * (1 - alpha) ** .5 - 2)) -
                                           .25 * alpha * np.log((-alpha + 2 * (1 - alpha) ** .5 + 2) / alpha))
        self.c = np.inf if beta == 0 else np.log((-beta + 2 * (1 - beta) ** .5 + 2) / beta)
        self.d = 1 if beta == 0 else (1 / (1 - beta / (beta - 2 * (1 - beta) ** .5 - 2)) -
                                      .25 * beta * np.log((-beta + 2 * (1 - beta) ** .5 + 2) / beta))

    def func(self, X):
        Xo = 4 * (X + self.offset)
        return np.where(Xo > self.c, .25 * self.beta * Xo + self.d,
                        np.where(Xo > self.a, 1 / (1 + np.exp(-Xo)), .25 * self.alpha * Xo + self.b))

    def grad(self, X):
        Xo = 4 * (X + self.offset)
        Y = 1 / (1 + np.exp(-Xo))
        return np.where(Xo > self.c, self.beta, np.where(Xo > self.a, 4 * Y * (1 - Y), self.alpha))


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
        return Xo * Y / Z ** 2 + 1 / Z


class GELU:
    def __init__(self, offset=0):
        self.offset = offset

    def func(self, X):
        Xo = X + self.offset
        return Xo * norm.cdf(Xo)

    def grad(self, X):
        Xo = X + self.offset
        return Xo * norm.pdf(Xo) + norm.cdf(Xo)
