from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


Intr = Callable[[NDArray], tuple[NDArray, ...]]
FuncIntr = Callable[[tuple[NDArray, ...]], NDArray]
GradIntr = Callable[[tuple[NDArray, ...]], NDArray]


class Base(ABC):
    intr: Intr
    func_intr: FuncIntr
    grad_intr: GradIntr

    def __init__(self) -> None:
        self.intr = self._make_intr()
        self.func_intr = self._make_func_intr()
        self.grad_intr = self._make_grad_intr()

    def func(self, X: NDArray) -> NDArray:
        I = self.intr(X)
        F = self.func_intr(I)
        return F

    def grad(self, X: NDArray) -> NDArray:
        I = self.intr(X)
        G = self.grad_intr(I)
        return G

    @abstractmethod
    def _make_intr(self) -> Intr:
        ...

    @abstractmethod
    def _make_func_intr(self) -> FuncIntr:
        ...

    @abstractmethod
    def _make_grad_intr(self) -> GradIntr:
        ...


class ReLU(Base):
    def __init__(self, offset=0):
        self._offset = offset
        super().__init__()

    @staticmethod
    def _intr(X, offset):
        A = X + offset
        return (A,)

    @staticmethod
    def _func(I):
        A = I[0]
        F = np.maximum(A, 0)
        return F

    @staticmethod
    def _grad(I):
        A = I[0]
        G = (A > 0).astype(float)
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return self._func

    def _make_grad_intr(self) -> GradIntr:
        return self._grad


class LeakyReLU(Base):
    def __init__(self, offset=0, alpha=1e-2):
        self._offset = offset
        self._alpha = alpha
        super().__init__()

    @staticmethod
    def _intr(X, offset):
        A = X + offset
        return (A,)

    @staticmethod
    def _func(I, alpha):
        A = I[0]
        F = np.where(A > 0, A, alpha * A)
        return F

    @staticmethod
    def _grad(I, alpha):
        A = I[0]
        G = np.where(A > 0, 1, alpha)
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return partial(self._func, alpha=self._alpha)

    def _make_grad_intr(self) -> GradIntr:
        return partial(self._grad, alpha=self._alpha)


class SmoothReLU1(Base):
    def __init__(self, offset=0):
        self._offset = offset
        super().__init__()

    @staticmethod
    def _intr(X, offset):
        A = X + offset
        B = A + 0.5
        return (A, B)

    @staticmethod
    def _func(I):
        A = I[0]
        B = I[1]
        F = np.where(A > 0.5, A,
                     np.where(A > -0.5, np.square(B) / 2, 0))
        return F

    @staticmethod
    def _grad(I):
        A = I[0]
        B = I[1]
        G = np.where(A > 0.5, 1,
                     np.where(A > -0.5, B, 0))
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return self._func

    def _make_grad_intr(self) -> GradIntr:
        return self._grad


class LeakySmoothReLU1(Base):
    def __init__(self, offset=0, alpha=1e-2):
        assert alpha <= 1
        self._offset = offset
        self._alpha = alpha
        self._a = alpha - 0.5
        self._b = 0.5 * (alpha - alpha ** 2)
        super().__init__()

    @staticmethod
    def _intr(X, offset):
        A = X + offset
        B = A + 0.5
        return (A, B)

    @staticmethod
    def _func(I, alpha, a, b):
        A = I[0]
        B = I[1]
        F = np.where(A > 0.5, A,
                     np.where(A > a, np.square(B) / 2, alpha * A + b))
        return F

    @staticmethod
    def _grad(I, alpha, a):
        A = I[0]
        B = I[1]
        G = np.where(A > 0.5, 1,
                     np.where(A > a, B, alpha))
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return partial(self._func, alpha=self._alpha, a=self._a, b=self._b)

    def _make_grad_intr(self) -> GradIntr:
        return partial(self._grad, alpha=self._alpha, a=self._a)


class SmoothReLU2(Base):
    def __init__(self, offset=0):
        self._offset = offset
        super().__init__()

    @staticmethod
    def _intr(X, offset):
        A = X + offset
        B = np.square(A)
        return (A, B)

    @staticmethod
    def _func(I):
        A = I[0]
        B = I[1]
        C = 2 / 3 * B * A
        D = B + A / 2 + 1 / 12
        F = np.where(A > 0.5, A,
                     np.where(A > 0, D - C,
                              np.where(A > -0.5, D + C, 0)))
        return F

    @staticmethod
    def _grad(I):
        A = I[0]
        B = I[1]
        C = 2 * B
        D = 2 * A + 1 / 2
        G = np.where(A > 0.5, 1,
                     np.where(A > 0, D - C,
                              np.where(A > -0.5, D + C, 0)))
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return self._func

    def _make_grad_intr(self) -> GradIntr:
        return self._grad


class LeakySmoothReLU2(Base):
    def __init__(self, offset=0, alpha=1e-2):
        assert 0 <= alpha <= 0.5
        self._offset = offset
        self._alpha = alpha
        self._a = np.sqrt(0.5 * alpha) - 0.5
        self._b = 0.5 * alpha - np.sqrt(2) / 3 * alpha ** (3 / 2)
        super().__init__()

    @staticmethod
    def _intr(X, offset):
        A = X + offset
        B = np.square(A)
        return (A, B)

    @staticmethod
    def _func(I, alpha, a, b):
        A = I[0]
        B = I[1]
        C = 2 / 3 * B * A
        D = B + A / 2 + 1 / 12
        F = np.where(A > 0.5, A,
                     np.where(A > 0, D - C,
                              np.where(A > a, D + C, alpha * A + b)))
        return F

    @staticmethod
    def _grad(I, alpha, a):
        A = I[0]
        B = I[1]
        C = 2 * B
        D = 2 * A + 1 / 2
        G = np.where(A > 0.5, 1,
                     np.where(A > 0, D - C,
                              np.where(A > a, D + C, alpha)))
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return partial(self._func, alpha=self._alpha, a=self._a, b=self._b)

    def _make_grad_intr(self) -> GradIntr:
        return partial(self._grad, alpha=self._alpha, a=self._a)


class SmoothReLU3(Base):
    def __init__(self, offset=0):
        self._offset = offset
        super().__init__()

    @staticmethod
    def _intr(X, offset):
        A = X + offset
        B = 1 + np.exp(4 * A)
        return (A, B)

    @staticmethod
    def _func(I):
        A = I[0]
        B = I[1]
        F = np.where(A > 0, np.log(B / 2) / 4 + 1 / 6,
                     np.where(A > -1, np.power(A + 1, 3) / 6, 0))
        return F

    @staticmethod
    def _grad(I):
        A = I[0]
        B = I[1]
        G = np.where(A > 0, 1 - 1 / B,
                     np.where(A > -1, np.square(A + 1) / 2, 0))
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return self._func

    def _make_grad_intr(self) -> GradIntr:
        return self._grad


class LeakySmoothReLU3(Base):
    def __init__(self, offset=0, alpha=1e-2):
        assert 0 <= alpha <= 0.5
        self._offset = offset
        self._alpha = alpha
        self._a = np.sqrt(2 * alpha) - 1
        self._b = alpha - 2 / 3 * np.sqrt(2) * alpha ** (3 / 2)
        super().__init__()

    @staticmethod
    def _intr(X, offset):
        A = X + offset
        B = 1 + np.exp(4 * A)
        return (A, B)

    @staticmethod
    def _func(I, alpha, a, b):
        A = I[0]
        B = I[1]
        F = np.where(A > 0, np.log(B / 2) / 4 + 1 / 6,
                     np.where(A > a, np.power(A + 1, 3) / 6, alpha * A + b))
        return F

    @staticmethod
    def _grad(I, alpha, a):
        A = I[0]
        B = I[1]
        G = np.where(A > 0, 1 - 1 / B,
                     np.where(A > a, np.square(A + 1) / 2, alpha))
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return partial(self._func, alpha=self._alpha, a=self._a, b=self._b)

    def _make_grad_intr(self) -> GradIntr:
        return partial(self._grad, alpha=self._alpha, a=self._a)


class Logistic(Base):
    def __init__(self, offset=0):
        self._offset = offset
        super().__init__()

    @staticmethod
    def _intr(X, offset):
        A = 4 * (X + offset)
        B = np.exp(A)
        return (A, B)

    @staticmethod
    def _func(I):
        A = I[0]
        B = I[1]
        F = np.where(np.isposinf(B), A, np.log1p(B)) / 4
        return F

    @staticmethod
    def _grad(I):
        B = I[1]
        G = np.where(np.isposinf(B), 1, 1 - 1 / (1 + B))
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return self._func

    def _make_grad_intr(self) -> GradIntr:
        return self._grad


class LeakyLogistic(Base):
    def __init__(self, offset=0, alpha=1e-2):
        assert 0 <= alpha < 1
        self._offset = offset
        self._alpha = alpha
        self._a = -np.inf if alpha == 0 else np.log(alpha / (1 - alpha))
        self._b = 0 if alpha == 0 else np.log(1 / (1 - alpha)) - alpha * np.log(alpha / (1 - alpha))
        super().__init__()

    @staticmethod
    def _intr(X, offset):
        A = 4 * (X + offset)
        B = np.exp(A)
        return (A, B)

    @staticmethod
    def _func(I, alpha, a, b):
        A = I[0]
        B = I[1]
        F = np.where(A > a, np.where(np.isposinf(B), A, np.log1p(B)), alpha * A + b) / 4
        return F

    @staticmethod
    def _grad(I, alpha, a):
        A = I[0]
        B = I[1]
        G = np.where(A > a, np.where(np.isposinf(B), 1, 1 - 1 / (1 + B)), alpha)
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return partial(self._func, alpha=self._alpha, a=self._a, b=self._b)

    def _make_grad_intr(self) -> GradIntr:
        return partial(self._grad, alpha=self._alpha, a=self._a)


class Softplus(Base):
    def __init__(self, offset=0):
        self._offset = offset
        super().__init__()

    @staticmethod
    def _intr(X, offset):
        A = X + offset
        B = np.exp(A)
        return (A, B)

    @staticmethod
    def _func(I):
        A = I[0]
        B = I[1]
        F = np.where(np.isposinf(B), A, np.log1p(B))
        return F

    @staticmethod
    def _grad(I):
        B = I[1]
        G = np.where(np.isposinf(B), 1, 1 - 1 / (1 + B))
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return self._func

    def _make_grad_intr(self) -> GradIntr:
        return self._grad


class LeakySoftplus(Base):
    def __init__(self, offset=0, alpha=1e-2):
        assert 0 <= alpha < 1
        self._offset = offset
        self._alpha = alpha
        self._a = -np.inf if alpha == 0 else np.log(alpha / (1 - alpha))
        self._b = 0 if alpha == 0 else np.log(1 / (1 - alpha)) - alpha * np.log(alpha / (1 - alpha))
        super().__init__()

    @staticmethod
    def _intr(X, offset):
        A = X + offset
        B = np.exp(A)
        return (A, B)

    @staticmethod
    def _func(I, alpha, a, b):
        A = I[0]
        B = I[1]
        F = np.where(A > a, np.where(np.isposinf(B), A, np.log1p(B)), alpha * A + b)
        return F

    @staticmethod
    def _grad(I, alpha, a):
        A = I[0]
        B = I[1]
        G = np.where(A > a, np.where(np.isposinf(B), 1, 1 - 1 / (1 + B)), alpha)
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return partial(self._func, alpha=self._alpha, a=self._a, b=self._b)

    def _make_grad_intr(self) -> GradIntr:
        return partial(self._grad, alpha=self._alpha, a=self._a)


class SELU(Base):
    def __init__(self, offset=0):
        self._offset = offset
        super().__init__()

    @staticmethod
    def _intr(X, offset):
        A = X + offset
        B = np.exp(A - 1)
        return (A, B)

    @staticmethod
    def _func(I):
        A = I[0]
        B = I[1]
        F = np.where(A > 1, A, B)
        return F

    @staticmethod
    def _grad(I):
        A = I[0]
        B = I[1]
        G = np.where(A > 1, 1, B)
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return self._func

    def _make_grad_intr(self) -> GradIntr:
        return self._grad


class LeakySELU(Base):
    def __init__(self, offset=0, alpha=1e-2):
        assert 0 <= alpha <= 1
        self._offset = offset
        self._alpha = alpha
        self._a = -np.inf if alpha == 0 else np.log(alpha) + 1
        self._b = 0 if alpha == 0 else -alpha * np.log(alpha)
        super().__init__()

    @staticmethod
    def _intr(X, offset):
        A = X + offset
        B = np.exp(A - 1)
        return (A, B)

    @staticmethod
    def _func(I, alpha, a, b):
        A = I[0]
        B = I[1]
        F = np.where(A > 1, A,
                     np.where(A > a, B, alpha * A + b))
        return F

    @staticmethod
    def _grad(I, alpha, a):
        A = I[0]
        B = I[1]
        G = np.where(A > 1, 1,
                     np.where(A > a, B, alpha))
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return partial(self._func, alpha=self._alpha, a=self._a, b=self._b)

    def _make_grad_intr(self) -> GradIntr:
        return partial(self._grad, alpha=self._alpha, a=self._a)


class Quadratic(Base):
    def __init__(self, offset=0):
        self._offset = offset
        super().__init__()

    @staticmethod
    def _intr(X, offset):
        A = X + offset
        return (A,)

    @staticmethod
    def _func(I):
        A = I[0]
        F = np.where(A > 0, np.square(A), 0)
        return F

    @staticmethod
    def _grad(I):
        A = I[0]
        G = np.where(A > 0, 2 * A, 0)
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return self._func

    def _make_grad_intr(self) -> GradIntr:
        return self._grad


class LeakyQuadratic(Base):
    def __init__(self, offset=0, alpha=1e-2):
        self._offset = offset
        self._alpha = alpha
        self._a = 0.5 * alpha
        self._b = -((0.5 * alpha) ** 2)
        super().__init__()

    @staticmethod
    def _intr(X, offset):
        A = X + offset
        return (A,)

    @staticmethod
    def _func(I, alpha, a, b):
        A = I[0]
        F = np.where(A > a, np.square(A), alpha * A + b)
        return F

    @staticmethod
    def _grad(I, alpha, a):
        A = I[0]
        G = np.where(A > a, 2 * A, alpha)
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return partial(self._func, alpha=self._alpha, a=self._a, b=self._b)

    def _make_grad_intr(self) -> GradIntr:
        return partial(self._grad, alpha=self._alpha, a=self._a)


class Sigmoid(Base):
    def __init__(self, offset=0):
        self._offset = offset
        super().__init__()

    @staticmethod
    def _intr(X, offset):
        B = 1 / (1 + np.exp(-4 * (X + offset)))
        return (B,)

    @staticmethod
    def _func(I):
        B = I[0]
        F = B
        return F

    @staticmethod
    def _grad(I):
        B = I[0]
        G = 4 * (B - np.square(B))
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return self._func

    def _make_grad_intr(self) -> GradIntr:
        return self._grad


class LeakySigmoid(Base):
    def __init__(self, offset=0, alpha=1e-2, beta=1e-2):
        assert 0 <= alpha <= 1
        assert 0 <= beta <= 1
        self._offset = offset
        self._alpha = alpha
        self._beta = beta
        self._a = -np.inf if alpha == 0 else np.log((-alpha - 2 * np.sqrt(1 - alpha) + 2) / alpha)
        self._b = 0 if alpha == 0 else 1 - (1 / (1 - alpha / (alpha - 2 * np.sqrt(1 - alpha) - 2)) -
                                            0.25 * alpha * np.log((-alpha + 2 * np.sqrt(1 - alpha) + 2) / alpha))
        self._c = np.inf if beta == 0 else np.log((-beta + 2 * np.sqrt(1 - beta) + 2) / beta)
        self._d = 1 if beta == 0 else (1 / (1 - beta / (beta - 2 * np.sqrt(1 - beta) - 2)) -
                                       0.25 * beta * np.log((-beta + 2 * np.sqrt(1 - beta) + 2) / beta))
        super().__init__()

    @staticmethod
    def _intr(X, offset):
        A = 4 * (X + offset)
        B = 1 / (1 + np.exp(np.negative(A)))
        return (A, B)

    @staticmethod
    def _func(I, alpha, beta, a, b, c, d):
        A = I[0]
        B = I[1]
        F = np.where(A > c, beta / 4 * A + d,
                     np.where(A > a, B, alpha / 4 * A + b))
        return F

    @staticmethod
    def _grad(I, alpha, beta, a, c):
        A = I[0]
        B = I[1]
        G = np.where(A > c, beta,
                     np.where(A > a, 4 * (B - np.square(B)), alpha))
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return partial(self._func, alpha=self._alpha, beta=self._beta, a=self._a, b=self._b, c=self._c, d=self._d)

    def _make_grad_intr(self) -> GradIntr:
        return partial(self._grad, alpha=self._alpha, beta=self._beta, a=self._a, c=self._c)


class SiLU(Base):
    def __init__(self, offset=0):
        self._offset = offset
        super().__init__()

    @staticmethod
    def _intr(X, offset):
        A = X + offset
        B = 1 + np.exp(-A)
        return (A, B)

    @staticmethod
    def _func(I):
        A = I[0]
        B = I[1]
        F = A / B
        return F

    @staticmethod
    def _grad(I):
        A = I[0]
        B = I[1]
        G = A * (B - 1) / np.square(B) + 1 / B
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return self._func

    def _make_grad_intr(self) -> GradIntr:
        return self._grad


class GELU(Base):
    def __init__(self, offset=0):
        self._offset = offset
        super().__init__()

    @staticmethod
    def _intr(X, offset):
        A = X + offset
        B = norm.cdf(A)
        return (A, B)

    @staticmethod
    def _func(I):
        A = I[0]
        B = I[1]
        F = A * B
        return F

    @staticmethod
    def _grad(I):
        A = I[0]
        B = I[1]
        G = A * norm.pdf(A) + B
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return self._func

    def _make_grad_intr(self) -> GradIntr:
        return self._grad


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
