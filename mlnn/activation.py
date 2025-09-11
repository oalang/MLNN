from abc import ABC, abstractmethod
from typing import Callable
from functools import partial

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


Intr = Callable[[NDArray], tuple[NDArray, ...]]
FuncIntr = Callable[[tuple[NDArray, ...]], NDArray]
GradIntr = Callable[[tuple[NDArray, ...]], NDArray]


class ActivationBase(ABC):
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


class ReLU(ActivationBase):
    def __init__(self, offset: float = 0.0) -> None:
        self._offset = offset
        super().__init__()

    @staticmethod
    def _intr(X: NDArray, offset: float) -> tuple[NDArray, ...]:
        A = X + offset
        return (A,)

    @staticmethod
    def _func(I: tuple[NDArray, ...]) -> NDArray:
        A = I[0]
        F = np.maximum(A, 0)
        return F

    @staticmethod
    def _grad(I: tuple[NDArray, ...]) -> NDArray:
        A = I[0]
        G = (A > 0).astype(float)
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return self._func

    def _make_grad_intr(self) -> GradIntr:
        return self._grad


class LeakyReLU(ActivationBase):
    def __init__(self, offset: float = 0.0, alpha: float = 1e-2) -> None:
        self._offset = offset
        self._alpha = alpha
        super().__init__()

    @staticmethod
    def _intr(X: NDArray, offset: float) -> tuple[NDArray, ...]:
        A = X + offset
        return (A,)

    @staticmethod
    def _func(I: tuple[NDArray, ...], alpha: float) -> NDArray:
        A = I[0]
        F = np.where(A > 0, A, alpha * A)
        return F

    @staticmethod
    def _grad(I: tuple[NDArray, ...], alpha: float) -> NDArray:
        A = I[0]
        G = np.where(A > 0, 1, alpha)
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return partial(self._func, alpha=self._alpha)

    def _make_grad_intr(self) -> GradIntr:
        return partial(self._grad, alpha=self._alpha)


class SmoothReLU1(ActivationBase):
    def __init__(self, offset: float = 0.0) -> None:
        self._offset = offset
        super().__init__()

    @staticmethod
    def _intr(X: NDArray, offset: float) -> tuple[NDArray, ...]:
        A = X + offset
        B = A + 0.5
        return (A, B)

    @staticmethod
    def _func(I: tuple[NDArray, ...]) -> NDArray:
        A = I[0]
        B = I[1]
        F = np.where(A > 0.5, A,
                     np.where(A > -0.5, np.square(B) / 2, 0))
        return F

    @staticmethod
    def _grad(I: tuple[NDArray, ...]) -> NDArray:
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


class LeakySmoothReLU1(ActivationBase):
    def __init__(self, offset: float = 0.0, alpha: float = 1e-2) -> None:
        assert alpha <= 1
        self._offset = offset
        self._alpha = alpha
        self._a = float(alpha - 0.5)
        self._b = float(0.5 * (alpha - alpha ** 2))
        super().__init__()

    @staticmethod
    def _intr(X: NDArray, offset: float) -> tuple[NDArray, ...]:
        A = X + offset
        B = A + 0.5
        return (A, B)

    @staticmethod
    def _func(I: tuple[NDArray, ...], alpha: float, a: float, b: float) -> NDArray:
        A = I[0]
        B = I[1]
        F = np.where(A > 0.5, A,
                     np.where(A > a, np.square(B) / 2, alpha * A + b))
        return F

    @staticmethod
    def _grad(I: tuple[NDArray, ...], alpha: float, a: float) -> NDArray:
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


class SmoothReLU2(ActivationBase):
    def __init__(self, offset: float = 0.0) -> None:
        self._offset = offset
        super().__init__()

    @staticmethod
    def _intr(X: NDArray, offset: float) -> tuple[NDArray, ...]:
        A = X + offset
        B = np.square(A)
        return (A, B)

    @staticmethod
    def _func(I: tuple[NDArray, ...]) -> NDArray:
        A = I[0]
        B = I[1]
        C = 2 / 3 * B * A
        D = B + A / 2 + 1 / 12
        F = np.where(A > 0.5, A,
                     np.where(A > 0, D - C,
                              np.where(A > -0.5, D + C, 0)))
        return F

    @staticmethod
    def _grad(I: tuple[NDArray, ...]) -> NDArray:
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


class LeakySmoothReLU2(ActivationBase):
    def __init__(self, offset: float = 0.0, alpha: float = 1e-2) -> None:
        assert 0 <= alpha <= 0.5
        self._offset = offset
        self._alpha = alpha
        self._a = float(np.sqrt(0.5 * alpha) - 0.5)
        self._b = float(0.5 * alpha - np.sqrt(2) / 3 * alpha ** (3 / 2))
        super().__init__()

    @staticmethod
    def _intr(X: NDArray, offset: float) -> tuple[NDArray, ...]:
        A = X + offset
        B = np.square(A)
        return (A, B)

    @staticmethod
    def _func(I: tuple[NDArray, ...], alpha: float, a: float, b: float) -> NDArray:
        A = I[0]
        B = I[1]
        C = 2 / 3 * B * A
        D = B + A / 2 + 1 / 12
        F = np.where(A > 0.5, A,
                     np.where(A > 0, D - C,
                              np.where(A > a, D + C, alpha * A + b)))
        return F

    @staticmethod
    def _grad(I: tuple[NDArray, ...], alpha: float, a: float) -> NDArray:
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


class SmoothReLU3(ActivationBase):
    def __init__(self, offset: float = 0.0) -> None:
        self._offset = offset
        super().__init__()

    @staticmethod
    def _intr(X: NDArray, offset: float) -> tuple[NDArray, ...]:
        A = X + offset
        B = 1 + np.exp(4 * A)
        return (A, B)

    @staticmethod
    def _func(I: tuple[NDArray, ...]) -> NDArray:
        A = I[0]
        B = I[1]
        F = np.where(A > 0, np.log(B / 2) / 4 + 1 / 6,
                     np.where(A > -1, np.power(A + 1, 3) / 6, 0))
        return F

    @staticmethod
    def _grad(I: tuple[NDArray, ...]) -> NDArray:
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


class LeakySmoothReLU3(ActivationBase):
    def __init__(self, offset: float = 0.0, alpha: float = 1e-2) -> None:
        assert 0 <= alpha <= 0.5
        self._offset = offset
        self._alpha = alpha
        self._a = float(np.sqrt(2 * alpha) - 1)
        self._b = float(alpha - 2 / 3 * np.sqrt(2) * alpha ** (3 / 2))
        super().__init__()

    @staticmethod
    def _intr(X: NDArray, offset: float) -> tuple[NDArray, ...]:
        A = X + offset
        B = 1 + np.exp(4 * A)
        return (A, B)

    @staticmethod
    def _func(I: tuple[NDArray, ...], alpha: float, a: float, b: float) -> NDArray:
        A = I[0]
        B = I[1]
        F = np.where(A > 0, np.log(B / 2) / 4 + 1 / 6,
                     np.where(A > a, np.power(A + 1, 3) / 6, alpha * A + b))
        return F

    @staticmethod
    def _grad(I: tuple[NDArray, ...], alpha: float, a: float) -> NDArray:
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


class Logistic(ActivationBase):
    def __init__(self, offset: float = 0.0) -> None:
        self._offset = offset
        super().__init__()

    @staticmethod
    def _intr(X: NDArray, offset: float) -> tuple[NDArray, ...]:
        A = 4 * (X + offset)
        B = np.exp(A)
        return (A, B)

    @staticmethod
    def _func(I: tuple[NDArray, ...]) -> NDArray:
        A = I[0]
        B = I[1]
        F = np.where(np.isposinf(B), A, np.log1p(B)) / 4
        return F

    @staticmethod
    def _grad(I: tuple[NDArray, ...]) -> NDArray:
        B = I[1]
        G = np.where(np.isposinf(B), 1, 1 - 1 / (1 + B))
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return self._func

    def _make_grad_intr(self) -> GradIntr:
        return self._grad


class LeakyLogistic(ActivationBase):
    def __init__(self, offset: float = 0.0, alpha: float = 1e-2) -> None:
        assert 0 <= alpha < 1
        self._offset = offset
        self._alpha = alpha
        self._a = float('-inf') if alpha == 0 else float(np.log(alpha / (1 - alpha)))
        self._b = 0.0 if alpha == 0 else float(np.log(1 / (1 - alpha)) - alpha * np.log(alpha / (1 - alpha)))
        super().__init__()

    @staticmethod
    def _intr(X: NDArray, offset: float) -> tuple[NDArray, ...]:
        A = 4 * (X + offset)
        B = np.exp(A)
        return (A, B)

    @staticmethod
    def _func(I: tuple[NDArray, ...], alpha: float, a: float, b: float) -> NDArray:
        A = I[0]
        B = I[1]
        F = np.where(A > a, np.where(np.isposinf(B), A, np.log1p(B)), alpha * A + b) / 4
        return F

    @staticmethod
    def _grad(I: tuple[NDArray, ...], alpha: float, a: float) -> NDArray:
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


class Softplus(ActivationBase):
    def __init__(self, offset: float = 0.0) -> None:
        self._offset = offset
        super().__init__()

    @staticmethod
    def _intr(X: NDArray, offset: float) -> tuple[NDArray, ...]:
        A = X + offset
        B = np.exp(A)
        return (A, B)

    @staticmethod
    def _func(I: tuple[NDArray, ...]) -> NDArray:
        A = I[0]
        B = I[1]
        F = np.where(np.isposinf(B), A, np.log1p(B))
        return F

    @staticmethod
    def _grad(I: tuple[NDArray, ...]) -> NDArray:
        B = I[1]
        G = np.where(np.isposinf(B), 1, 1 - 1 / (1 + B))
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return self._func

    def _make_grad_intr(self) -> GradIntr:
        return self._grad


class LeakySoftplus(ActivationBase):
    def __init__(self, offset: float = 0.0, alpha: float = 1e-2) -> None:
        assert 0 <= alpha < 1
        self._offset = offset
        self._alpha = alpha
        self._a = float('-inf') if alpha == 0 else float(np.log(alpha / (1 - alpha)))
        self._b = 0.0 if alpha == 0 else float(np.log(1 / (1 - alpha)) - alpha * np.log(alpha / (1 - alpha)))
        super().__init__()

    @staticmethod
    def _intr(X: NDArray, offset: float) -> tuple[NDArray, ...]:
        A = X + offset
        B = np.exp(A)
        return (A, B)

    @staticmethod
    def _func(I: tuple[NDArray, ...], alpha: float, a: float, b: float) -> NDArray:
        A = I[0]
        B = I[1]
        F = np.where(A > a, np.where(np.isposinf(B), A, np.log1p(B)), alpha * A + b)
        return F

    @staticmethod
    def _grad(I: tuple[NDArray, ...], alpha: float, a: float) -> NDArray:
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


class SELU(ActivationBase):
    def __init__(self, offset: float = 0.0) -> None:
        self._offset = offset
        super().__init__()

    @staticmethod
    def _intr(X: NDArray, offset: float) -> tuple[NDArray, ...]:
        A = X + offset
        B = np.exp(A - 1)
        return (A, B)

    @staticmethod
    def _func(I: tuple[NDArray, ...]) -> NDArray:
        A = I[0]
        B = I[1]
        F = np.where(A > 1, A, B)
        return F

    @staticmethod
    def _grad(I: tuple[NDArray, ...]) -> NDArray:
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


class LeakySELU(ActivationBase):
    def __init__(self, offset: float = 0.0, alpha: float = 1e-2) -> None:
        assert 0 <= alpha <= 1
        self._offset = offset
        self._alpha = alpha
        self._a = float('-inf') if alpha == 0 else float(np.log(alpha) + 1)
        self._b = 0.0 if alpha == 0 else float(-alpha * np.log(alpha))
        super().__init__()

    @staticmethod
    def _intr(X: NDArray, offset: float) -> tuple[NDArray, ...]:
        A = X + offset
        B = np.exp(A - 1)
        return (A, B)

    @staticmethod
    def _func(I: tuple[NDArray, ...], alpha: float, a: float, b: float) -> NDArray:
        A = I[0]
        B = I[1]
        F = np.where(A > 1, A,
                     np.where(A > a, B, alpha * A + b))
        return F

    @staticmethod
    def _grad(I: tuple[NDArray, ...], alpha: float, a: float) -> NDArray:
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


class Quadratic(ActivationBase):
    def __init__(self, offset: float = 0.0) -> None:
        self._offset = offset
        super().__init__()

    @staticmethod
    def _intr(X: NDArray, offset: float) -> tuple[NDArray, ...]:
        A = X + offset
        return (A,)

    @staticmethod
    def _func(I: tuple[NDArray, ...]) -> NDArray:
        A = I[0]
        F = np.where(A > 0, np.square(A), 0)
        return F

    @staticmethod
    def _grad(I: tuple[NDArray, ...]) -> NDArray:
        A = I[0]
        G = np.where(A > 0, 2 * A, 0)
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return self._func

    def _make_grad_intr(self) -> GradIntr:
        return self._grad


class LeakyQuadratic(ActivationBase):
    def __init__(self, offset: float = 0.0, alpha: float = 1e-2) -> None:
        self._offset = offset
        self._alpha = alpha
        self._a = float(0.5 * alpha)
        self._b = float(-((0.5 * alpha) ** 2))
        super().__init__()

    @staticmethod
    def _intr(X: NDArray, offset: float) -> tuple[NDArray, ...]:
        A = X + offset
        return (A,)

    @staticmethod
    def _func(I: tuple[NDArray, ...], alpha: float, a: float, b: float) -> NDArray:
        A = I[0]
        F = np.where(A > a, np.square(A), alpha * A + b)
        return F

    @staticmethod
    def _grad(I: tuple[NDArray, ...], alpha: float, a: float) -> NDArray:
        A = I[0]
        G = np.where(A > a, 2 * A, alpha)
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return partial(self._func, alpha=self._alpha, a=self._a, b=self._b)

    def _make_grad_intr(self) -> GradIntr:
        return partial(self._grad, alpha=self._alpha, a=self._a)


class Sigmoid(ActivationBase):
    def __init__(self, offset: float = 0.0) -> None:
        self._offset = offset
        super().__init__()

    @staticmethod
    def _intr(X: NDArray, offset: float) -> tuple[NDArray, ...]:
        B = 1 / (1 + np.exp(-4 * (X + offset)))
        return (B,)

    @staticmethod
    def _func(I: tuple[NDArray, ...]) -> NDArray:
        B = I[0]
        F = B
        return F

    @staticmethod
    def _grad(I: tuple[NDArray, ...]) -> NDArray:
        B = I[0]
        G = 4 * (B - np.square(B))
        return G

    def _make_intr(self) -> Intr:
        return partial(self._intr, offset=self._offset)

    def _make_func_intr(self) -> FuncIntr:
        return self._func

    def _make_grad_intr(self) -> GradIntr:
        return self._grad


class LeakySigmoid(ActivationBase):
    def __init__(self, offset: float = 0.0, alpha: float = 1e-2, beta: float = 1e-2) -> None:
        assert 0 <= alpha <= 1
        assert 0 <= beta <= 1
        self._offset = offset
        self._alpha = alpha
        self._beta = beta
        self._a = float('-inf') if alpha == 0 else float(np.log((-alpha - 2 * np.sqrt(1 - alpha) + 2) / alpha))
        self._b = 0.0 if alpha == 0 else float(1 - (1 / (1 - alpha / (alpha - 2 * np.sqrt(1 - alpha) - 2)) -
                                                    0.25 * alpha * np.log((-alpha + 2 * np.sqrt(1 - alpha) + 2) / alpha)))
        self._c = float('inf') if beta == 0 else float(np.log((-beta + 2 * np.sqrt(1 - beta) + 2) / beta))
        self._d = 1.0 if beta == 0 else float(1 / (1 - beta / (beta - 2 * np.sqrt(1 - beta) - 2)) -
                                              0.25 * beta * np.log((-beta + 2 * np.sqrt(1 - beta) + 2) / beta))
        super().__init__()

    @staticmethod
    def _intr(X: NDArray, offset: float) -> tuple[NDArray, ...]:
        A = 4 * (X + offset)
        B = 1 / (1 + np.exp(np.negative(A)))
        return (A, B)

    @staticmethod
    def _func(I: tuple[NDArray, ...], alpha: float, beta: float, a: float, b: float, c: float, d: float) -> NDArray:
        A = I[0]
        B = I[1]
        F = np.where(A > c, beta / 4 * A + d,
                     np.where(A > a, B, alpha / 4 * A + b))
        return F

    @staticmethod
    def _grad(I: tuple[NDArray, ...], alpha: float, beta: float, a: float, c: float) -> NDArray:
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


class SiLU(ActivationBase):
    def __init__(self, offset: float = 0.0) -> None:
        self._offset = offset
        super().__init__()

    @staticmethod
    def _intr(X: NDArray, offset: float) -> tuple[NDArray, ...]:
        A = X + offset
        B = 1 + np.exp(-A)
        return (A, B)

    @staticmethod
    def _func(I: tuple[NDArray, ...]) -> NDArray:
        A = I[0]
        B = I[1]
        F = A / B
        return F

    @staticmethod
    def _grad(I: tuple[NDArray, ...]) -> NDArray:
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


class GELU(ActivationBase):
    def __init__(self, offset: float = 0.0) -> None:
        self._offset = offset
        super().__init__()

    @staticmethod
    def _intr(X: NDArray, offset: float) -> tuple[NDArray, ...]:
        A = X + offset
        B = norm.cdf(A)
        return (A, B)

    @staticmethod
    def _func(I: tuple[NDArray, ...]) -> NDArray:
        A = I[0]
        B = I[1]
        F = A * B
        return F

    @staticmethod
    def _grad(I: tuple[NDArray, ...]) -> NDArray:
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


def get_activation_function(type: str = 'smooth_relu2', offset: float = 1.0, slope: float = 1e-2) -> ActivationBase:
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
