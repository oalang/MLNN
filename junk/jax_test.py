from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

print(jax.__version__)


@partial(jax.jit, static_argnames=['offset'])
def intr_compiled(X, offset):
    A = X + offset
    B = jnp.square(A)
    return (A, B)


def intr(X, params):
    offset = params['offset']

    return intr_compiled(X, offset)


@partial(jax.jit, static_argnames=['alpha', 'a', 'b'])
def func_compiled(I, alpha, a, b):
    A = I[0]
    B = I[1]
    C = 2 / 3 * B * A
    D = B + A / 2 + 1 / 12
    F = jnp.where(A > 0.5, A,
                  jnp.where(A > 0, D - C,
                            jnp.where(A > a, D + C, alpha * A + b)))
    return F


def func(I, params):
    alpha = params['alpha']
    a = params['a']
    b = params['b']

    return func_compiled(I, alpha, a, b)


@partial(jax.jit, static_argnames=['alpha', 'a'])
def grad_compiled(I, alpha, a):
    A = I[0]
    B = I[1]
    C = 2 * B
    D = 2 * A + 1 / 2
    G = jnp.where(A > 0.5, 1,
                  jnp.where(A > 0, D - C,
                            jnp.where(A > a, D + C, alpha)))
    return G


def grad(I, params):
    alpha = params['alpha']
    a = params['a']

    return grad_compiled(I, alpha, a)


offset = 1.0
alpha = 0.1

assert 0 <= alpha <= 0.5

params = {
    'offset': offset,
    'alpha': alpha,
    'a': (jnp.sqrt(0.5 * alpha) - 0.5).item(),
    'b': (0.5 * alpha - jnp.sqrt(2) / 3 * alpha ** (3 / 2)).item(),
}

X = jnp.arange(-3, 3, 0.01)

I = intr(X, params)
F = func(I, params)
G = grad(I, params)

plt.plot(X, F)
plt.plot(X, G)
plt.show()
