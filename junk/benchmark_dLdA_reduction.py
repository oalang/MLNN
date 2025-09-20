import argparse
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from mlnn.engine import compute_dLdA

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    JAX_AVAILABLE = True
except Exception:
    jax = None
    jnp = None
    lax = None
    JAX_AVAILABLE = False


@dataclass
class BenchResult:
    name: str
    ms_per_call: float
    iters: int


def timeit_fn(fn: Callable[[], None], warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def make_numpy_inputs(n: int, m: int, active_frac: float, a_mode: str, l: float = 1.0) -> dict:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, m)).astype(float)
    # Create a symmetric U with sparse activity; sign and magnitude arbitrary
    U = rng.standard_normal((n, n)).astype(float)
    U = (U + U.T) / 2
    # Active set: choose exactly k indices
    k = max(1, int(active_frac * n))
    active_idx = np.sort(rng.choice(n, size=k, replace=False))
    active_rows = np.zeros(n, dtype=bool)
    active_rows[active_idx] = True
    active_cols = active_rows.copy()
    active_data = np.logical_or(active_rows, active_cols)
    any_active = bool(active_data.any())
    A = rng.standard_normal((m, m)).astype(float)
    A = (A + A.T) / 2
    reduce_derivative_matrix = False
    return dict(
        X=X,
        A=A,
        U=U,
        active_data=active_data,
        active_idx=active_idx,
        any_active=any_active,
        l=l,
        a_mode=a_mode,
        reduce_derivative_matrix=reduce_derivative_matrix,
    )


def make_jax_inputs(np_inputs: dict) -> Optional[dict]:
    if not JAX_AVAILABLE:
        return None
    J = {}
    for k, v in np_inputs.items():
        if isinstance(v, np.ndarray):
            J[k] = jnp.array(v)
        else:
            J[k] = v
    return J


def bench_numpy(inputs: dict, iters: int) -> list[BenchResult]:
    X = inputs['X']
    A = inputs['A']
    U = inputs['U']
    active_data = inputs['active_data']
    any_active = inputs['any_active']
    l = inputs['l']
    a_mode = inputs['a_mode']

    def run_no_reduce():
        _ = compute_dLdA(X, A, U, active_data, any_active, l, a_mode, reduce_derivative_matrix=False)

    def run_reduce():
        _ = compute_dLdA(X, A, U, active_data, any_active, l, a_mode, reduce_derivative_matrix=True)

    ms_no = timeit_fn(run_no_reduce, iters=iters)
    ms_red = timeit_fn(run_reduce, iters=iters)
    return [
        BenchResult('numpy/no_reduce', ms_no, iters),
        BenchResult('numpy/reduce', ms_red, iters),
    ]


# JAX equivalents (re-implemented per engine.compute_dLdA logic)
if JAX_AVAILABLE:
    def compute_dLdA_jax(X, A, U, active_data, active_idx, any_active, l, a_mode, reduce_derivative_matrix):
        pred = jnp.asarray(any_active, dtype=bool)

        def zeros_for_mode(A_local):
            if a_mode == 'diagonal':
                return jnp.zeros((A_local.shape[0], 1), dtype=A_local.dtype)
            else:
                return jnp.zeros_like(A_local)

        def true_branch(args):
            X_local, A_local, U_local = args
            if reduce_derivative_matrix:
                U_r = U_local[jnp.ix_(active_idx, active_idx)]
                X_r = X_local[active_idx]
            else:
                U_r = U_local
                X_r = X_local
            W0 = -(U_r + U_r.T)
            W = W0.at[jnp.arange(W0.shape[0]), jnp.arange(W0.shape[0])].add(-jnp.sum(W0, axis=0))
            if a_mode == 'full':
                return l * (X_r.T @ W @ X_r)
            elif a_mode == 'diagonal':
                return l * jnp.sum(X_r.T * (W @ X_r).T, axis=1, keepdims=True)
            elif a_mode == 'decomposed':
                return l * 2 * ((A_local @ X_r.T) @ W @ X_r)
            else:
                return zeros_for_mode(A_local)

        def false_branch(args):
            _, A_local, _ = args
            return zeros_for_mode(A_local)

        return lax.cond(pred, true_branch, false_branch, (X, A, U))


def bench_jax(jinputs: dict, iters: int, jit: bool) -> list[BenchResult]:
    fn = compute_dLdA_jax
    if jit:
        fn = jax.jit(fn, static_argnames=['a_mode', 'reduce_derivative_matrix'])

    def run_no_reduce():
        Y = fn(jinputs['X'], jinputs['A'], jinputs['U'], jinputs['active_data'], jinputs['active_idx'], jinputs['any_active'], jinputs['l'], jinputs['a_mode'], False)
        if jit:
            _ = Y.block_until_ready()

    def run_reduce():
        Y = fn(jinputs['X'], jinputs['A'], jinputs['U'], jinputs['active_data'], jinputs['active_idx'], jinputs['any_active'], jinputs['l'], jinputs['a_mode'], True)
        if jit:
            _ = Y.block_until_ready()

    # Warm compile for JIT
    if jit:
        run_no_reduce()
        run_reduce()

    ms_no = timeit_fn(run_no_reduce, iters=iters)
    ms_red = timeit_fn(run_reduce, iters=iters)
    tag = 'jax-jit' if jit else 'jax'
    return [
        BenchResult(f'{tag}/no_reduce', ms_no, iters),
        BenchResult(f'{tag}/reduce', ms_red, iters),
    ]


def print_results(title: str, results: list[BenchResult]) -> None:
    print(f'\n== {title} ==')
    for r in results:
        print(f'{r.name:28s}  {r.ms_per_call:8.3f} ms  (iters={r.iters})')


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark compute_dLdA with/without reduction.')
    parser.add_argument('--n', type=int, default=4096, help='Number of samples (rows of X)')
    parser.add_argument('--m', type=int, default=512, help='Feature dimension (cols of X)')
    parser.add_argument('--active_frac', type=float, default=0.1, help='Fraction of active rows/cols in U (0-1)')
    parser.add_argument('--a_mode', type=str, default='full', choices=['full', 'diagonal', 'decomposed'])
    parser.add_argument('--iters', type=int, default=5, help='Benchmark iterations per case')
    args = parser.parse_args()

    np_inputs = make_numpy_inputs(args.n, args.m, args.active_frac, args.a_mode)

    res_np = bench_numpy(np_inputs, args.iters)
    print_results('NumPy', res_np)

    if JAX_AVAILABLE:
        jinputs = make_jax_inputs(np_inputs)
        res_jax = bench_jax(jinputs, args.iters, jit=False)
        print_results('JAX (eager)', res_jax)
        res_jit = bench_jax(jinputs, args.iters, jit=True)
        print_results('JAX (JIT)', res_jit)
    else:
        print('\nJAX not available; skipping JAX benchmarks.')


if __name__ == '__main__':
    main()
