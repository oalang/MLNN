import argparse
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from mlnn.activation import get_activation
from mlnn.engine import (
    compute_I,
    alt_compute_I,
    compute_O,
    alt_compute_O,
    compute_U,
    alt_compute_U,
)

# Optional JAX import
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except Exception:
    jax = None
    jnp = None
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
    ms = (end - start) * 1000.0 / iters
    return ms


def make_numpy_inputs(n: int, c: int, q: float) -> dict:
    rng = np.random.default_rng(0)
    # Y is (n,1) per spec
    Y = rng.integers(0, c, size=(n, 1))
    Y1 = Y.ravel()
    # Original T: 1/-1, Alt T: boolean. Both NxN.
    T_alt = np.equal.outer(Y1, Y1)
    T_orig = np.where(T_alt, 1.0, -1.0)
    # Q depends on T and q (original path uses compute_Q, alt uses mask with where)
    Q_orig = np.where(T_orig == 1.0, q, 1.0)
    E = np.array([[1.3]], dtype=float)
    D = rng.standard_normal((n, n)).astype(float)
    # Inner/outer losses from activation (default used by engine)
    act = get_activation('relu', 1.0)
    inner_loss_intr = act.intr
    inner_loss_func_intr = act.func_intr
    inner_loss_grad_intr = act.grad_intr
    outer_loss_intr = None
    outer_loss_grad_intr = None
    # Build a valid I tuple for O/U benches (independent tests)
    I_tuple = inner_loss_intr(D - E)  # shape-compatible with NxN
    return dict(
        Y=Y,
        T_orig=T_orig,
        T_alt=T_alt,
        Q_orig=Q_orig,
        E=E,
        D=D,
        I_tuple=I_tuple,
        q=q,
        inner_loss_intr=inner_loss_intr,
        inner_loss_func_intr=inner_loss_func_intr,
        inner_loss_grad_intr=inner_loss_grad_intr,
        outer_loss_intr=outer_loss_intr,
        outer_loss_grad_intr=outer_loss_grad_intr,
    )


def make_jax_inputs(np_inputs: dict) -> Optional[dict]:
    if not JAX_AVAILABLE:
        return None
    # Convert numpy arrays to jax arrays
    J = {}
    for k, v in np_inputs.items():
        if isinstance(v, np.ndarray):
            J[k] = jnp.array(v)
        elif isinstance(v, tuple):
            # Convert tuple of arrays to tuple of jax arrays
            J[k] = tuple(jnp.array(x) for x in v)
        else:
            J[k] = v
    return J


def bench_compute_I_numpy(inputs: dict, iters: int) -> list[BenchResult]:
    E = inputs['E']
    D = inputs['D']
    T_orig = inputs['T_orig']
    T_alt = inputs['T_alt']
    inner_loss_intr = inputs['inner_loss_intr']

    def run_orig():
        _ = compute_I(E, T_orig, D, inner_loss_intr)

    def run_alt():
        _ = alt_compute_I(E, T_alt, D, inner_loss_intr)

    ms_orig = timeit_fn(run_orig, iters=iters)
    ms_alt = timeit_fn(run_alt, iters=iters)
    return [
        BenchResult('numpy/compute_I (orig T=±1)', ms_orig, iters),
        BenchResult('numpy/alt_compute_I (alt T=bool)', ms_alt, iters),
    ]


def bench_compute_O_numpy(inputs: dict, iters: int) -> list[BenchResult]:
    T_alt = inputs['T_alt']
    Q_orig = inputs['Q_orig']
    I_tuple = inputs['I_tuple']
    q = inputs['q']
    inner_loss_func_intr = inputs['inner_loss_func_intr']
    outer_loss_intr = inputs['outer_loss_intr']

    # For original compute_O, args: (N, Q, I, q, inner_loss_func_intr, outer_loss_intr)
    n = I_tuple[0].shape[0]
    N_orig = np.zeros((n, 1), dtype=float)

    def run_orig():
        _ = compute_O(N_orig, Q_orig, I_tuple, q, inner_loss_func_intr, outer_loss_intr)

    # For alt_compute_O, args: (T, N, I, q, inner_loss_func_intr, outer_loss_intr)
    def run_alt():
        _ = alt_compute_O(T_alt, N_orig, I_tuple, q, inner_loss_func_intr, outer_loss_intr)

    ms_orig = timeit_fn(run_orig, iters=iters)
    ms_alt = timeit_fn(run_alt, iters=iters)
    return [
        BenchResult('numpy/compute_O (orig)', ms_orig, iters),
        BenchResult('numpy/alt_compute_O (alt)', ms_alt, iters),
    ]


def bench_compute_U_numpy(inputs: dict, iters: int) -> list[BenchResult]:
    T_orig = inputs['T_orig']
    T_alt = inputs['T_alt']
    Q_orig = inputs['Q_orig']
    I_tuple = inputs['I_tuple']
    q = inputs['q']
    inner_loss_grad_intr = inputs['inner_loss_grad_intr']
    outer_loss_grad_intr = inputs['outer_loss_grad_intr']

    n = I_tuple[0].shape[0]
    O = np.zeros((n, n), dtype=float)

    def run_orig():
        _ = compute_U(T_orig, Q_orig, I_tuple, O, q, inner_loss_grad_intr, outer_loss_grad_intr)

    def run_alt():
        _ = alt_compute_U(T_alt, I_tuple, O, q, inner_loss_grad_intr, outer_loss_grad_intr)

    ms_orig = timeit_fn(run_orig, iters=iters)
    ms_alt = timeit_fn(run_alt, iters=iters)
    return [
        BenchResult('numpy/compute_U (orig)', ms_orig, iters),
        BenchResult('numpy/alt_compute_U (alt)', ms_alt, iters),
    ]


# Minimal JAX ReLU intr/func/grad equivalents to match engine defaults
if JAX_AVAILABLE:
    def relu_intr_jax(X):
        return (X + 1.0,)

    def relu_func_intr_jax(I):
        A = I[0]
        return jnp.maximum(A, 0)

    def relu_grad_intr_jax(I):
        A = I[0]
        return (A > 0).astype(A.dtype)


def bench_compute_I_jax(jinputs: dict, iters: int, jit: bool) -> list[BenchResult]:
    E = jinputs['E']
    D = jinputs['D']
    T_orig = jinputs['T_orig']
    T_alt = jinputs['T_alt']

    def compute_I_orig_jax():
        return relu_intr_jax((D - E) * T_orig)

    def compute_I_alt_jax():
        X = E - D
        X = jnp.where(T_alt, -X, X)
        return relu_intr_jax(X)

    fn_orig = jax.jit(compute_I_orig_jax) if jit else compute_I_orig_jax
    fn_alt = jax.jit(compute_I_alt_jax) if jit else compute_I_alt_jax

    if jit:
        _ = fn_orig()[0].block_until_ready()
        _ = fn_alt()[0].block_until_ready()

    def run_orig():
        y = fn_orig()
        if jit:
            _ = y[0].block_until_ready()

    def run_alt():
        y = fn_alt()
        if jit:
            _ = y[0].block_until_ready()

    ms_orig = timeit_fn(run_orig, iters=iters)
    ms_alt = timeit_fn(run_alt, iters=iters)
    tag = 'jax-jit' if jit else 'jax'
    return [
        BenchResult(f'{tag}/compute_I (orig T=±1)', ms_orig, iters),
        BenchResult(f'{tag}/alt_compute_I (alt T=bool)', ms_alt, iters),
    ]


def bench_compute_O_jax(jinputs: dict, iters: int, jit: bool) -> list[BenchResult]:
    T_alt = jinputs['T_alt']
    T_orig = jinputs['T_orig']
    q = float(jinputs['q'])

    # Build an I tuple independently
    I_tuple = relu_intr_jax(jinputs['D'] - jinputs['E'])

    def compute_O_orig_jax():
        O = relu_func_intr_jax(I_tuple)
        if q != 1:
            Q = jnp.where(T_orig == 1.0, q, 1.0)
            O = O * Q
        return O

    def compute_O_alt_jax():
        O = relu_func_intr_jax(I_tuple)
        if q != 1:
            O = jnp.where(T_alt, q * O, O)
        return O

    fn_orig = jax.jit(compute_O_orig_jax) if jit else compute_O_orig_jax
    fn_alt = jax.jit(compute_O_alt_jax) if jit else compute_O_alt_jax
    if jit:
        _ = fn_orig().block_until_ready()
        _ = fn_alt().block_until_ready()

    def run_orig():
        y = fn_orig()
        if jit:
            _ = y.block_until_ready()

    def run_alt():
        y = fn_alt()
        if jit:
            _ = y.block_until_ready()

    ms_orig = timeit_fn(run_orig, iters=iters)
    ms_alt = timeit_fn(run_alt, iters=iters)
    tag = 'jax-jit' if jit else 'jax'
    return [
        BenchResult(f'{tag}/compute_O (orig)', ms_orig, iters),
        BenchResult(f'{tag}/alt_compute_O (alt)', ms_alt, iters),
    ]


def bench_compute_U_jax(jinputs: dict, iters: int, jit: bool) -> list[BenchResult]:
    T_orig = jinputs['T_orig']
    T_alt = jinputs['T_alt']
    q = float(jinputs['q'])

    # Build an I tuple independently
    I_tuple = relu_intr_jax(jinputs['D'] - jinputs['E'])

    def compute_U_orig_jax():
        U = relu_grad_intr_jax(I_tuple) * T_orig
        if q != 1:
            Q = jnp.where(T_orig == 1.0, q, 1.0)
            U = U * Q
        return U

    def compute_U_alt_jax():
        U = relu_grad_intr_jax(I_tuple)
        U = jnp.where(T_alt, U, -U)
        if q != 1:
            U = jnp.where(T_alt, q * U, U)
        return U

    fn_orig = jax.jit(compute_U_orig_jax) if jit else compute_U_orig_jax
    fn_alt = jax.jit(compute_U_alt_jax) if jit else compute_U_alt_jax
    if jit:
        _ = fn_orig().block_until_ready()
        _ = fn_alt().block_until_ready()

    def run_orig():
        y = fn_orig()
        if jit:
            _ = y.block_until_ready()

    def run_alt():
        y = fn_alt()
        if jit:
            _ = y.block_until_ready()

    ms_orig = timeit_fn(run_orig, iters=iters)
    ms_alt = timeit_fn(run_alt, iters=iters)
    tag = 'jax-jit' if jit else 'jax'
    return [
        BenchResult(f'{tag}/compute_U (orig)', ms_orig, iters),
        BenchResult(f'{tag}/alt_compute_U (alt)', ms_alt, iters),
    ]


def print_results(title: str, results: list[BenchResult]) -> None:
    print(f'\n== {title} ==')
    for r in results:
        print(f'{r.name:40s}  {r.ms_per_call:8.3f} ms  (iters={r.iters})')


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark compute functions (orig vs alt, NumPy vs JAX).')
    parser.add_argument('--n', type=int, default=10000, help='Sample count (n)')
    parser.add_argument('--c', type=int, default=3, help='Number of classes (c)')
    parser.add_argument('--q', type=float, default=2.3, help='Scalar q')
    parser.add_argument('--iters', type=int, default=5, help='Benchmark iterations per case')
    args = parser.parse_args()

    np_inputs = make_numpy_inputs(args.n, args.c, args.q)

    # NumPy benches
    res_I_np = bench_compute_I_numpy(np_inputs, args.iters)
    res_O_np = bench_compute_O_numpy(np_inputs, args.iters)
    res_U_np = bench_compute_U_numpy(np_inputs, args.iters)

    print_results('NumPy', res_I_np + res_O_np + res_U_np)

    # JAX benches if available
    if JAX_AVAILABLE:
        jinputs = make_jax_inputs(np_inputs)
        res_I_jax = bench_compute_I_jax(jinputs, args.iters, jit=False)
        res_O_jax = bench_compute_O_jax(jinputs, args.iters, jit=False)
        res_U_jax = bench_compute_U_jax(jinputs, args.iters, jit=False)
        print_results('JAX (eager)', res_I_jax + res_O_jax + res_U_jax)

        res_I_jit = bench_compute_I_jax(jinputs, args.iters, jit=True)
        res_O_jit = bench_compute_O_jax(jinputs, args.iters, jit=True)
        res_U_jit = bench_compute_U_jax(jinputs, args.iters, jit=True)
        print_results('JAX (JIT)', res_I_jit + res_O_jit + res_U_jit)
    else:
        print('\nJAX not available; skipping JAX benchmarks.')


if __name__ == '__main__':
    main()
