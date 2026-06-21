#!/usr/bin/env python3
"""Generate RecFNO-compatible heat conduction dataset (temperature6000.h5)."""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

L = 0.1
N = 200
DX = L / (N - 1)
T_REF = 298.0


def kappa(u: float) -> float:
    return 1.0 + 0.05 * (u - T_REF)


def gaussian_sources(rng: np.random.Generator) -> np.ndarray:
    x = np.linspace(0.0, L, N, dtype=np.float64)
    y = np.linspace(0.0, L, N, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing='ij')
    f = np.zeros((N, N), dtype=np.float64)
    for _ in range(int(rng.integers(1, 5))):
        x0 = rng.uniform(0.15 * L, 0.85 * L)
        y0 = rng.uniform(0.25 * L, 0.85 * L)
        sigma = rng.uniform(0.004, 0.015)
        amp = rng.uniform(3e4, 1.2e5)
        f += amp * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * sigma ** 2))
    return f


def apply_bc(u: np.ndarray, u_d: float, j0: int, j1: int) -> None:
    u[0, j0:j1] = u_d
    u[0, :j0] = u[1, :j0]
    u[0, j1:] = u[1, j1:]
    u[-1, :] = u[-2, :]
    u[:, 0] = u[:, 1]
    u[:, -1] = u[:, -2]


def build_system(u: np.ndarray, f: np.ndarray, u_d: float, j0: int, j1: int):
    rows, cols, data, rhs = [], [], [], []
    sink = np.zeros((N, N), dtype=bool)
    sink[0, j0:j1] = True

    def idx(i, j):
        return i * N + j

    for i in range(N):
        for j in range(N):
            p = idx(i, j)
            if sink[i, j]:
                rows.append(p); cols.append(p); data.append(1.0); rhs.append(u_d)
                continue
            if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                # Neumann: mirror neighbor
                ni, nj = i, j
                if i == 0: ni = 1
                elif i == N - 1: ni = N - 2
                if j == 0: nj = 1
                elif j == N - 1: nj = N - 2
                rows.append(p); cols.append(p); data.append(1.0)
                rows.append(p); cols.append(idx(ni, nj)); data.append(-1.0)
                rhs.append(0.0)
                continue

            ke = 0.5 * (kappa(u[i, j]) + kappa(u[i + 1, j]))
            kw = 0.5 * (kappa(u[i, j]) + kappa(u[i - 1, j]))
            kn = 0.5 * (kappa(u[i, j]) + kappa(u[i, j + 1]))
            ks = 0.5 * (kappa(u[i, j]) + kappa(u[i, j - 1]))
            axp, axm = ke / DX ** 2, kw / DX ** 2
            ayp, aym = kn / DX ** 2, ks / DX ** 2
            rows.extend([p, p, p, p, p])
            cols.extend([p, idx(i + 1, j), idx(i - 1, j), idx(i, j + 1), idx(i, j - 1)])
            data.extend([axp + axm + ayp + aym, -axp, -axm, -ayp, -aym])
            rhs.append(f[i, j])

    A = sparse.csr_matrix((data, (rows, cols)), shape=(N * N, N * N))
    return A, np.asarray(rhs, dtype=np.float64)


def solve_sample(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u_d = float(rng.uniform(293.0, 303.0))
    f = gaussian_sources(rng)
    j0, j1 = N // 2 - 8, N // 2 + 8

    u = np.full((N, N), 305.0, dtype=np.float64)
    for _ in range(8):
        A, rhs = build_system(u, f, u_d, j0, j1)
        u = spsolve(A, rhs).reshape(N, N)
        apply_bc(u, u_d, j0, j1)
    return u.astype(np.float32)


def _worker(seed: int):
    return seed, solve_sample(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=6000)
    parser.add_argument('--out', default=None,
                        help='output .h5 path (default: $RECFNO_DATA_ROOT/heat/temperature6000.h5)')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    if args.out is None:
        import sys
        _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from data.paths import HEAT_H5
        args.out = HEAT_H5

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + '.partial'
    if os.path.exists(tmp):
        os.remove(tmp)

    with h5py.File(tmp, 'w') as hf:
        dset = hf.create_dataset(
            'u', shape=(args.n_samples, 1, N, N), dtype='float32',
            chunks=(1, 1, N, N), compression='gzip', compression_opts=4,
        )
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_worker, s) for s in range(args.n_samples)]
            done = 0
            for fut in as_completed(futures):
                seed, field = fut.result()
                dset[seed] = field[None, None, :, :]
                done += 1
                if done % 50 == 0 or done == args.n_samples:
                    print(f'generated {done}/{args.n_samples}', flush=True)

    os.replace(tmp, args.out)
    u = h5py.File(args.out, 'r')['u']
    print('saved', args.out)
    print('shape', u.shape, 'range', float(u[:100].min()), float(u[:100].max()))


if __name__ == '__main__':
    main()
