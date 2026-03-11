#!/usr/bin/env python3
"""
Validator for problem 047b: Improve a 10D Periodic Packing (P10c Baseline)

A periodic packing is specified by:
- a full-rank lattice basis matrix B (10x10, ROWS are basis vectors in R^10)
- a list of k shift vectors s_i in R^10 (k spheres per fundamental cell)

Packing centers are:
    P = union_{i=1..k} (L + s_i),  where L = { z^T B : z in Z^10 }.

The packing radius is r = d_min / 2 where
    d_min = min_{i,j} min_{z in Z^10} || (s_i - s_j) + B^T z ||,
with the convention that for i=j we exclude z=0.

Packing density:
    density = k * Vol(Ball_10(r)) / covolume,
    covolume = |det(B)|.

We compute d_min using:
- LLL reduction of the lattice basis
- QR decomposition B = Q R
- Schnorr-Euchner enumeration for SVP and CVP (with a global cutoff)

Metric key: "packing_density" (maximize).
"""

import argparse
import math
from typing import Any, Tuple, Optional

import numpy as np

from . import ValidationResult, load_solution, output_result, success, failure


DIMENSION = 10
TOL_DET = 1e-12
TOL_SHIFT0 = 1e-8
MAX_ABS_ENTRY = 1e3
MAX_ABS_SHIFT = 1e3
MAX_COND = 1e10
MAX_SHIFTS = 64


def sphere_volume(r: float, n: int) -> float:
    return (math.pi ** (n / 2.0)) * (r ** n) / math.gamma(n / 2.0 + 1.0)


def gram_schmidt_cols(B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m, n = B.shape
    Bstar = np.zeros((m, n), dtype=np.float64)
    mu = np.zeros((n, n), dtype=np.float64)
    bstar_norm2 = np.zeros(n, dtype=np.float64)

    for i in range(n):
        v = B[:, i].copy()
        for j in range(i):
            denom = bstar_norm2[j]
            if denom <= 0:
                mu[i, j] = 0.0
                continue
            mu[i, j] = float(np.dot(B[:, i], Bstar[:, j]) / denom)
            v -= mu[i, j] * Bstar[:, j]
        Bstar[:, i] = v
        bstar_norm2[i] = float(np.dot(v, v))
        if bstar_norm2[i] <= 0:
            bstar_norm2[i] = 0.0
    return Bstar, mu, bstar_norm2


def lll_reduce_cols(B: np.ndarray, delta: float = 0.99, max_iter: int = 8000) -> np.ndarray:
    B = B.copy().astype(np.float64)
    n = B.shape[1]
    k = 1
    it = 0
    Bstar, mu, bstar_norm2 = gram_schmidt_cols(B)

    while k < n and it < max_iter:
        it += 1

        for j in range(k - 1, -1, -1):
            q = int(np.round(mu[k, j]))
            if q != 0:
                B[:, k] -= q * B[:, j]

        Bstar, mu, bstar_norm2 = gram_schmidt_cols(B)
        if bstar_norm2[k] == 0 or bstar_norm2[k - 1] == 0:
            return B

        if bstar_norm2[k] >= (delta - mu[k, k - 1] ** 2) * bstar_norm2[k - 1]:
            k += 1
        else:
            B[:, [k, k - 1]] = B[:, [k - 1, k]]
            Bstar, mu, bstar_norm2 = gram_schmidt_cols(B)
            k = max(k - 1, 1)

    return B


def _nearest_plane(R: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    n = R.shape[0]
    z = np.zeros(n, dtype=np.int64)
    y = target.astype(np.float64).copy()

    for k in range(n - 1, -1, -1):
        Rkk = float(R[k, k])
        if abs(Rkk) < 1e-18:
            z[k] = 0
            continue
        ck = y[k] / Rkk
        z[k] = int(np.round(ck))
        if k > 0:
            y[:k] -= R[:k, k] * z[k]

    resid = R @ z - target
    return z, float(np.dot(resid, resid))


def _enum_se(
    R: np.ndarray,
    target: np.ndarray,
    best: float,
    require_nonzero: bool = False
) -> Tuple[float, Optional[np.ndarray]]:
    n = R.shape[0]
    z = np.zeros(n, dtype=np.int64)
    best_z: Optional[np.ndarray] = None

    def rec(k: int, dist2: float):
        nonlocal best, best_z
        if dist2 >= best:
            return
        if k < 0:
            if require_nonzero and np.all(z == 0):
                return
            best = dist2
            best_z = z.copy()
            return

        s = float(target[k])
        if k + 1 < n:
            s -= float(np.dot(R[k, k + 1 :], z[k + 1 :]))

        Rkk = float(R[k, k])
        if abs(Rkk) < 1e-18:
            rec(k - 1, dist2 + s * s)
            return

        c = s / Rkk
        m = int(np.round(c))

        step = 0
        while True:
            if step == 0:
                candidates = [m]
                d = abs(c - m)
                if dist2 + (Rkk * d) ** 2 >= best:
                    break
            else:
                t_probe = m + step
                d = abs(c - t_probe)
                if dist2 + (Rkk * d) ** 2 >= best:
                    break
                candidates = [m + step, m - step]

            for t in candidates:
                z[k] = int(t)
                diff = s - Rkk * float(t)
                rec(k - 1, dist2 + diff * diff)

            step += 1

    rec(n - 1, 0.0)
    return best, best_z


def reduce_shifts_mod_lattice(B_cols: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    invB = np.linalg.inv(B_cols)
    reduced = np.zeros_like(shifts)
    for i, s in enumerate(shifts):
        coords = invB @ s
        nint = np.round(coords)
        reduced[i] = s - (B_cols @ nint)
    return reduced


def shortest_vector_sq(R: np.ndarray, B_cols_red: np.ndarray) -> float:
    col_norm2 = np.sum(B_cols_red * B_cols_red, axis=0)
    best = float(np.min(col_norm2))
    best, _ = _enum_se(R, target=np.zeros(DIMENSION), best=best, require_nonzero=True)
    return best


def coset_distance_sq(R: np.ndarray, Q: np.ndarray, t: np.ndarray, cutoff: float) -> float:
    c = Q.T @ t
    target = -c  # minimize ||Rz - target||^2
    _, dist2_babai = _nearest_plane(R, target)
    best = min(cutoff, dist2_babai)
    best, _ = _enum_se(R, target=target, best=best, require_nonzero=False)
    return best


def validate(solution: Any) -> ValidationResult:
    if not isinstance(solution, dict):
        return failure("Invalid format: expected dict with 'basis' and 'shifts'")
    if "basis" not in solution or "shifts" not in solution:
        return failure("Missing keys: expected {'basis': ..., 'shifts': ...}")

    try:
        B_rows = np.array(solution["basis"], dtype=np.float64)
        shifts = np.array(solution["shifts"], dtype=np.float64)
    except (ValueError, TypeError) as e:
        return failure(f"Failed to parse input: {e}")

    if B_rows.shape != (DIMENSION, DIMENSION):
        return failure(f"Basis must be {DIMENSION}x{DIMENSION}, got {B_rows.shape}")
    if shifts.ndim != 2 or shifts.shape[1] != DIMENSION:
        return failure(f"Shifts must be a k x {DIMENSION} array, got shape {shifts.shape}")

    k = shifts.shape[0]
    if k < 1 or k > MAX_SHIFTS:
        return failure(f"Number of shifts k must be in [1,{MAX_SHIFTS}], got {k}")

    if not np.all(np.isfinite(B_rows)) or not np.all(np.isfinite(shifts)):
        return failure("Non-finite entries in basis or shifts")
    if float(np.max(np.abs(B_rows))) > MAX_ABS_ENTRY:
        return failure(f"Basis entries too large (>|{MAX_ABS_ENTRY}|)")
    if float(np.max(np.abs(shifts))) > MAX_ABS_SHIFT:
        return failure(f"Shift entries too large (>|{MAX_ABS_SHIFT}|)")

    if float(np.linalg.norm(shifts[0])) > TOL_SHIFT0:
        return failure("Require shifts[0] to be the zero vector (for canonicalization)")

    # Rows -> columns
    B_cols = B_rows.T.copy()
    det = float(np.linalg.det(B_cols))
    if not np.isfinite(det) or abs(det) < TOL_DET:
        return failure("Basis is singular (determinant ~ 0)")
    covolume = abs(det)

    cond = float(np.linalg.cond(B_cols))
    if not np.isfinite(cond) or cond > MAX_COND:
        return failure(f"Basis is ill-conditioned (cond={cond:.3e} > {MAX_COND:g})")

    shifts = reduce_shifts_mod_lattice(B_cols, shifts)

    # Duplicate shifts check (after reduction)
    rounded = np.round(shifts / 1e-8).astype(np.int64)
    uniq = {tuple(row.tolist()) for row in rounded}
    if len(uniq) != k:
        return failure("Duplicate shifts detected modulo lattice (coincident centers)")

    # Reduce lattice for faster enumeration
    B_cols_red = lll_reduce_cols(B_cols)
    Q, R = np.linalg.qr(B_cols_red)

    # SVP for i=j (intra-coset distance)
    svp2 = shortest_vector_sq(R, B_cols_red)
    if not np.isfinite(svp2) or svp2 <= 0:
        return failure("Failed to compute a valid shortest lattice vector")
    min_dist2 = svp2

    # Pairwise cosets with global cutoff
    for i in range(k):
        si = shifts[i]
        for j in range(i + 1, k):
            t = si - shifts[j]
            d2 = coset_distance_sq(R, Q, t, cutoff=min_dist2)
            if d2 < min_dist2:
                min_dist2 = d2
                if min_dist2 < 1e-14:
                    return failure("Packing has overlapping spheres (min distance ~ 0)")

    min_dist = float(np.sqrt(min_dist2))
    packing_radius = min_dist / 2.0
    density = (k * sphere_volume(packing_radius, DIMENSION)) / covolume

    return success(
        f"Periodic packing in R^{DIMENSION}: k={k}, min distance ~ {min_dist:.8f}, packing density ~ {density:.12f}",
        dimension=DIMENSION,
        k=int(k),
        determinant=float(det),
        covolume=float(covolume),
        min_distance=float(min_dist),
        packing_radius=float(packing_radius),
        packing_density=float(density),
        metric_key="packing_density",
    )


def main():
    parser = argparse.ArgumentParser(description="Validate periodic sphere packing in dimension 10")
    parser.add_argument("solution", help="Solution as JSON string or path to JSON file")
    args = parser.parse_args()

    sol = load_solution(args.solution)
    result = validate(sol)
    output_result(result)


if __name__ == "__main__":
    main()
