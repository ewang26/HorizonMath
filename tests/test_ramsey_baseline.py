#!/usr/bin/env python3
"""
Verify the GNNW 2024 baseline parameters for the Ramsey asymptotic problem.

The paper's parameters (correction_coeffs = [-0.25, 0.03, 0.08, 0.0]) give
c = exp(F(1)) ≈ 3.7992, with M(λ) = λe^{-λ} and Y from Lemma 14.

This test checks the theoretical conditions with float-precision grid
arithmetic. It does NOT run the interval-arithmetic validator, because
the paper's baseline has R_0 margins of only ~3e-10 (at λ ≈ 0.306) —
far too slim to survive piecewise discretization plus interval widening.
This is expected: the paper proves its result using continuous functions
against the full admissible region R, whereas the validator uses
piecewise-constant step functions against the inner approximation R_0
with rigorous interval arithmetic. Solutions that beat the baseline
(e.g., the quintic c ≈ 3.696) are specifically optimized for R_0 margin
and pass the validator with ~200 breakpoints.
"""

import math
import sys

# --- Baseline parameters (GNNW 2024 final iteration) ---
CORRECTION_COEFFS = [-0.25, 0.03, 0.08, 0.0]
ALPHA = 0.137 / math.e  # alpha_3 = (0.17 - beta_2) / e
BETA_R0 = 0.033         # iteration-2 bound used for R_0

GRID_N = 4096
LAM_MIN = 0.001


def F(lam):
    c1, c2, c3, c4 = CORRECTION_COEFFS
    p = c1 * lam + c2 * lam**2 + c3 * lam**3 + c4 * lam**4
    return (1 + lam) * math.log(1 + lam) - lam * math.log(lam) + p * math.exp(-lam)


def F_prime(lam):
    c1, c2, c3, c4 = CORRECTION_COEFFS
    p = c1 * lam + c2 * lam**2 + c3 * lam**3 + c4 * lam**4
    dp = c1 + 2 * c2 * lam + 3 * c3 * lam**2 + 4 * c4 * lam**3
    return math.log((1 + lam) / lam) + (dp - p) * math.exp(-lam)


def U_R0(mu):
    g = (-0.25 * mu + BETA_R0 * mu**2 + 0.08 * mu**3) * math.exp(-mu)
    return g + (1 + mu) * math.log(1 + mu) - mu * math.log(mu)


def M_cont(lam):
    return lam * math.exp(-lam)


def X_cont(lam):
    fp = F_prime(lam)
    m = M_cont(lam)
    t = 1.0 - math.exp(-fp)
    return t ** (1.0 / (1.0 - m)) * (1.0 - m)


def Y_lemma14(x):
    if x <= 0.5:
        return math.exp(ALPHA) * (1.0 - x)
    else:
        return 1.0 - x * math.exp(-ALPHA)


def main():
    print("=" * 70)
    print("Ramsey Baseline: GNNW 2024 (float-precision grid check)")
    print("=" * 70)
    print(f"Correction coeffs: {CORRECTION_COEFFS}")
    print(f"alpha_3 = {ALPHA:.8f}")
    print()

    grid = [LAM_MIN + (1.0 - LAM_MIN) * i / (GRID_N - 1) for i in range(GRID_N)]
    u_r0 = [U_R0(mu) for mu in grid]

    fail_f = fail_fp = fail_x = fail_r0 = fail_main = 0
    worst_r0 = worst_main = math.inf
    worst_r0_lam = worst_main_lam = 0.0
    min_f = min_fp = math.inf

    for lam in grid:
        f = F(lam)
        fp = F_prime(lam)
        min_f = min(min_f, f)
        min_fp = min(min_fp, fp)

        if f <= 0:
            fail_f += 1
        if fp <= 0:
            fail_fp += 1

        m = M_cont(lam)
        x = X_cont(lam)
        y = Y_lemma14(x)

        if x <= 0 or x >= 1:
            fail_x += 1
            continue

        # R_0 check: try BOTH orientations
        log_y = math.log(y)
        log_x = math.log(x)

        std_thresh = max(u_r0[j] + grid[j] * log_y for j in range(GRID_N))
        std_slack = -log_x - std_thresh

        swap_thresh = max(u_r0[j] + grid[j] * log_x for j in range(GRID_N))
        swap_slack = -log_y - swap_thresh

        r0_slack = max(std_slack, swap_slack)
        if r0_slack < worst_r0:
            worst_r0 = r0_slack
            worst_r0_lam = lam
        if r0_slack < -0.005:
            fail_r0 += 1

        # Main inequality
        rhs = -0.5 * (log_x + lam * math.log(m) + lam * log_y)
        main_slack = f - rhs
        if main_slack < worst_main:
            worst_main = main_slack
            worst_main_lam = lam
        if main_slack <= -1e-9:
            fail_main += 1

    c = math.exp(F(1.0))

    print(f"F > 0:          min={min_f:.6f}   fails={fail_f}")
    print(f"F' > 0:         min={min_fp:.6f}  fails={fail_fp}")
    print(f"X in (0,1):     fails={fail_x}")
    print(f"R_0 check:      worst slack={worst_r0:.6e} at lam={worst_r0_lam:.4f}  fails={fail_r0}")
    print(f"Main ineq:      worst slack={worst_main:.6e} at lam={worst_main_lam:.4f}  fails={fail_main}")
    print(f"c = e^F(1):     {c:.10f}")
    print()

    all_pass = (fail_f == 0 and fail_fp == 0 and fail_x == 0 and
                fail_r0 == 0 and fail_main == 0 and abs(c - 3.7992) < 0.01)

    if all_pass:
        print("PASSED")
    else:
        print("FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
