#!/usr/bin/env python3
"""
Verify that the GNNW 2024 baseline parameters pass the Ramsey validator.

Paper: Gupta-Ndiaye-Norin-Wei, "Optimizing the CGMS upper bound on Ramsey numbers"
Final iteration: beta_3 = 0.03, alpha_3 = (0.17 - 0.033)/e = 0.137/e

The validator's R_0 uses the iteration-2 bound (beta_R0 = 0.033).
By the paper's convexity argument, G_R0(mu) <= -alpha_3*mu with equality at mu=1,
so the Lemma 14 Y values should satisfy R_0.
"""

import math
import json
import sys

# --- Parameters ---
CORRECTION_COEFFS = [-0.25, 0.03, 0.08, 0.0]
ALPHA = 0.137 / math.e  # alpha_3 = (0.17 - beta_2) / e
BETA_R0 = 0.033         # iteration-2 bound used for R_0

GRID_N = 4096
LAM_MIN = 0.01  # Theorem 13 proof only needs lambda >= epsilon' > 0
N_PIECES = 200  # piecewise-constant breakpoints (more = less interpolation error)


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
    """Rate function for R_0 (iteration-2 bound, beta = 0.033)."""
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


def make_piecewise(func, n_pieces):
    """Build piecewise-constant approximation. Returns (breakpoints, values)."""
    step = (1.0 - LAM_MIN) / n_pieces
    breakpoints = []
    values = []
    for i in range(n_pieces):
        lo = LAM_MIN + i * step
        hi = lo + step
        mid = (lo + hi) / 2.0
        if i > 0:
            breakpoints.append(round(lo, 10))
        values.append(func(mid))
    # Last value covers [last_bp, 1]
    return breakpoints, values


def eval_pw(breakpoints, values, lam):
    for i, b in enumerate(breakpoints):
        if lam < b:
            return values[i]
    return values[len(breakpoints)]


def main():
    print("=" * 70)
    print("Ramsey Baseline Verification")
    print("=" * 70)
    print(f"Correction coeffs: {CORRECTION_COEFFS}")
    print(f"alpha_3 = {ALPHA:.8f}")
    print(f"beta_R0 = {BETA_R0}")
    print()

    # Build grid
    grid = [LAM_MIN + (1.0 - LAM_MIN) * i / (GRID_N - 1) for i in range(GRID_N)]

    # Build piecewise M and Y
    m_bp, m_vals = make_piecewise(M_cont, N_PIECES)
    y_func = lambda lam: Y_lemma14(X_cont(lam))
    y_bp, y_vals = make_piecewise(y_func, N_PIECES)

    # Precompute U_R0 on grid
    u_r0 = [U_R0(mu) for mu in grid]

    # Precompute R_0 thresholds for distinct Y values
    distinct_y = set()
    for lam in grid:
        distinct_y.add(eval_pw(y_bp, y_vals, lam))

    y_threshold = {}
    for y in distinct_y:
        log_y = math.log(y)
        y_threshold[y] = max(u_r0[j] + grid[j] * log_y for j in range(GRID_N))

    # --- Check conditions ---
    fail_f = 0
    fail_fp = 0
    fail_x = 0
    fail_r0 = 0
    fail_main = 0

    worst_r0 = math.inf
    worst_r0_lam = 0
    worst_main = math.inf
    worst_main_lam = 0
    min_f = math.inf
    min_fp = math.inf

    for lam in grid:
        f = F(lam)
        fp = F_prime(lam)
        min_f = min(min_f, f)
        min_fp = min(min_fp, fp)

        if f <= 0:
            fail_f += 1
        if fp <= 0:
            fail_fp += 1

        m = eval_pw(m_bp, m_vals, lam)
        y = eval_pw(y_bp, y_vals, lam)

        t = 1.0 - math.exp(-fp)
        x = t ** (1.0 / (1.0 - m)) * (1.0 - m)

        if x <= 0 or x >= 1:
            fail_x += 1
            continue

        # R_0 check: try BOTH orientations (R is symmetric)
        neg_log_x = -math.log(x)
        neg_log_y = -math.log(y)

        # Standard: -log(x) >= max_mu [U_R0(mu) + mu*log(y)]
        std_slack = neg_log_x - y_threshold[y]

        # Swapped: -log(y) >= max_mu [U_R0(mu) + mu*log(x)]
        log_x = math.log(x)
        swap_thresh = max(u_r0[j] + grid[j] * log_x for j in range(GRID_N))
        swap_slack = neg_log_y - swap_thresh

        r0_slack = max(std_slack, swap_slack)

        if r0_slack < worst_r0:
            worst_r0 = r0_slack
            worst_r0_lam = lam
        if r0_slack < -0.005:  # match validator R0_TOL
            fail_r0 += 1

        # Main inequality
        rhs = -0.5 * (math.log(x) + lam * math.log(m) + lam * math.log(y))
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
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")

    # Output baseline solution dict
    solution = {
        "correction_coeffs": CORRECTION_COEFFS,
        "M": {"breakpoints": [round(b, 10) for b in m_bp],
              "values": [round(v, 10) for v in m_vals]},
        "Y": {"breakpoints": [round(b, 10) for b in y_bp],
              "values": [round(v, 10) for v in y_vals]},
        "notes": "GNNW 2024 final iteration (beta=0.03, alpha=0.137/e)"
    }

    print(f"\nM: {len(m_bp)} breakpoints, {len(m_vals)} values")
    print(f"Y: {len(y_bp)} breakpoints, {len(y_vals)} values")
    print(f"\nSolution dict (for validator):")
    print(json.dumps(solution, indent=2)[:500] + "...")

    if not all_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
