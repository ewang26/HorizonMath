#!/usr/bin/env python3
"""
Validator: Ramsey diagonal upper-bound base c via Theorem 13 of GNNW 2024.

The submission provides parameters for a certificate that R(k,k) <= c^{k+o(k)}.

F is parameterized as:
  F(lam) = (1+lam)*log(1+lam) - lam*log(lam) + p(lam)*exp(-lam)
where p(lam) = c1*lam + c2*lam^2 + c3*lam^3 + c4*lam^4.

The validator checks:
  1. F(lam) > 0 and F'(lam) > 0 on the grid
  2. (X(lam), Y(lam)) in R_0 for all lam (full check against all mu)
  3. Main inequality: F(lam) > -1/2 * (log X + lam*log M + lam*log Y)
  4. Metric: c = exp(F(1)) (minimize)

R_0 is a fixed inner approximation of R defined by the iteration-2 bound
from GNNW 2024 (beta_R0 = 0.033):
  G_R0(mu) = (-0.25*mu + 0.033*mu^2 + 0.08*mu^3)*exp(-mu)
  U_R0(mu) = G_R0(mu) + (1+mu)*log(1+mu) - mu*log(mu)
  (x,y) in R_0  iff  -log(x) - mu*log(y) >= U_R0(mu) for all mu in (0,1]

Input format:
    {
        "correction_coeffs": [c1, c2, c3, c4],
        "M": {"breakpoints": [...], "values": [...]},
        "Y": {"breakpoints": [...], "values": [...]},
        "notes": "..."
    }

M and Y are piecewise-constant: breakpoints is a strictly increasing list
in (0,1) of length n (0 <= n <= 200), and values has length n+1.
"""

import argparse
import math
from typing import Any

from . import ValidationResult, load_solution, output_result, success, failure

# --- Constants ---
LAMBDA_MIN = 0.01   # Theorem 13 only needs lambda >= epsilon' > 0; avoids piecewise noise at small lambda
GRID_N = 4096
STRICT_EPS = 1e-10
R0_TOL = 0.003      # tolerance for R_0 check (piecewise discretization at crossover)
MAIN_TOL = 1e-4     # tolerance for main inequality (tight: continuous slack is ~2.7e-4 at baseline)
MAX_BREAKPOINTS = 200

# R_0 is defined by the iteration-2 bound (beta = 0.033) from GNNW 2024.
BETA_R0 = 0.033


def eval_piecewise(breakpoints: list[float], values: list[float],
                   lam: float) -> float:
    """Evaluate a piecewise-constant function.

    breakpoints: [b_0, ..., b_{n-1}], strictly increasing in (0,1)
    values: [v_0, ..., v_n], length n+1
    Returns v_i where b_{i-1} <= lam < b_i (b_{-1} = -inf, b_n = +inf).
    """
    for i, b in enumerate(breakpoints):
        if lam < b:
            return values[i]
    return values[len(breakpoints)]


def validate_piecewise(data: Any, name: str) -> tuple[list[float], list[float], str | None]:
    """Parse and validate a piecewise-constant function spec.

    Returns (breakpoints, values, error_msg). error_msg is None on success.
    """
    if not isinstance(data, dict):
        return [], [], f"{name}: expected dict with 'breakpoints' and 'values'"

    breakpoints = data.get('breakpoints')
    values = data.get('values')

    if breakpoints is None or values is None:
        return [], [], f"{name}: missing 'breakpoints' or 'values'"
    if not isinstance(breakpoints, list) or not isinstance(values, list):
        return [], [], f"{name}: 'breakpoints' and 'values' must be lists"
    if len(values) != len(breakpoints) + 1:
        return [], [], (
            f"{name}: len(values) must be len(breakpoints)+1, "
            f"got {len(values)} vs {len(breakpoints)}"
        )
    if len(breakpoints) > MAX_BREAKPOINTS:
        return [], [], f"{name}: too many breakpoints ({len(breakpoints)} > {MAX_BREAKPOINTS})"

    try:
        bp_out = [float(b) for b in breakpoints]
    except (ValueError, TypeError) as e:
        return [], [], f"{name}: invalid breakpoint: {e}"

    for i, b in enumerate(bp_out):
        if not math.isfinite(b) or b <= 0 or b >= 1:
            return [], [], f"{name}: breakpoint {i} = {b} not in (0,1)"

    for i in range(len(bp_out) - 1):
        if bp_out[i] >= bp_out[i + 1]:
            return [], [], (
                f"{name}: breakpoints not strictly increasing at {i}: "
                f"{bp_out[i]} >= {bp_out[i+1]}"
            )

    try:
        val_out = [float(v) for v in values]
    except (ValueError, TypeError) as e:
        return [], [], f"{name}: invalid value: {e}"

    for i, v in enumerate(val_out):
        if not math.isfinite(v) or v <= 0 or v >= 1:
            return [], [], f"{name}: value {i} = {v} not in (0,1)"

    return bp_out, val_out, None


def validate(solution: Any) -> ValidationResult:
    """Validate a Ramsey upper-bound certificate based on Theorem 13."""

    if not isinstance(solution, dict):
        return failure("Invalid format: expected dict")

    # --- Parse correction_coeffs ---
    coeffs_raw = solution.get('correction_coeffs')
    if not isinstance(coeffs_raw, list) or len(coeffs_raw) != 4:
        return failure("'correction_coeffs' must be a list of 4 numbers")
    try:
        c1, c2, c3, c4 = [float(x) for x in coeffs_raw]
    except (ValueError, TypeError) as e:
        return failure(f"Invalid correction coefficient: {e}")
    for i, c in enumerate([c1, c2, c3, c4]):
        if not math.isfinite(c):
            return failure(f"correction_coeffs[{i}] is not finite")

    # --- Parse M and Y ---
    m_data = solution.get('M')
    if m_data is None:
        return failure("Missing 'M'")
    m_bp, m_vals, err = validate_piecewise(m_data, "M")
    if err:
        return failure(err)

    y_data = solution.get('Y')
    if y_data is None:
        return failure("Missing 'Y'")
    y_bp, y_vals, err = validate_piecewise(y_data, "Y")
    if err:
        return failure(err)

    # --- Build grid ---
    step = (1.0 - LAMBDA_MIN) / (GRID_N - 1)
    grid = [LAMBDA_MIN + i * step for i in range(GRID_N)]

    # --- Precompute U_R0(mu) on grid ---
    # G_R0(mu) = (-0.25*mu + BETA_R0*mu^2 + 0.08*mu^3) * exp(-mu)
    # U_R0(mu) = G_R0(mu) + (1+mu)*log(1+mu) - mu*log(mu)
    u_r0 = []
    for mu in grid:
        g = (-0.25 * mu + BETA_R0 * mu * mu + 0.08 * mu * mu * mu) * math.exp(-mu)
        u = g + (1.0 + mu) * math.log(1.0 + mu) - mu * math.log(mu)
        u_r0.append(u)

    # --- Precompute R_0 thresholds for each distinct Y value ---
    # Since R(k,l) = R(l,k), (x,y) in R iff (y,x) in R.
    # We check BOTH orientations and pass if either works:
    #   Standard: -log(x) >= max_mu [U_R0(mu) + mu*log(y)]
    #   Swapped:  -log(y) >= max_mu [U_R0(mu) + mu*log(x)]
    # Precompute standard thresholds per distinct Y value.
    distinct_y = set()
    for lam in grid:
        distinct_y.add(eval_piecewise(y_bp, y_vals, lam))

    y_threshold: dict[float, float] = {}
    for y in distinct_y:
        log_y = math.log(y)
        best = -math.inf
        for j in range(GRID_N):
            val = u_r0[j] + grid[j] * log_y
            if val > best:
                best = val
        y_threshold[y] = best

    # --- Check all grid points ---
    worst_r0_slack = math.inf
    worst_r0_lam = 0.0
    worst_main_slack = math.inf
    worst_main_lam = 0.0

    for lam in grid:
        # Compute p(lam) and p'(lam)
        p_val = c1 * lam + c2 * lam ** 2 + c3 * lam ** 3 + c4 * lam ** 4
        p_prime = c1 + 2 * c2 * lam + 3 * c3 * lam ** 2 + 4 * c4 * lam ** 3

        exp_neg_lam = math.exp(-lam)

        # F(lam) = (1+lam)*log(1+lam) - lam*log(lam) + p(lam)*exp(-lam)
        f_val = (1.0 + lam) * math.log(1.0 + lam) - lam * math.log(lam) + p_val * exp_neg_lam

        # F'(lam) = log((1+lam)/lam) + [p'(lam) - p(lam)]*exp(-lam)
        f_prime = math.log((1.0 + lam) / lam) + (p_prime - p_val) * exp_neg_lam

        if not (math.isfinite(f_val) and math.isfinite(f_prime)):
            return failure(f"Non-finite F or F' at lam={lam:.6e}")
        if f_val <= STRICT_EPS:
            return failure(f"F(lam) = {f_val:.6e} <= 0 at lam={lam:.6e}")
        if f_prime <= STRICT_EPS:
            return failure(f"F'(lam) = {f_prime:.6e} <= 0 at lam={lam:.6e}")

        # M(lam), Y(lam)
        m_val = eval_piecewise(m_bp, m_vals, lam)
        y_val = eval_piecewise(y_bp, y_vals, lam)

        # X(lam) = (1 - exp(-F'(lam)))^{1/(1-M(lam))} * (1 - M(lam))
        t = 1.0 - math.exp(-f_prime)
        exponent = 1.0 / (1.0 - m_val)
        x_val = (t ** exponent) * (1.0 - m_val)

        if not (STRICT_EPS < x_val < 1.0 - STRICT_EPS):
            return failure(f"X(lam) = {x_val:.6e} out of (0,1) at lam={lam:.6e}")

        # R_0 check: try BOTH orientations (since R is symmetric).
        # Standard: -log(x) >= max_mu [U_R0(mu) + mu*log(y)]
        # Swapped:  -log(y) >= max_mu [U_R0(mu) + mu*log(x)]
        neg_log_x = -math.log(x_val)
        neg_log_y = -math.log(y_val)

        # Standard check (precomputed threshold for y)
        std_slack = neg_log_x - y_threshold[y_val]

        # Swapped check (compute threshold for x on the fly)
        log_x = math.log(x_val)
        swap_threshold = -math.inf
        for j in range(GRID_N):
            val = u_r0[j] + grid[j] * log_x
            if val > swap_threshold:
                swap_threshold = val
        swap_slack = neg_log_y - swap_threshold

        r0_slack = max(std_slack, swap_slack)

        if r0_slack < worst_r0_slack:
            worst_r0_slack = r0_slack
            worst_r0_lam = lam

        if r0_slack < -R0_TOL:
            return failure(
                f"R_0 check failed at lam={lam:.6e}: "
                f"std_slack={std_slack:.3e}, swap_slack={swap_slack:.3e} "
                f"(best={r0_slack:.3e})"
            )

        # Main inequality: F(lam) > -1/2 * (log X + lam*log M + lam*log Y)
        rhs = -0.5 * (math.log(x_val) + lam * math.log(m_val) + lam * math.log(y_val))
        main_slack = f_val - rhs

        if main_slack < worst_main_slack:
            worst_main_slack = main_slack
            worst_main_lam = lam

        if main_slack < -MAIN_TOL:
            return failure(
                f"Main inequality failed at lam={lam:.6e}: "
                f"F={f_val:.8f} < RHS={rhs:.8f} (slack={main_slack:.3e})"
            )

    # Compute metric: c = exp(F(1))
    p_at_1 = c1 + c2 + c3 + c4
    f_at_1 = 2.0 * math.log(2.0) + p_at_1 * math.exp(-1.0)
    growth_base_c = math.exp(f_at_1)

    if not math.isfinite(growth_base_c) or growth_base_c <= 0:
        return failure("Computed c is non-finite or non-positive")

    return success(
        f"Valid certificate; c = e^{{F(1)}} = {growth_base_c:.10f}; "
        f"worst R_0 slack = {worst_r0_slack:.3e} (at lam={worst_r0_lam:.4f}); "
        f"worst main slack = {worst_main_slack:.3e} (at lam={worst_main_lam:.4f})",
        growth_base_c=growth_base_c,
        f_at_1=f_at_1,
        worst_r0_slack=worst_r0_slack,
        worst_r0_lambda=worst_r0_lam,
        worst_main_slack=worst_main_slack,
        worst_main_lambda=worst_main_lam,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Validate Ramsey upper-bound certificate (Theorem 13, GNNW 2024)'
    )
    parser.add_argument('solution', help='Solution as JSON string or path to JSON file')
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == '__main__':
    main()
