#!/usr/bin/env python3
"""
Validator: Ramsey upper-bound certificate (split Theorem 13 validator),
with arbitrary-degree polynomial corrections.

Design:
- For 0 < lambda <= LAMBDA_SPLIT, use the fixed analytic choices
      M(lambda) = lambda * exp(-lambda)
      Y(lambda) = exp(alpha_small) * (1 - X(lambda))   if X(lambda) <= 1/2
                  1 - X(lambda) * exp(-alpha_small)    if X(lambda) > 1/2
  with alpha_small = (0.17 - 0.033) * exp(-1).
  On this interval, admissibility (X(lambda), Y(lambda)) in R is taken from
  Lemma 14 / Theorem 1. The validator still checks F > 0, F' > 0, and the
  main inequality.

- For LAMBDA_SPLIT <= lambda <= 1, the submission supplies piecewise-constant
  M and Y, and the validator checks condition (2) against the fixed inner
  approximation R_0.

The polynomial correction is now
    p(lambda) = a1*lambda + a2*lambda^2 + ... + ad*lambda^d
for any finite degree d >= 1, supplied as `polynomial_coeffs = [a1, ..., ad]`.
For backward compatibility, `correction_coeffs` is also accepted and is treated
as the same list.
"""

import argparse
from typing import Any, Sequence

import mpmath as mp

from . import ValidationResult, load_solution, output_result, success, failure

# --- Global constants ---
LAMBDA_SPLIT = mp.mpf("1e-3")
MAX_BREAKPOINTS = 500
BETA_R0 = mp.mpf("0.033")
ALPHA_SMALL = (mp.mpf("0.17") - BETA_R0) / mp.e

WORK_DPS = 100
mp.mp.dps = WORK_DPS
mp.iv.dps = WORK_DPS
iv = mp.iv

LOG_SUBDIVS_SMALL = 120
LINEAR_SUBDIVS_LARGE = 32


# ---------- piecewise parsing ----------
def validate_piecewise(data: Any, name: str) -> tuple[list[mp.mpf], list[mp.mpf], str | None]:
    if not isinstance(data, dict):
        return [], [], f"{name}: expected dict with 'breakpoints' and 'values'"

    breakpoints = data.get("breakpoints")
    values = data.get("values")

    if breakpoints is None or values is None:
        return [], [], f"{name}: missing 'breakpoints' or 'values'"
    if not isinstance(breakpoints, list) or not isinstance(values, list):
        return [], [], f"{name}: 'breakpoints' and 'values' must be lists"
    if len(values) != len(breakpoints) + 1:
        return [], [], (
            f"{name}: len(values) must be len(breakpoints)+1, "
            f"got {len(values)} vs {len(breakpoints)}"
        )
    try:
        bp_all = [mp.mpf(str(b)) for b in breakpoints]
        val_all = [mp.mpf(str(v)) for v in values]
    except Exception as e:
        return [], [], f"{name}: invalid numeric value: {e}"

    # Discard breakpoints <= LAMBDA_SPLIT (and their preceding values).
    # If breakpoints[0..k-1] are all <= LAMBDA_SPLIT, the piecewise function
    # for lambda >= LAMBDA_SPLIT starts with values[k].
    first_kept = 0
    while first_kept < len(bp_all) and bp_all[first_kept] <= LAMBDA_SPLIT:
        first_kept += 1
    bp_out = bp_all[first_kept:]
    val_out = [val_all[first_kept]] + val_all[first_kept + 1:]

    for i, b in enumerate(bp_out):
        if not mp.isfinite(b) or not (b < 1):
            return [], [], f"{name}: breakpoint {i} = {b} not in ({LAMBDA_SPLIT}, 1)"

    if len(bp_out) > MAX_BREAKPOINTS:
        return [], [], f"{name}: too many breakpoints ({len(bp_out)} > {MAX_BREAKPOINTS})"

    for i in range(len(bp_out) - 1):
        if not (bp_out[i] < bp_out[i + 1]):
            return [], [], (
                f"{name}: breakpoints not strictly increasing at {i}: "
                f"{bp_out[i]} >= {bp_out[i + 1]}"
            )

    for i, v in enumerate(val_out):
        if not mp.isfinite(v) or not (0 < v < 1):
            return [], [], f"{name}: value {i} = {v} not in (0,1)"

    return bp_out, val_out, None


def eval_piecewise_scalar(breakpoints: list[mp.mpf], values: list[mp.mpf], lam: mp.mpf) -> mp.mpf:
    for i, b in enumerate(breakpoints):
        if lam < b:
            return values[i]
    return values[-1]


# ---------- polynomial helpers ----------
def parse_polynomial_coeffs(solution: Any) -> tuple[list[mp.mpf], str | None]:
    raw = solution.get("polynomial_coeffs")
    legacy = solution.get("correction_coeffs")

    if raw is not None and legacy is not None:
        return [], "Provide only one of 'polynomial_coeffs' or 'correction_coeffs', not both"
    if raw is None:
        raw = legacy
        key = "correction_coeffs"
    else:
        key = "polynomial_coeffs"

    if not isinstance(raw, list) or len(raw) == 0:
        return [], f"'{key}' must be a nonempty list of numbers"

    try:
        coeffs = [mp.mpf(str(x)) for x in raw]
    except Exception as e:
        return [], f"Invalid polynomial coefficient: {e}"

    for i, c in enumerate(coeffs):
        if not mp.isfinite(c):
            return [], f"{key}[{i}] is not finite"

    return coeffs, None


def horner_scalar(lam: mp.mpf, coeffs: Sequence[mp.mpf]) -> mp.mpf:
    """Evaluate p(lambda) = a1*lambda + ... + ad*lambda^d with Horner's rule."""
    acc = mp.mpf("0")
    for a in reversed(coeffs):
        acc = (acc + a) * lam
    return acc


def derivative_coeffs_scalar(coeffs: Sequence[mp.mpf]) -> list[mp.mpf]:
    return [mp.mpf(i + 1) * coeffs[i] for i in range(1, len(coeffs))]


def horner_interval(lam, coeffs: Sequence[mp.mpf]):
    acc = iv.mpf(0)
    for a in reversed(coeffs):
        acc = (acc + iv.mpf(a)) * lam
    return acc


def derivative_coeffs_interval(coeffs: Sequence[mp.mpf]):
    return [mp.mpf(i + 1) * coeffs[i] for i in range(1, len(coeffs))]


# ---------- F and F' ----------
def p_scalar(lam: mp.mpf, coeffs: Sequence[mp.mpf]) -> mp.mpf:
    return horner_scalar(lam, coeffs)


def dp_scalar(lam: mp.mpf, coeffs: Sequence[mp.mpf]) -> mp.mpf:
    dcoeffs = derivative_coeffs_scalar(coeffs)
    if not dcoeffs:
        return mp.mpf("0")
    return horner_scalar(lam, dcoeffs)


def f_scalar(lam: mp.mpf, coeffs: Sequence[mp.mpf]) -> mp.mpf:
    base = (1 + lam) * mp.log(1 + lam) - lam * mp.log(lam)
    return base + p_scalar(lam, coeffs) * mp.e**(-lam)


def fp_scalar(lam: mp.mpf, coeffs: Sequence[mp.mpf]) -> mp.mpf:
    p = p_scalar(lam, coeffs)
    dp = dp_scalar(lam, coeffs) + coeffs[0]  # add constant term of p'
    return mp.log((1 + lam) / lam) + (dp - p) * mp.e**(-lam)


def p_interval(lam, coeffs: Sequence[mp.mpf]):
    return horner_interval(lam, coeffs)


def dp_interval(lam, coeffs: Sequence[mp.mpf]):
    dcoeffs = derivative_coeffs_interval(coeffs)
    if not dcoeffs:
        return iv.mpf(0)
    return horner_interval(lam, dcoeffs)


def f_interval(lo: mp.mpf, hi: mp.mpf, coeffs: Sequence[mp.mpf]):
    lam = iv.mpf([lo, hi])
    one = iv.mpf(1)
    base = (one + lam) * iv.log(one + lam) - lam * iv.log(lam)
    return base + p_interval(lam, coeffs) * iv.exp(-lam)


def fp_interval(lo: mp.mpf, hi: mp.mpf, coeffs: Sequence[mp.mpf]):
    lam = iv.mpf([lo, hi])
    one = iv.mpf(1)
    p = p_interval(lam, coeffs)
    dp = dp_interval(lam, coeffs) + iv.mpf(coeffs[0])  # add constant term of p'
    return iv.log((one + lam) / lam) + (dp - p) * iv.exp(-lam)


# ---------- R_0 boundary function ----------
def U(mu: mp.mpf) -> mp.mpf:
    g = (-mp.mpf("0.25") * mu + BETA_R0 * mu**2 + mp.mpf("0.08") * mu**3) * mp.e**(-mu)
    return g + (1 + mu) * mp.log(1 + mu) - mu * mp.log(mu)


def Up(mu: mp.mpf) -> mp.mpf:
    s = -mp.mpf("0.25") * mu + BETA_R0 * mu**2 + mp.mpf("0.08") * mu**3
    sp = -mp.mpf("0.25") + 2 * BETA_R0 * mu + mp.mpf("0.24") * mu**2
    return mp.log((1 + mu) / mu) + mp.e**(-mu) * (sp - s)


U1 = U(mp.mpf(1))
UP1 = Up(mp.mpf(1))
A1 = U1 - UP1


def A_of_mu(mu: mp.mpf) -> mp.mpf:
    return U(mu) - mu * Up(mu)


def _bracket_A(a: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
    """Bisect to find mu* where A(mu*) = a. Returns bracket [lo, hi]."""
    lo = mp.mpf("1e-60")
    hi = mp.mpf(1)
    for _ in range(200):
        mid = (lo + hi) / 2
        if A_of_mu(mid) < a:
            lo = mid
        else:
            hi = mid
    return lo, hi


def _bracket_Up(a: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
    """Bisect to find mu* where Up(mu*) = a. Returns bracket [lo, hi]."""
    lo = mp.mpf("1e-60")
    hi = mp.mpf(1)
    for _ in range(200):
        mid = (lo + hi) / 2
        if Up(mid) > a:  # Up is decreasing on (0,1]
            lo = mid
        else:
            hi = mid
    return lo, hi


def _Up_interval(mu_lo: mp.mpf, mu_hi: mp.mpf) -> mp.mpf:
    """Rigorous upper bound on Up(mu) for mu in [mu_lo, mu_hi]."""
    mu = iv.mpf([mu_lo, mu_hi])
    one = iv.mpf(1)
    s = -iv.mpf(mp.mpf("0.25")) * mu + iv.mpf(BETA_R0) * mu**2 + iv.mpf(mp.mpf("0.08")) * mu**3
    sp = -iv.mpf(mp.mpf("0.25")) + iv.mpf(2 * BETA_R0) * mu + iv.mpf(mp.mpf("0.24")) * mu**2
    result = iv.log((one + mu) / mu) + iv.exp(-mu) * (sp - s)
    return mp.mpf(result.b)  # upper bound


def _U_minus_mu_a_interval(mu_lo: mp.mpf, mu_hi: mp.mpf, a: mp.mpf) -> mp.mpf:
    """Rigorous upper bound on U(mu) - mu*a for mu in [mu_lo, mu_hi]."""
    mu = iv.mpf([mu_lo, mu_hi])
    one = iv.mpf(1)
    g = (-iv.mpf(mp.mpf("0.25")) * mu + iv.mpf(BETA_R0) * mu**2 + iv.mpf(mp.mpf("0.08")) * mu**3) * iv.exp(-mu)
    u = g + (one + mu) * iv.log(one + mu) - mu * iv.log(mu)
    result = u - mu * iv.mpf(a)
    return mp.mpf(result.b)  # upper bound


def B_of_a(a: mp.mpf) -> mp.mpf:
    """Rigorous upper bound on the symmetric R_0 boundary threshold.

    If a = -log x and b = -log y, then the pair is accepted iff
        b >= B_of_a(a).
    This accounts for symmetry: either (x,y) in R_0 or (y,x) in R_0.

    Uses interval arithmetic over bisection brackets for rigorous bounds.
    """
    if a >= U1:
        bu = mp.mpf(0)
    elif a > A1:
        bu = U1 - a
    else:
        lo, hi = _bracket_A(a)
        bu = _Up_interval(lo, hi)

    if a < UP1:
        bs = U1 - a
    else:
        lo, hi = _bracket_Up(a)
        bs = _U_minus_mu_a_interval(lo, hi, a)

    return max(mp.mpf(0), min(bu, bs))


# ---------- interval helpers ----------
def interval_lower(x) -> mp.mpf:
    return mp.mpf(x.a)


def interval_upper(x) -> mp.mpf:
    return mp.mpf(x.b)


def geometric_intervals(lo: mp.mpf, hi: mp.mpf, n: int) -> list[tuple[mp.mpf, mp.mpf]]:
    ratio = (hi / lo) ** (mp.mpf(1) / n)
    out: list[tuple[mp.mpf, mp.mpf]] = []
    a = lo
    for _ in range(n):
        b = a * ratio
        out.append((a, b))
        a = b
    out[-1] = (out[-1][0], hi)
    return out


def linear_intervals(lo: mp.mpf, hi: mp.mpf, n: int) -> list[tuple[mp.mpf, mp.mpf]]:
    out: list[tuple[mp.mpf, mp.mpf]] = []
    step = (hi - lo) / n
    a = lo
    for _ in range(n):
        b = a + step
        out.append((a, b))
        a = b
    out[-1] = (out[-1][0], hi)
    return out


# ---------- small-lambda proof machinery ----------
def polynomial_tail_bounds(coeffs: Sequence[mp.mpf], delta: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
    """For 0 < lambda <= delta <= 1 and p(lambda)=sum_{i>=1} a_i lambda^i,
    use local small-regime bounds
        |p(lambda)| <= C0(delta) * lambda,
        |p'(lambda) - p(lambda)| <= C1(delta).

    Writing i = j+1 for coeffs[j] = a_i, we have
        |a_i lambda^i| <= |a_i| delta^(i-1) lambda,
    so
        C0(delta) = sum_{i>=1} |a_i| delta^(i-1).

    Also
        p'(lambda) - p(lambda) = sum_{i>=1} a_i lambda^(i-1) (i - lambda).
    For i=1, sup_{0<lambda<=delta} |1-lambda| = 1.
    For i>=2 and delta <= 1, lambda^(i-1)(i-lambda) is increasing on (0,delta],
    so its supremum is delta^(i-1) (i-delta).
    Hence we may take
        C1(delta) = |a_1| + sum_{i>=2} |a_i| delta^(i-1) (i-delta).

    These are much sharper than the global (0,1] bounds and are sufficient because
    the analytic tail proof is only used on (0, lambda_tail] with lambda_tail <= delta.
    """
    if not (0 < delta <= 1):
        raise ValueError(f"delta must satisfy 0 < delta <= 1, got {delta}")

    C0 = mp.mpf("0")
    C1 = mp.mpf("0")
    delta_pow = mp.mpf("1")  # delta^(i-1)

    for j, a in enumerate(coeffs):
        i = j + 1
        aa = abs(a)
        C0 += aa * delta_pow
        if i == 1:
            C1 += aa
        else:
            C1 += aa * delta_pow * (i - delta)
        delta_pow *= delta

    return C0, C1


def prove_small_tail_endpoint(coeffs: Sequence[mp.mpf]) -> mp.mpf:
    """Choose a tiny lambda_tail > 0 such that on (0, lambda_tail] the theorem conditions
    follow from simple analytic inequalities depending only on the submitted coefficients.
    """
    C0, C1 = polynomial_tail_bounds(coeffs, LAMBDA_SPLIT)

    # If q = exp(-F'), M <= 1/4 and q <= 1/4, then
    #   -log X <= M/(1-M) + q/((1-M)(1-q)) <= Acoef * lambda.
    logY_floor = mp.log(1 - mp.e**(-ALPHA_SMALL))
    Acoef = mp.mpf("4") / 3 + (mp.mpf("16") / 9) * mp.e**(C1)

    candidates = [
        LAMBDA_SPLIT,
        mp.mpf("0.25"),
        mp.e**(-(C1 + mp.log(4))),  # ensures F' >= log 4, hence q <= 1/4
        mp.e**(-(C0 + 1)),          # ensures F > 0 from -lambda log lambda dominance
        mp.e**(-2 * (C0 + Acoef / 2 - logY_floor / 2 + 1)),
    ]
    return min(candidates)


def validate_analytic_small_tail(coeffs: Sequence[mp.mpf]) -> tuple[bool, str, mp.mpf]:
    """Prove the theorem conditions on (0, lambda_tail] analytically."""
    C0, C1 = polynomial_tail_bounds(coeffs, LAMBDA_SPLIT)
    logY_floor = mp.log(1 - mp.e**(-ALPHA_SMALL))
    Acoef = mp.mpf("4") / 3 + (mp.mpf("16") / 9) * mp.e**(C1)

    lam = prove_small_tail_endpoint(coeffs)

    # F(lambda) >= -lambda log lambda - C0 lambda
    f_lb = -lam * mp.log(lam) - C0 * lam
    if f_lb <= 0:
        return False, f"analytic small-tail proof failed for F at lambda={lam}", lam

    # F'(lambda) >= -log lambda - C1
    fp_lb = -mp.log(lam) - C1
    if fp_lb <= mp.log(4):
        return False, f"analytic small-tail proof failed for F' at lambda={lam}", lam

    # With M=lambda e^{-lambda} and Y >= 1-exp(-alpha_small),
    # slack >= 1/2 lambda log(1/lambda) - O(lambda).
    slack_lb = (
        -lam * mp.log(lam) - C0 * lam
        + mp.mpf("0.5") * (-Acoef * lam + lam * mp.log(lam) - lam**2 + lam * logY_floor)
    )
    if slack_lb <= 0:
        return False, f"analytic small-tail proof failed for the main inequality at lambda={lam}", lam

    return True, "", lam


def logX_interval_small(lo: mp.mpf, hi: mp.mpf, coeffs: Sequence[mp.mpf]):
    lam = iv.mpf([lo, hi])
    one = iv.mpf(1)
    m_int = lam * iv.exp(-lam)
    fp_int = fp_interval(lo, hi, coeffs)
    return iv.log(one - m_int) + iv.log(one - iv.exp(-fp_int)) / (one - m_int)


def small_branch_logY(logx_int):
    x_int = iv.exp(logx_int)
    half_log = mp.log(mp.mpf("0.5"))
    one = iv.mpf(1)

    # Entire interval in X > 1/2 branch.
    if interval_lower(logx_int) > half_log:
        return iv.log(one - x_int * mp.e**(-ALPHA_SMALL)), "upper"

    # Entire interval in X < 1/2 branch.
    if interval_upper(logx_int) < half_log:
        return ALPHA_SMALL + iv.log(one - x_int), "lower"

    return None, "split"


def check_small_interval(lo: mp.mpf, hi: mp.mpf,
                         coeffs: Sequence[mp.mpf],
                         depth: int = 0) -> tuple[bool, str, mp.mpf]:
    if depth > 40:
        return False, f"small-lambda branch ambiguity persisted on [{lo}, {hi}]", mp.inf

    f_int = f_interval(lo, hi, coeffs)
    fp_int = fp_interval(lo, hi, coeffs)

    if interval_lower(f_int) <= 0:
        return False, f"F(lambda) <= 0 somewhere on [{lo}, {hi}]", interval_lower(f_int)
    if interval_lower(fp_int) <= 0:
        return False, f"F'(lambda) <= 0 somewhere on [{lo}, {hi}]", interval_lower(fp_int)

    lam = iv.mpf([lo, hi])
    logx_int = logX_interval_small(lo, hi, coeffs)
    logy_int, branch = small_branch_logY(logx_int)

    if logy_int is None:
        mid = mp.sqrt(lo * hi)
        ok, msg, val = check_small_interval(lo, mid, coeffs, depth + 1)
        if not ok:
            return ok, msg, val
        return check_small_interval(mid, hi, coeffs, depth + 1)

    # The theorem requires Y in (0,1). In the lower branch Y = exp(alpha)*(1-X),
    # which can exceed 1 when X is very small. Reject if Y >= 1 anywhere.
    if interval_upper(logy_int) >= 0:
        return False, f"small-lambda Y(lambda) >= 1 somewhere on [{lo}, {hi}]", interval_upper(logy_int)

    half = iv.mpf(mp.mpf("0.5"))
    # Since M(lambda) = lambda e^{-lambda}, we have lambda log M = lambda(log lambda - lambda).
    slack_int = f_int + half * (logx_int + lam * (iv.log(lam) - lam) + lam * logy_int)

    if interval_lower(slack_int) <= 0:
        return False, f"main inequality failed somewhere on [{lo}, {hi}]", interval_lower(slack_int)

    return True, branch, interval_lower(slack_int)


# ---------- large-lambda checks ----------
def check_large_interval(lo: mp.mpf, hi: mp.mpf,
                         coeffs: Sequence[mp.mpf],
                         m_const: mp.mpf, y_const: mp.mpf) -> tuple[bool, str, mp.mpf, mp.mpf]:
    f_int = f_interval(lo, hi, coeffs)
    fp_int = fp_interval(lo, hi, coeffs)

    if interval_lower(f_int) <= 0:
        return False, f"F(lambda) <= 0 somewhere on [{lo}, {hi}]", mp.inf, mp.inf
    if interval_lower(fp_int) <= 0:
        return False, f"F'(lambda) <= 0 somewhere on [{lo}, {hi}]", mp.inf, mp.inf

    lam = iv.mpf([lo, hi])
    one = iv.mpf(1)
    m_iv = iv.mpf(m_const)

    # log X = log(1-M) + (1/(1-M)) log(1 - exp(-F')).
    logx_int = iv.log(one - m_iv) + iv.log(one - iv.exp(-fp_int)) / (one - m_iv)
    a_lo = -interval_upper(logx_int)  # smallest possible a = -log X on this lambda-interval

    if a_lo <= 0:
        return False, f"X(lambda) >= 1 somewhere on [{lo}, {hi}]", mp.inf, mp.inf

    # Since B(a) is nonincreasing, b - B(a_lo) is a lower bound for the R_0 margin.
    # Use interval log for rigorous lower bound on b = -log(Y).
    b_const = -interval_upper(iv.log(iv.mpf(y_const)))
    r0_margin = b_const - B_of_a(a_lo)
    if r0_margin <= 0:
        return False, f"R_0 check failed somewhere on [{lo}, {hi}]", r0_margin, mp.inf

    half = iv.mpf(mp.mpf("0.5"))
    log_m_iv = iv.log(iv.mpf(m_const))
    log_y_iv = iv.log(iv.mpf(y_const))
    slack_int = f_int + half * (logx_int + lam * log_m_iv + lam * log_y_iv)
    if interval_lower(slack_int) <= 0:
        return False, f"main inequality failed somewhere on [{lo}, {hi}]", r0_margin, interval_lower(slack_int)

    return True, "", r0_margin, interval_lower(slack_int)


# ---------- main validator ----------
def validate(solution: Any) -> ValidationResult:
    if not isinstance(solution, dict):
        return failure("Invalid format: expected dict")

    coeffs, err = parse_polynomial_coeffs(solution)
    if err:
        return failure(err)

    m_data = solution.get("M")
    if m_data is None:
        return failure("Missing 'M'")
    m_bp, m_vals, err = validate_piecewise(m_data, "M")
    if err:
        return failure(err)

    y_data = solution.get("Y")
    if y_data is None:
        return failure("Missing 'Y'")
    y_bp, y_vals, err = validate_piecewise(y_data, "Y")
    if err:
        return failure(err)

    # Small-lambda analytic tail near 0.
    ok, msg, tail = validate_analytic_small_tail(coeffs)
    if not ok:
        return failure(msg)

    worst_small_slack = mp.inf
    worst_large_r0_slack = mp.inf
    worst_large_main_slack = mp.inf

    # Interval proof on [tail, LAMBDA_SPLIT] using the fixed analytic small-lambda model.
    for lo, hi in geometric_intervals(tail, LAMBDA_SPLIT, LOG_SUBDIVS_SMALL):
        ok, msg, slack_lb = check_small_interval(lo, hi, coeffs)
        if not ok:
            return failure(msg)
        worst_small_slack = min(worst_small_slack, slack_lb)

    # Large-lambda partition: refine along the union of the M and Y breakpoints.
    large_edges = sorted(set(m_bp) | set(y_bp) | {mp.mpf(1)})
    left = LAMBDA_SPLIT
    for right in large_edges:
        # No breakpoint lies inside (left, right), so M and Y are constant there.
        sample = mp.sqrt(left * right) if right / left > mp.mpf("1.2") else (left + right) / 2
        m_const = eval_piecewise_scalar(m_bp, m_vals, sample)
        y_const = eval_piecewise_scalar(y_bp, y_vals, sample)

        for lo, hi in linear_intervals(left, right, LINEAR_SUBDIVS_LARGE):
            ok, msg, r0_lb, slack_lb = check_large_interval(lo, hi, coeffs, m_const, y_const)
            if not ok:
                return failure(msg)
            worst_large_r0_slack = min(worst_large_r0_slack, r0_lb)
            worst_large_main_slack = min(worst_large_main_slack, slack_lb)

        left = right

    f_at_1 = f_scalar(mp.mpf(1), coeffs)
    growth_base_c = mp.e**f_at_1

    if not mp.isfinite(growth_base_c) or growth_base_c <= 0:
        return failure("Computed c is non-finite or non-positive")

    coeff_key = "polynomial_coeffs" if solution.get("polynomial_coeffs") is not None else "correction_coeffs"

    return success(
        f"Valid split certificate; c = e^{{F(1)}} = {mp.nstr(growth_base_c, 12)}; "
        f"{coeff_key} degree = {len(coeffs)}; "
        f"analytic tail endpoint = {mp.nstr(tail, 6)}; "
        f"worst small-lambda slack = {mp.nstr(worst_small_slack, 6)}; "
        f"worst large R_0 slack = {mp.nstr(worst_large_r0_slack, 6)}; "
        f"worst large main slack = {mp.nstr(worst_large_main_slack, 6)}",
        growth_base_c=float(growth_base_c),
        f_at_1=float(f_at_1),
        lambda_split=float(LAMBDA_SPLIT),
        polynomial_degree=len(coeffs),
        small_tail_endpoint=mp.nstr(tail, 20),
        worst_small_slack=mp.nstr(worst_small_slack, 20),
        worst_large_r0_slack=mp.nstr(worst_large_r0_slack, 20),
        worst_large_main_slack=mp.nstr(worst_large_main_slack, 20),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Validate Ramsey upper-bound certificate (split Theorem 13 validator; arbitrary polynomial degree)"
    )
    parser.add_argument("solution", help="Solution as JSON string or path to JSON file")
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == "__main__":
    main()
