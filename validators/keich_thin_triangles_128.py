#!/usr/bin/env python3
"""
Validator for problem keich_thin_triangles_128

We fix:
  N = 128
  delta = 1/128
  slopes a_i = i/128, i=0..127

A solution provides intercepts b_i defining lines y = a_i x + b_i on x in [0,1].
Each defines a thin triangle R_delta(l_i) whose vertical cross-section at x is
  [a_i x + b_i - delta*(1-x), a_i x + b_i]

We score by Area(E) where E = union_i R_delta(l_i).

Baseline: Keich's Theorem 1 construction for N=2^n slopes, instantiated at n=7,
gives intercepts via the formula
  b(l_k) = sum_{i=1}^{n} ((1-i)/n) * eps_i(k) * 2^{-i}
where eps_i(k) are the binary digits of k/2^n. The exact area of that construction
is 191403/1605632 ≈ 0.11920726542570154.

See: https://www.cs.cornell.edu/~keich/papers/Kakeya.pdf (Theorem 1, property (i)).
"""

import argparse
import math
from typing import Any, List, Tuple

from . import ValidationResult, load_solution, output_result, success, failure

N = 128
DELTA = 1.0 / 128.0
BASELINE = 191403.0 / 1605632.0  # exact Keich n=7 area


def _union_length(intervals: List[Tuple[float, float]]) -> float:
    """Compute length of union of closed intervals [l,r] with l<=r."""
    if not intervals:
        return 0.0
    intervals.sort(key=lambda t: t[0])
    total = 0.0
    cur_l, cur_r = intervals[0]
    for l, r in intervals[1:]:
        if l > cur_r:
            total += (cur_r - cur_l)
            cur_l, cur_r = l, r
        else:
            if r > cur_r:
                cur_r = r
    total += (cur_r - cur_l)
    return total


def _union_length_at(bs: List[float], x: float) -> float:
    """Compute union length of thin-triangle cross-sections at position x."""
    one_minus_x = 1.0 - x
    intervals = []
    for i, b in enumerate(bs):
        a = i / 128.0
        top = a * x + b
        bot = top - DELTA * one_minus_x
        intervals.append((bot, top))
    return _union_length(intervals)


def _exact_area_from_intercepts(bs: List[float]) -> float:
    """
    Compute exact area of union of thin triangles via piecewise-linear integration.

    Each line i defines an interval at position x:
      top_i(x) = a_i * x + b_i
      bot_i(x) = (a_i + delta) * x + (b_i - delta)

    All 256 endpoint functions are linear in x. The union length is piecewise-
    linear, changing slope only at x-values where two endpoint functions cross.
    Between crossings, the trapezoid rule is exact.
    """
    n = len(bs)
    delta = 1.0 / n

    # Build linear functions: f(x) = slope * x + const
    # top_i(x) = a_i * x + b_i
    # bot_i(x) = (a_i + delta) * x + (b_i - delta)
    slopes = []
    consts = []
    for i in range(n):
        a_i = i / n
        slopes.append(a_i)
        consts.append(bs[i])
        slopes.append(a_i + delta)
        consts.append(bs[i] - delta)

    # Find all crossings in (0, 1)
    crossings = [0.0, 1.0]
    nf = len(slopes)
    for j in range(nf):
        for k in range(j + 1, nf):
            ds = slopes[j] - slopes[k]
            if abs(ds) < 1e-15:
                continue
            x_cross = (consts[k] - consts[j]) / ds
            if 0.0 < x_cross < 1.0:
                crossings.append(x_cross)

    crossings.sort()
    # Remove near-duplicates
    unique = [crossings[0]]
    for x in crossings[1:]:
        if x - unique[-1] > 1e-14:
            unique.append(x)
    crossings = unique

    # Integrate using trapezoid rule (exact for piecewise-linear)
    area = 0.0
    prev_x = crossings[0]
    prev_len = _union_length_at(bs, prev_x)
    for x in crossings[1:]:
        cur_len = _union_length_at(bs, x)
        area += 0.5 * (prev_len + cur_len) * (x - prev_x)
        prev_x = x
        prev_len = cur_len

    return area


def validate(solution: Any) -> ValidationResult:
    try:
        if not isinstance(solution, dict):
            return failure("Invalid format: expected dict")

        if "intercepts" not in solution:
            return failure("Missing key 'intercepts'")

        bs = solution["intercepts"]
        if not isinstance(bs, list):
            return failure("'intercepts' must be a list")

        if len(bs) != N:
            return failure(f"Expected {N} intercepts, got {len(bs)}")

        # Convert to floats and sanity-check for NaN/inf
        bs_f: List[float] = []
        for j, v in enumerate(bs):
            if not isinstance(v, (int, float)):
                return failure(f"Intercept {j} is not a number")
            f = float(v)
            if not math.isfinite(f):
                return failure(f"Intercept {j} is not finite")
            bs_f.append(f)

        area = _exact_area_from_intercepts(bs_f)

        return success(
            f"Valid. Union area={area:.15f}, baseline={BASELINE:.15f}.",
            area=float(area),
            baseline=float(BASELINE),
            N=N,
            delta=float(DELTA),
        )

    except Exception as e:
        return failure(f"Validation error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Validate thin-triangle Kakeya (N=128) construction")
    parser.add_argument("solution", help="Solution as JSON string or path to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    sol = load_solution(args.solution)
    res = validate(sol)
    output_result(res)


if __name__ == "__main__":
    main()
