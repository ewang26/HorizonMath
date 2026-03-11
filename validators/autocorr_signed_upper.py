#!/usr/bin/env python3
"""
Validator for problem 040: Signed Autocorrelation Constant C' Upper Bound

The signed autocorrelation constant C' is defined as:
    C' = inf_f max_t (f*f)(t) / (∫f)^2
where f may be positive or negative (not restricted to be non-negative)
and is supported on [-1/4, 1/4].

The current best upper bound is C' ≤ 1.4557 (AlphaEvolve, 2025).

The model provides a step function as a list of real values on N equal-width
subintervals of [-1/4, 1/4]. The validator computes the autoconvolution ratio
and checks if it improves the best known upper bound.

Expected input format:
    {"values": [v_0, v_1, ..., v_{N-1}]}
    or [v_0, v_1, ..., v_{N-1}]
"""

import argparse
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from . import ValidationResult, load_solution, output_result, success, failure


BEST_KNOWN_UPPER = 1.4557
MIN_INTERVALS = 10
MAX_INTERVALS = 1_000_000


def compute_autoconvolution_ratio(values: np.ndarray) -> float:
    """
    Compute max_t (f*f)(t) / (∫f)^2 for a step function.

    The function f is defined on N equal-width subintervals of [-1/4, 1/4].
    Each subinterval has width h = (1/2) / N.

    The autoconvolution (f*f)(t) = ∫ f(t-x)f(x) dx is computed via
    discrete convolution of the step function values scaled by the
    subinterval width h.

    This is equivalent to the AlphaEvolve evaluator formula:
        score = 2n * max(convolve(a, a)) / (sum(a))^2
    since h = 1/(2n), so max(conv)*h / (sum(a)*h)^2
         = max(conv) / (sum(a)^2 * h) = 2n * max(conv) / sum(a)^2.
    """
    n = len(values)
    h = 0.5 / n  # width of each subinterval

    # Discrete convolution: (f*f) sampled at points spaced by h
    # fftconvolve gives the convolution of the coefficient sequences;
    # multiply by h to account for the integral approximation
    conv = fftconvolve(values, values) * h

    max_conv = np.max(conv)
    integral_f = np.sum(values) * h

    if integral_f == 0:
        return float('inf')

    return max_conv / (integral_f ** 2)


def validate(solution: Any) -> ValidationResult:
    """
    Validate a signed autocorrelation upper bound construction.

    Args:
        solution: Dict with 'values' key or list of real values

    Returns:
        ValidationResult with autoconvolution ratio
    """
    try:
        if isinstance(solution, dict) and 'values' in solution:
            values_data = solution['values']
        elif isinstance(solution, list):
            values_data = solution
        else:
            return failure("Invalid format: expected dict with 'values' or list")

        values = np.array(values_data, dtype=np.float64)
    except (ValueError, TypeError) as e:
        return failure(f"Failed to parse values: {e}")

    if values.ndim != 1:
        return failure(f"Values must be a 1D array, got {values.ndim}D")

    n = len(values)
    if n < MIN_INTERVALS:
        return failure(f"Need at least {MIN_INTERVALS} intervals, got {n}")

    if n > MAX_INTERVALS:
        return failure(f"Too many intervals ({n}), maximum is {MAX_INTERVALS}")

    # Check all entries are finite reals (reject NaN/inf)
    if not np.all(np.isfinite(values)):
        return failure("All values must be finite real numbers (no NaN or inf)")

    # Check function is not identically zero
    if np.all(values == 0):
        return failure("Function is identically zero")

    # Check sum is nonzero (otherwise the ratio is undefined)
    if np.sum(values) == 0:
        return failure("Sum of values is zero (autoconvolution ratio is undefined)")

    ratio = compute_autoconvolution_ratio(values)

    if not np.isfinite(ratio):
        return failure(
            "Computed ratio is not finite, indicating a numerical issue",
            autoconvolution_ratio=float(ratio)
        )

    return success(
        f"Step function with {n} intervals achieves autoconvolution ratio {ratio:.6f} "
        f"(best known: {BEST_KNOWN_UPPER})",
        num_intervals=n,
        autoconvolution_ratio=ratio,
        best_known_upper=BEST_KNOWN_UPPER,
        improves_bound=ratio < BEST_KNOWN_UPPER
    )


def main():
    parser = argparse.ArgumentParser(
        description='Validate signed autocorrelation upper bound construction'
    )
    parser.add_argument('solution', help='Solution as JSON string or path to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == '__main__':
    main()
