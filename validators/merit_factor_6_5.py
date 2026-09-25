#!/usr/bin/env python3
"""
Validator for problem 043: Polynomial with Maximum Merit Factor

A binary polynomial has coefficients ±1. The merit factor of a polynomial
p(z) = Σᵢ aᵢzⁱ is defined as:
    F = n² / (2·Σₖ Cₖ²)
where Cₖ = Σᵢ aᵢ·aᵢ₊ₖ is the aperiodic autocorrelation at lag k.

The goal is to find a polynomial of length n ≥ 100 with merit factor > 9.5851.
Short sequences can achieve high merit factors trivially (e.g. Barker sequences),
so a minimum length is required to ensure the result is meaningful evidence
toward the asymptotic merit factor problem.

Expected input format:
    {"coefficients": [a₀, a₁, ..., aₙ₋₁]}  where each aᵢ ∈ {-1, 1}
    or [a₀, a₁, ..., aₙ₋₁]
"""

import argparse
from typing import Any

from . import ValidationResult, load_solution, output_result, success, failure

MIN_LENGTH = 100
THRESHOLD = 9.5851


def compute_merit_factor(coeffs: list[int]) -> float:
    """Compute the merit factor of a binary polynomial."""
    n = len(coeffs)
    if n <= 1:
        return 0.0

    # Compute aperiodic autocorrelations
    # C_k = sum_{i=0}^{n-1-k} a_i * a_{i+k}
    autocorr_sum = 0.0
    for k in range(1, n):
        c_k = sum(coeffs[i] * coeffs[i + k] for i in range(n - k))
        autocorr_sum += c_k ** 2

    if autocorr_sum == 0:
        return float('inf')

    return (n ** 2) / (2 * autocorr_sum)


def validate(solution: Any) -> ValidationResult:
    """
    Validate a binary polynomial of length >= 100 has merit factor > 9.5851.

    Args:
        solution: Dict with 'coefficients' key or list of ±1 values

    Returns:
        ValidationResult with success/failure and computed merit factor
    """
    try:
        if isinstance(solution, dict) and 'coefficients' in solution:
            coeffs = solution['coefficients']
        elif isinstance(solution, list):
            coeffs = solution
        else:
            return failure("Invalid format: expected dict with 'coefficients' or list")

        coeffs = [int(c) for c in coeffs]
    except (ValueError, TypeError) as e:
        return failure(f"Failed to parse coefficients: {e}")

    n = len(coeffs)
    if n < MIN_LENGTH:
        return failure(
            f"Sequence length {n} is below the minimum required length {MIN_LENGTH}",
            length=n
        )

    # Check all coefficients are ±1
    invalid = [c for c in coeffs if c not in (-1, 1)]
    if invalid:
        return failure(f"Coefficients must be ±1, found invalid values: {invalid[:5]}")

    merit = compute_merit_factor(coeffs)

    if merit < THRESHOLD:
        return failure(
            f"Merit factor {merit:.6f} is below required threshold {THRESHOLD}",
            length=n,
            merit_factor=merit
        )

    return success(
        f"Valid polynomial of length {n} with merit factor {merit:.6f} > {THRESHOLD}",
        length=n,
        merit_factor=merit
    )


def main():
    parser = argparse.ArgumentParser(description='Validate polynomial with maximum merit factor')
    parser.add_argument('solution', help='Solution as JSON string or path to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == '__main__':
    main()
