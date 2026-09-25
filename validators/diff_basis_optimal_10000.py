#!/usr/bin/env python3
"""
Validator for problem 042: Optimal Difference Basis Construction for n=10000

A set B ⊆ {0, 1, ..., n-1} is a difference basis for [1, n-1] if every integer
in [1, n-1] can be written as |a - b| for some a, b ∈ B.

For n=10000, the goal is to minimize |B|.

Expected input format:
    {"basis": [b0, b1, b2, ...]}
    or [b0, b1, b2, ...]
"""

import argparse
from typing import Any

from . import ValidationResult, load_solution, parse_integer, output_result, success, failure


TARGET_N = 10000


def validate(solution: Any) -> ValidationResult:
    """
    Validate a difference basis for n=10000.

    Args:
        solution: Dict with 'basis' key or list of basis elements

    Returns:
        ValidationResult with success/failure and basis size
    """
    try:
        if isinstance(solution, dict) and 'basis' in solution:
            basis = [parse_integer(b) for b in solution['basis']]
        elif isinstance(solution, list):
            basis = [parse_integer(b) for b in solution]
        else:
            return failure("Invalid format: expected dict with 'basis' key or list")
    except (ValueError, TypeError) as e:
        return failure(f"Failed to parse solution: {e}")

    n = TARGET_N
    B = set(basis)

    # Check all elements are in valid range
    for b in B:
        if b < 0 or b >= n:
            return failure(f"Basis element {b} not in range [0, {n-1}]")

    # Compute all differences
    differences = set()
    for a in B:
        for b in B:
            diff = abs(a - b)
            if diff > 0:
                differences.add(diff)

    # Check coverage of [1, n-1]
    missing = set(range(1, n)) - differences
    if missing:
        sample = sorted(list(missing))[:5]
        return failure(
            f"Not a difference basis: missing {len(missing)} values. Examples: {sample}",
            missing_count=len(missing)
        )

    size = len(B)
    ratio = (size ** 2) / n

    return success(
        f"Verified difference basis for n={n}: |B|={size}, |B|²/n = {ratio:.6f}",
        n=n, basis_size=size, ratio=ratio
    )


def main():
    parser = argparse.ArgumentParser(description='Validate difference basis for n=10000')
    parser.add_argument('solution', help='Solution as JSON string or path to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == '__main__':
    main()
