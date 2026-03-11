#!/usr/bin/env python3
"""
Validator for problem 065: General Algorithm for Difference Bases

Test an algorithm that produces difference bases at multiple values of n.

Expected input format:
    {
        "algorithm": "description",
        "test_cases": [
            {"n": n, "basis": [b0, b1, ...]},
            ...
        ]
    }
"""

import argparse
import math
from typing import Any

from . import ValidationResult, load_solution, output_result, success, failure


def baseline_ratio(n: int) -> float:
    """Compute baseline efficiency: (2 * ceil(sqrt(n)))^2 / n."""
    return (2 * math.ceil(math.sqrt(n))) ** 2 / n


def verify_difference_basis(n: int, basis: list[int]) -> tuple[bool, int]:
    """Verify a difference basis and return (valid, size)."""
    B = set(basis)

    # Check range
    for b in B:
        if b < 0 or b >= n:
            return False, len(B)

    # Compute all differences
    differences = set()
    for a in B:
        for b in B:
            diff = abs(a - b)
            if diff > 0:
                differences.add(diff)

    # Check coverage
    missing = set(range(1, n)) - differences
    return len(missing) == 0, len(B)


def validate(solution: Any) -> ValidationResult:
    """
    Validate a general difference basis algorithm.

    Args:
        solution: Dict with algorithm and test cases

    Returns:
        ValidationResult with performance analysis
    """
    try:
        if not isinstance(solution, dict):
            return failure("Invalid format: expected dict")

        algorithm = solution.get('algorithm', 'not provided')
        test_cases = solution.get('test_cases', [])

        if not test_cases:
            return failure("Need at least one test case")

    except (ValueError, TypeError) as e:
        return failure(f"Failed to parse solution: {e}")

    results = []
    all_valid = True
    beats_baseline_count = 0

    for tc in test_cases:
        n = int(tc['n'])
        basis = [int(b) for b in tc['basis']]

        valid, size = verify_difference_basis(n, basis)
        ratio = (size ** 2) / n
        bl_ratio = baseline_ratio(n)
        beats = valid and ratio < bl_ratio

        if not valid:
            all_valid = False
        if beats:
            beats_baseline_count += 1

        results.append({
            'n': n,
            'basis_size': size,
            'ratio': ratio,
            'baseline_ratio': bl_ratio,
            'beats_baseline': beats,
            'valid': valid
        })

    if not all_valid:
        invalid = [r for r in results if not r['valid']]
        return failure(
            f"Invalid difference basis for n={invalid[0]['n']}",
            test_results=results
        )

    avg_ratio = sum(r['ratio'] for r in results) / len(results)
    metrics = dict(
        algorithm=algorithm,
        test_results=results,
        average_ratio=avg_ratio,
        beats_baseline_count=beats_baseline_count,
        num_test_cases=len(results),
    )

    if beats_baseline_count == 0:
        details = [f"n={r['n']}: ratio={r['ratio']:.4f} vs baseline={r['baseline_ratio']:.4f}" for r in results[:5]]
        return failure(
            f"Valid bases but none beat the baseline (need |B|²/n < (2*ceil(sqrt(n)))²/n). "
            f"{'; '.join(details)}",
            **metrics,
        )

    return success(
        f"Difference basis algorithm valid for all {len(results)} test cases "
        f"(avg |B|²/n: {avg_ratio:.4f}). "
        f"Beats baseline in {beats_baseline_count}/{len(results)} test cases.",
        **metrics,
    )


def main():
    parser = argparse.ArgumentParser(description='Validate general difference basis algorithm')
    parser.add_argument('solution', help='Solution as JSON string or path to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == '__main__':
    main()
