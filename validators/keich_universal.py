#!/usr/bin/env python3
"""
Validator for problem 058: Universal Formula Improving Keich's Construction

Keich's construction gives a Kakeya set with area tending to 0 as n → ∞.
This problem asks for a universal formula that improves upon Keich's approach.

Expected input format:
    {
        "formula": "description of the construction",
        "test_cases": [
            {"n": 64, "area": computed_area},
            {"n": 128, "area": computed_area},
            ...
        ]
    }
"""

import argparse
from typing import Any

from . import ValidationResult, load_solution, output_result, success, failure


def keich_bound(n: int) -> float:
    """Compute Keich's construction area bound for n directions."""
    # Keich's construction: area ≈ π/8 * (1/log n)
    # This is a simplified bound
    import math
    if n <= 1:
        return float('inf')
    return math.pi / 8 / math.log(n)


def validate(solution: Any) -> ValidationResult:
    """
    Validate a universal Kakeya construction formula.

    Args:
        solution: Dict with formula description and test cases

    Returns:
        ValidationResult with comparison to Keich bound
    """
    try:
        if not isinstance(solution, dict):
            return failure("Invalid format: expected dict")

        formula = solution.get('formula', 'not provided')
        test_cases = solution.get('test_cases', [])

        if not test_cases:
            return failure("Need at least one test case to validate")

    except (ValueError, TypeError) as e:
        return failure(f"Failed to parse solution: {e}")

    # Verify each test case improves on Keich
    improvements = []
    for tc in test_cases:
        n = int(tc['n'])
        area = float(tc['area'])

        keich_area = keich_bound(n)
        improvement = (keich_area - area) / keich_area * 100

        improvements.append({
            'n': n,
            'area': area,
            'keich_bound': keich_area,
            'improvement_percent': improvement
        })

    # Check if all cases improve on Keich
    all_improve = all(imp['improvement_percent'] > 0 for imp in improvements)

    if all_improve:
        avg_improvement = sum(imp['improvement_percent'] for imp in improvements) / len(improvements)
        return success(
            f"Universal formula improves on Keich by avg {avg_improvement:.2f}%",
            formula=formula,
            test_cases=improvements,
            average_improvement=avg_improvement
        )
    else:
        failing = [imp for imp in improvements if imp['improvement_percent'] <= 0]
        return failure(
            f"Does not improve on Keich for n={failing[0]['n']}",
            test_cases=improvements
        )


def main():
    parser = argparse.ArgumentParser(description='Validate universal Kakeya construction')
    parser.add_argument('solution', help='Solution as JSON string or path to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == '__main__':
    main()
