#!/usr/bin/env python3
"""
Validator for problem 045: 2-Coloring with No Monochromatic 7-Term Arithmetic Progression

Find a 2-coloring of {0, 1, ..., n-1} that avoids monochromatic arithmetic
progressions of length 7, maximizing n.

Expected input format:
    {
        "coloring": [0, 1, 0, 1, ...]  # list of 0/1 values, one per element
    }
"""

import argparse
from typing import Any

from . import ValidationResult, load_solution, output_result, success, failure


AP_LENGTH = 7


def validate(solution: Any) -> ValidationResult:
    """
    Validate a 2-coloring of {0,...,n-1} has no monochromatic 7-AP.

    Args:
        solution: Dict with 'coloring' (list of 0/1 values)

    Returns:
        ValidationResult with verification status and length metric
    """
    try:
        if not isinstance(solution, dict):
            return failure("Invalid format: expected dict")

        if 'coloring' not in solution:
            return failure("Missing 'coloring' key")

        coloring = list(solution['coloring'])
        n = len(coloring)

        if n == 0:
            return failure("Coloring is empty")

        # Validate entries are 0 or 1
        for i, c in enumerate(coloring):
            if c not in (0, 1):
                return failure(f"coloring[{i}] = {c}, expected 0 or 1")

    except (ValueError, TypeError) as e:
        return failure(f"Failed to parse solution: {e}")

    # Check all 7-term arithmetic progressions a, a+d, a+2d, ..., a+6d
    for d in range(1, (n - 1) // (AP_LENGTH - 1) + 1):
        for a in range(n - (AP_LENGTH - 1) * d):
            color = coloring[a]
            mono = True
            for k in range(1, AP_LENGTH):
                if coloring[a + k * d] != color:
                    mono = False
                    break
            if mono:
                ap = [a + k * d for k in range(AP_LENGTH)]
                return failure(
                    f"Monochromatic {AP_LENGTH}-AP found: {ap} all color {color}"
                )

    return success(
        f"Valid 2-coloring of {{0,...,{n-1}}} with no monochromatic {AP_LENGTH}-AP",
        length=n,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Validate 2-coloring avoiding monochromatic 7-AP'
    )
    parser.add_argument('solution', help='Solution as JSON string or path to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == '__main__':
    main()
