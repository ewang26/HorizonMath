#!/usr/bin/env python3
"""
Validator for problem 094: Primitive Sum of Three Cubes for n = 192

Validates that integers x, y, z satisfy:
1. x³ + y³ + z³ = 192
2. gcd(x, y, z) = 1 (primitive solution)

Expected input format:
    {"x": <int>, "y": <int>, "z": <int>}
    or [x, y, z]
"""

import argparse
from typing import Any

from . import ValidationResult, load_solution, parse_integer, gcd, output_result, success, failure


TARGET = 192


def validate(solution: Any) -> ValidationResult:
    """
    Validate that the solution satisfies x³ + y³ + z³ = 192 with gcd(x,y,z) = 1.

    Args:
        solution: Dict with keys x, y, z or list [x, y, z]

    Returns:
        ValidationResult with success/failure and computed values
    """
    try:
        if isinstance(solution, dict):
            x = parse_integer(solution['x'])
            y = parse_integer(solution['y'])
            z = parse_integer(solution['z'])
        elif isinstance(solution, (list, tuple)) and len(solution) == 3:
            x, y, z = [parse_integer(v) for v in solution]
        else:
            return failure(f"Invalid solution format: expected dict or list of 3 integers")
    except (KeyError, ValueError, TypeError) as e:
        return failure(f"Failed to parse solution: {e}")

    # Compute sum of cubes
    result = x**3 + y**3 + z**3

    if result != TARGET:
        return failure(
            f"Failed: ({x})³ + ({y})³ + ({z})³ = {result} ≠ {TARGET}",
            x=str(x), y=str(y), z=str(z), computed_sum=result, target=TARGET
        )

    # Check primitivity
    g = gcd(x, y, z)
    if g != 1:
        return failure(
            f"Not primitive: gcd({x}, {y}, {z}) = {g} ≠ 1",
            x=str(x), y=str(y), z=str(z), gcd=g
        )

    return success(
        f"Verified primitive solution: ({x})³ + ({y})³ + ({z})³ = {TARGET}, gcd = 1",
        x=str(x), y=str(y), z=str(z), sum=TARGET, gcd=1
    )


def main():
    parser = argparse.ArgumentParser(description='Validate primitive solution for sum of three cubes = 192')
    parser.add_argument('solution', help='Solution as JSON string or path to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == '__main__':
    main()
