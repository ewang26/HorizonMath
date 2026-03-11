#!/usr/bin/env python3
"""
Validator for problem 052: Smaller Kakeya Set in 𝔽_p³

A Kakeya set in 𝔽_p³ contains a line in every direction.
The goal is to find a smaller such set.

Expected input format:
    {
        "p": prime,
        "points": [[x, y, z], ...]  # Points in F_p³
    }
"""

import argparse
from typing import Any

from . import ValidationResult, load_solution, output_result, success, failure


def validate(solution: Any) -> ValidationResult:
    """
    Validate a Kakeya set in F_p³.

    Args:
        solution: Dict with prime p and list of points

    Returns:
        ValidationResult with size and coverage verification
    """
    try:
        if not isinstance(solution, dict):
            return failure("Invalid format: expected dict with 'p' and 'points'")

        p = int(solution['p'])
        points = solution['points']

        if p < 2:
            return failure(f"p must be at least 2, got {p}")

        # Convert points to tuples for set operations
        point_set = set()
        for pt in points:
            if len(pt) != 3:
                return failure(f"Points must be 3D, got {len(pt)}D")
            x, y, z = int(pt[0]) % p, int(pt[1]) % p, int(pt[2]) % p
            point_set.add((x, y, z))

    except (KeyError, ValueError, TypeError) as e:
        return failure(f"Failed to parse solution: {e}")

    # Check that set contains a line in every direction
    # Directions in P²(F_p) have p² + p + 1 elements
    # Represented as [a:b:c] with normalization

    def get_directions(p):
        """Generate all projective directions in P²(F_p)."""
        directions = []
        # [1:b:c] for all b,c
        for b in range(p):
            for c in range(p):
                directions.append((1, b, c))
        # [0:1:c] for all c
        for c in range(p):
            directions.append((0, 1, c))
        # [0:0:1]
        directions.append((0, 0, 1))
        return directions

    directions = get_directions(p)
    num_directions = len(directions)  # Should be p² + p + 1

    missing_directions = []
    for d in directions:
        a, b, c = d
        # Find a line in direction (a, b, c) contained in point_set
        # A line is {(x₀ + t*a, y₀ + t*b, z₀ + t*c) : t ∈ F_p}
        found_line = False

        for pt in point_set:
            x0, y0, z0 = pt
            # Check if entire line through pt in direction d is in set
            line_in_set = True
            for t in range(p):
                line_pt = (
                    (x0 + t * a) % p,
                    (y0 + t * b) % p,
                    (z0 + t * c) % p
                )
                if line_pt not in point_set:
                    line_in_set = False
                    break
            if line_in_set:
                found_line = True
                break

        if not found_line:
            missing_directions.append(d)

    if missing_directions:
        sample = missing_directions[:3]
        return failure(
            f"Missing lines in {len(missing_directions)} directions. Examples: {sample}",
            missing_count=len(missing_directions),
            total_directions=num_directions
        )

    size = len(point_set)
    density = size / (p ** 3)

    return success(
        f"Valid Kakeya set in F_{p}³: {size} points ({density*100:.2f}% density), "
        f"contains line in all {num_directions} directions",
        prime=p,
        size=size,
        density=density,
        num_directions=num_directions
    )


def main():
    parser = argparse.ArgumentParser(description='Validate Kakeya set in F_p^3')
    parser.add_argument('solution', help='Solution as JSON string or path to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == '__main__':
    main()
