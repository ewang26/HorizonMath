#!/usr/bin/env python3
"""
Validator for problem 061: Heilbronn Configuration for n=12

The Heilbronn problem asks to place n points in [0,1]² to maximize
the minimum area of any triangle formed by three points.

For n=12, this validator:
1. Checks all points are in [0,1]²
2. Computes the minimum triangle area over all (n choose 3) triangles
3. Reports the configuration quality

Expected input format:
    {"points": [[x, y], ...]}  12 points in [0,1]²
    or [[x, y], ...]
"""

import argparse
from itertools import combinations
from typing import Any

import numpy as np

from . import ValidationResult, load_solution, output_result, success, failure


TARGET_N = 12
TOLERANCE = 1e-9


def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute area of triangle using cross product formula."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def validate(solution: Any) -> ValidationResult:
    """
    Validate a Heilbronn configuration for n=12.

    Args:
        solution: Dict with 'points' key or list of 12 2D points

    Returns:
        ValidationResult with minimum triangle area
    """
    try:
        if isinstance(solution, dict) and 'points' in solution:
            points_data = solution['points']
        elif isinstance(solution, list):
            points_data = solution
        else:
            return failure("Invalid format: expected dict with 'points' or list")

        points = np.array(points_data, dtype=np.float64)
    except (ValueError, TypeError) as e:
        return failure(f"Failed to parse points: {e}")

    if points.ndim != 2:
        return failure(f"Points must be 2D array, got {points.ndim}D")

    n, d = points.shape
    if d != 2:
        return failure(f"Points must be in ℝ², got dimension {d}")

    if n != TARGET_N:
        return failure(f"Expected {TARGET_N} points, got {n}")

    # Check all points are in [0,1]²
    if np.any(points < -TOLERANCE) or np.any(points > 1 + TOLERANCE):
        out_of_bounds = np.sum((points < -TOLERANCE) | (points > 1 + TOLERANCE))
        return failure(
            f"Points must be in [0,1]², found {out_of_bounds} out-of-bounds coordinates"
        )

    # Compute minimum triangle area
    min_area = float('inf')
    min_triangle = (0, 1, 2)

    for i, j, k in combinations(range(n), 3):
        area = triangle_area(points[i], points[j], points[k])
        if area < min_area:
            min_area = area
            min_triangle = (i, j, k)

    # Check for collinear points (degenerate triangles)
    if min_area < TOLERANCE:
        return failure(
            f"Points {min_triangle} are collinear (area ≈ 0)",
            min_area=min_area
        )

    return success(
        f"Heilbronn configuration for n={n}: minimum triangle area = {min_area:.10f}",
        num_points=n,
        min_triangle_area=min_area,
        worst_triangle=list(min_triangle)
    )


def main():
    parser = argparse.ArgumentParser(description='Validate Heilbronn configuration for n=12')
    parser.add_argument('solution', help='Solution as JSON string or path to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == '__main__':
    main()
