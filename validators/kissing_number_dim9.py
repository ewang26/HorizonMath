#!/usr/bin/env python3
"""
Validator for problem 050: Kissing Number in Dimension 9

The kissing number τ₉ is the maximum number of non-overlapping unit spheres
that can touch a central unit sphere in 9 dimensions.

Known bounds: 306 ≤ τ₉ ≤ 380

This validator checks that:
1. All points are on the unit sphere S⁸ (|x| = 1)
2. All pairwise dot products are ≤ 1/2 (equivalently, distances ≥ 1)
3. Counts the number of valid points

Expected input format:
    {"points": [[x₁, ...], [x₁, ...], ...]}  each point in ℝ⁹
    or [[x₁, ...], [x₁, ...], ...]
"""

import argparse
from typing import Any

import numpy as np

from . import ValidationResult, load_solution, output_result, success, failure


DIMENSION = 9
MIN_CONTACT_DISTANCE = 1.0  # minimum distance between contact points on the unit sphere
TOLERANCE = 1e-9


def validate(solution: Any) -> ValidationResult:
    """
    Validate a kissing configuration in dimension 9.

    Args:
        solution: Dict with 'points' key or list of points

    Returns:
        ValidationResult with point count and minimum distance
    """
    try:
        if isinstance(solution, dict) and 'points' in solution:
            points_data = solution['points']
        elif isinstance(solution, list):
            points_data = solution
        else:
            return failure("Invalid format: expected dict with 'points' or list of points")

        points = np.array(points_data, dtype=np.float64)
    except (ValueError, TypeError) as e:
        return failure(f"Failed to parse points: {e}")

    if points.ndim != 2:
        return failure(f"Points must be 2D array, got {points.ndim}D")

    n, d = points.shape
    if d != DIMENSION:
        return failure(f"Points must be in ℝ⁹, got dimension {d}")

    if n == 0:
        return failure("No points provided")

    # Check all points are on unit sphere
    norms = np.linalg.norm(points, axis=1)
    off_sphere = np.abs(norms - 1.0) > TOLERANCE
    if np.any(off_sphere):
        worst_idx = np.argmax(np.abs(norms - 1.0))
        return failure(
            f"Point {worst_idx} not on unit sphere: |x| = {norms[worst_idx]:.10f}",
            off_sphere_count=int(np.sum(off_sphere))
        )

    # Check pairwise dot products ≤ 1/2 (equivalently, distances ≥ 1)
    # Use the Gram matrix for efficiency and numerical clarity
    gram = points @ points.T
    min_dist = float('inf')
    min_pair = (0, 0)
    max_dot = -float('inf')
    max_dot_pair = (0, 0)

    for i in range(n):
        for j in range(i + 1, n):
            dot_ij = gram[i, j]
            if dot_ij > max_dot:
                max_dot = dot_ij
                max_dot_pair = (i, j)
            dist_ij = np.sqrt(max(2.0 - 2.0 * dot_ij, 0.0))
            if dist_ij < min_dist:
                min_dist = dist_ij
                min_pair = (i, j)

    if max_dot > 0.5 + TOLERANCE:
        return failure(
            f"Points {max_dot_pair[0]} and {max_dot_pair[1]} violate non-overlap: "
            f"dot product = {max_dot:.12f} > 0.5 "
            f"(distance = {min_dist:.12f} < 1)",
            min_distance=min_dist,
            max_dot_product=max_dot,
            violating_pair=list(max_dot_pair)
        )

    return success(
        f"Valid kissing configuration in ℝ⁹: {n} points, "
        f"min distance = {min_dist:.10f}, max dot product = {max_dot:.10f}",
        dimension=DIMENSION,
        num_points=n,
        min_distance=min_dist,
        max_dot_product=max_dot
    )


def main():
    parser = argparse.ArgumentParser(description='Validate kissing configuration in dimension 9')
    parser.add_argument('solution', help='Solution as JSON string or path to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == '__main__':
    main()
