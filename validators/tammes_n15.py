#!/usr/bin/env python3
"""
Validator for problem 060: Tammes Problem for n=15

The Tammes problem asks to place n points on a unit sphere to maximize
the minimum pairwise distance.

For n=15, this validator:
1. Checks all points are on the unit sphere S²
2. Computes the minimum pairwise distance
3. Reports the angular separation in degrees

Expected input format:
    {"points": [[x, y, z], ...]}  15 points on S²
    or [[x, y, z], ...]
"""

import argparse
import math
from typing import Any

import numpy as np

from . import ValidationResult, load_solution, output_result, success, failure


TARGET_N = 15
TOLERANCE = 1e-9


def validate(solution: Any) -> ValidationResult:
    """
    Validate a Tammes configuration for n=15.

    Args:
        solution: Dict with 'points' key or list of 15 3D points

    Returns:
        ValidationResult with minimum distance and angular separation
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
    if d != 3:
        return failure(f"Points must be in ℝ³, got dimension {d}")

    if n != TARGET_N:
        return failure(f"Expected {TARGET_N} points, got {n}")

    # Check all points are on unit sphere
    norms = np.linalg.norm(points, axis=1)
    off_sphere = np.abs(norms - 1.0) > TOLERANCE
    if np.any(off_sphere):
        worst_idx = np.argmax(np.abs(norms - 1.0))
        return failure(
            f"Point {worst_idx} not on unit sphere: |x| = {norms[worst_idx]:.10f}",
            off_sphere_count=int(np.sum(off_sphere))
        )

    # Compute minimum pairwise distance
    min_dist = float('inf')
    min_pair = (0, 0)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points[i] - points[j])
            if dist < min_dist:
                min_dist = dist
                min_pair = (i, j)

    if min_dist < TOLERANCE:
        return failure(f"Points {min_pair[0]} and {min_pair[1]} are coincident")

    # Convert to angular separation (chord length to angle)
    # For unit sphere, if chord = d, then angle = 2*arcsin(d/2)
    angular_sep_rad = 2 * math.asin(min(min_dist / 2, 1.0))
    angular_sep_deg = math.degrees(angular_sep_rad)

    return success(
        f"Tammes configuration for n={n}: min distance = {min_dist:.10f}, "
        f"angular separation = {angular_sep_deg:.4f}°",
        num_points=n,
        min_distance=min_dist,
        angular_separation_degrees=angular_sep_deg
    )


def main():
    parser = argparse.ArgumentParser(description='Validate Tammes configuration for n=15')
    parser.add_argument('solution', help='Solution as JSON string or path to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == '__main__':
    main()
