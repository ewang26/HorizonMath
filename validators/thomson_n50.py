#!/usr/bin/env python3
"""
Validator for problem 059: Thomson Problem for n=50

The Thomson problem asks for n points on a unit sphere that minimize
the electrostatic potential energy E = Σᵢ<ⱼ 1/|xᵢ - xⱼ|.

For n=50, this validator:
1. Checks all points are on the unit sphere S²
2. Computes the electrostatic energy
3. Reports the configuration quality

Expected input format:
    {"points": [[x, y, z], ...]}  50 points on S²
    or [[x, y, z], ...]
"""

import argparse
from typing import Any

import numpy as np

from . import ValidationResult, load_solution, output_result, success, failure


TARGET_N = 50
TOLERANCE = 1e-9


def validate(solution: Any) -> ValidationResult:
    """
    Validate a Thomson configuration for n=50.

    Args:
        solution: Dict with 'points' key or list of 50 3D points

    Returns:
        ValidationResult with energy and configuration properties
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

    # Compute electrostatic energy
    energy = 0.0
    min_dist = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points[i] - points[j])
            if dist < TOLERANCE:
                return failure(f"Points {i} and {j} are coincident")
            energy += 1.0 / dist
            min_dist = min(min_dist, dist)

    return success(
        f"Thomson configuration for n={n}: energy = {energy:.10f}, min distance = {min_dist:.6f}",
        num_points=n,
        energy=energy,
        min_distance=min_dist
    )


def main():
    parser = argparse.ArgumentParser(description='Validate Thomson configuration for n=50')
    parser.add_argument('solution', help='Solution as JSON string or path to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == '__main__':
    main()
