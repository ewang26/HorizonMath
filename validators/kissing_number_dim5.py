#!/usr/bin/env python3
"""
Validator for problem 049: Kissing Number in Dimension 5

The kissing number τ₅ is the maximum number of non-overlapping unit spheres
that can touch a central unit sphere in 5 dimensions.

Known bounds: 40 ≤ τ₅ ≤ 44

A valid kissing configuration is a set of unit vectors in R⁵ (contact points
on the central sphere) such that the dot product between any two distinct
unit vectors is at most 1/2 (angular separation ≥ 60°).  Equivalently, the
Euclidean distance between any two contact points is at least 1.

This validator checks that:
1. All points lie on the unit sphere S⁴ (‖x‖ = 1)
2. All pairwise dot products are ≤ 1/2 (equivalently, distances ≥ 1)
3. No two points are identical (deduplication)
4. Reports the number of valid contact points

Expected input format:
    {"points": [[x₁, …, x₅], …]}  each point in R⁵
    or [[x₁, …, x₅], …]
"""

import argparse
from typing import Any

import numpy as np

from . import ValidationResult, load_solution, output_result, success, failure


DIMENSION = 5
# Two unit spheres of radius 1 touching the central unit sphere of radius 1
# are non-overlapping iff the distance between their centers is ≥ 2.
# The centers sit at distance 2 from the origin (radius of central + radius of
# kissing sphere), so the contact points on the central sphere are at distance 1.
# The distance between two contact points p, q on the unit sphere is
# |p - q| = sqrt(2 - 2·p·q).  Non-overlap requires |p - q| ≥ sqrt(2)
# for sphere centers, but since contact points are at unit distance:
# center_i = 2·p_i, so |center_i - center_j| = 2·|p_i - p_j| ≥ 2
# ⟹ |p_i - p_j| ≥ 1.  Wait — let's be precise.
#
# Actually: contact points are ON the unit sphere (norm 1).  The centers of
# the kissing spheres are at 2·p_i (distance 2 from origin).  Two kissing
# spheres (each radius 1) are non-overlapping iff |2p_i - 2p_j| ≥ 2, i.e.
# |p_i - p_j| ≥ 1.  Since |p_i - p_j|² = 2 - 2·p_i·p_j, the condition is
# p_i · p_j ≤ 1/2.
#
# This is the standard formulation: unit vectors with pairwise dot product ≤ 1/2.

MIN_CONTACT_DISTANCE = 1.0  # minimum distance between contact points on the unit sphere
TOLERANCE = 1e-9


def validate(solution: Any) -> ValidationResult:
    """
    Validate a kissing configuration in dimension 5.

    Args:
        solution: Dict with 'points' key or list of points

    Returns:
        ValidationResult with point count and minimum distance
    """
    # --- Parse input ---
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
        return failure(f"Points must be in R^{DIMENSION}, got dimension {d}")

    if n == 0:
        return failure("No points provided")

    # --- Check all points are on the unit sphere ---
    norms = np.linalg.norm(points, axis=1)
    off_sphere = np.abs(norms - 1.0) > TOLERANCE
    if np.any(off_sphere):
        worst_idx = int(np.argmax(np.abs(norms - 1.0)))
        return failure(
            f"Point {worst_idx} not on unit sphere: |x| = {norms[worst_idx]:.12f}",
            off_sphere_count=int(np.sum(off_sphere))
        )

    # --- Deduplicate: remove points that are identical up to tolerance ---
    # (prevents inflating count with repeated vectors)
    unique_mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not unique_mask[i]:
            continue
        for j in range(i + 1, n):
            if not unique_mask[j]:
                continue
            if np.linalg.norm(points[i] - points[j]) < TOLERANCE:
                unique_mask[j] = False

    n_unique = int(np.sum(unique_mask))
    if n_unique < n:
        points = points[unique_mask]
        n = n_unique

    # --- Check pairwise distances ≥ 1 (equivalently, dot products ≤ 0.5) ---
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
        f"Valid kissing configuration in R^{DIMENSION}: {n} points, "
        f"min distance = {min_dist:.10f}, max dot product = {max_dot:.10f}",
        dimension=DIMENSION,
        num_points=n,
        min_distance=min_dist,
        max_dot_product=max_dot
    )


def main():
    parser = argparse.ArgumentParser(description='Validate kissing configuration in dimension 5')
    parser.add_argument('solution', help='Solution as JSON string or path to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == '__main__':
    main()
