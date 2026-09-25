#!/usr/bin/env python3
"""
Validator for problem 055: Spherical 7-Design with Minimal Points

A spherical t-design on Sⁿ⁻¹ is a finite set of points such that the average
of any polynomial of degree ≤ t over the points equals the integral over the sphere.

For a spherical 7-design on S³ (4D sphere), the minimum number of points
is bounded below by (t+d-1 choose d-1) + (t+d-2 choose d-1) for d=4, t=7.

This validator checks:
1. All points are on the unit sphere S³
2. The design property holds for all polynomials up to degree 7
   (verified by checking moment conditions)

Expected input format:
    {"points": [[x₁, x₂, x₃, x₄], ...]}  points on S³ (4D sphere)
    or [[x₁, x₂, x₃, x₄], ...]
"""

import argparse
from itertools import product
from typing import Any

import numpy as np
from scipy.special import comb

from . import ValidationResult, load_solution, output_result, success, failure


DIMENSION = 4  # Points on S³
DESIGN_DEGREE = 7
TOLERANCE = 1e-8


def sphere_moment(powers: tuple[int, ...]) -> float:
    """
    Compute the integral of x₁^p₁ * x₂^p₂ * ... over the unit sphere.

    For the unit sphere Sⁿ⁻¹, the integral is:
    - 0 if any power is odd
    - Product of double factorial ratios otherwise
    """
    n = len(powers)

    # If any power is odd, integral is 0
    if any(p % 2 == 1 for p in powers):
        return 0.0

    # All powers even: use the formula
    # ∫ x₁^(2a₁) ... xₙ^(2aₙ) dσ =
    #   (2a₁-1)!! ... (2aₙ-1)!! / (n + 2(a₁+...+aₙ) - 2)!! * surface area factor

    total_degree = sum(powers)

    # Compute using gamma function formula for the probability measure on S^(n-1):
    # (1/|S^(n-1)|) ∫ x₁^p₁ ... xₙ^pₙ dσ = ∏Γ((pᵢ+1)/2) · Γ(n/2) / [Γ((Σpᵢ+n)/2) · π^(n/2)]
    from math import gamma, pi

    numerator = 1.0
    for p in powers:
        numerator *= gamma((p + 1) / 2)

    denominator = gamma((n + total_degree) / 2)

    return numerator / denominator * gamma(n / 2) / pi ** (n / 2)


def validate(solution: Any) -> ValidationResult:
    """
    Validate a spherical 7-design on S³.

    Args:
        solution: Dict with 'points' key or list of 4D points

    Returns:
        ValidationResult with design verification
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
    if d != DIMENSION:
        return failure(f"Points must be in ℝ⁴, got dimension {d}")

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

    # Check spherical design property for all monomials up to degree t
    max_error = 0.0
    worst_monomial = None

    for total_deg in range(DESIGN_DEGREE + 1):
        # Generate all monomials of this degree
        for powers in product(range(total_deg + 1), repeat=DIMENSION):
            if sum(powers) != total_deg:
                continue

            # Compute average over points
            monomial_values = np.prod(points ** powers, axis=1)
            point_avg = np.mean(monomial_values)

            # Compute sphere integral
            sphere_avg = sphere_moment(powers)

            error = abs(point_avg - sphere_avg)
            if error > max_error:
                max_error = error
                worst_monomial = powers

    if max_error > TOLERANCE:
        return failure(
            f"Not a {DESIGN_DEGREE}-design: max error = {max_error:.2e} at monomial {worst_monomial}",
            max_moment_error=max_error
        )

    return success(
        f"Valid spherical {DESIGN_DEGREE}-design on S³ with {n} points (max error: {max_error:.2e})",
        dimension=DIMENSION,
        num_points=n,
        design_degree=DESIGN_DEGREE,
        max_moment_error=max_error
    )


def main():
    parser = argparse.ArgumentParser(description='Validate spherical 7-design on S³')
    parser.add_argument('solution', help='Solution as JSON string or path to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == '__main__':
    main()
