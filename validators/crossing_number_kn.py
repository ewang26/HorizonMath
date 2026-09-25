#!/usr/bin/env python3
"""
Validator for problem 068: Rectilinear Crossing Number upper bound for K_99

This benchmark evaluates *rectilinear* (straight-line) drawings: vertices are points
in the plane (no three collinear), and edges are straight-line segments.

Given a submitted point set for K_99, the validator counts crossings between pairs
of non-adjacent edges in the straight-line drawing, and returns crossing_count.

Baseline (published upper bound): 1,404,552 crossings for a rectilinear drawing of K_99.
A valid submission "beats baseline" iff crossing_count < 1404552.
"""

import argparse
import math
from itertools import combinations
from typing import Any

from . import ValidationResult, load_solution, output_result, success, failure


MAX_N = 150          # keep O(n^4) tractable; unused if we force TARGET_N
TARGET_N = 99        # this benchmark instance is for K_99
BASELINE = 1404552   # published upper bound to beat
COORD_BOUND = 1e9    # avoid overflow / numeric pathologies


def _cross(o, a, b):
    """2D cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_cross(p1, p2, p3, p4):
    """Check if open segment p1-p2 properly crosses open segment p3-p4."""
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)

    # Proper crossing test (strict orientation); excludes endpoint intersections.
    return (
        ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and
        ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))
    )


def count_crossings(points):
    """
    Count the number of edge crossings in a straight-line drawing of K_n.

    For each 4-subset of vertices, checks the three possible disjoint edge pairings.
    In a straight-line drawing with vertices in general position, at most one pairing
    per 4-subset can cross.
    """
    n = len(points)
    crossings = 0

    for a, b, c, d in combinations(range(n), 4):
        pa, pb, pc, pd = points[a], points[b], points[c], points[d]
        if segments_cross(pa, pb, pc, pd):
            crossings += 1
        elif segments_cross(pa, pc, pb, pd):
            crossings += 1
        elif segments_cross(pa, pd, pb, pc):
            crossings += 1

    return crossings


def points_in_general_position(points):
    """Check that no three points are collinear."""
    n = len(points)
    for i, j, k in combinations(range(n), 3):
        if _cross(points[i], points[j], points[k]) == 0:
            return False, (i, j, k)
    return True, None


def validate(solution: Any) -> ValidationResult:
    """
    Validate rectilinear drawings and compute crossing_count for K_99.

    Returns:
        ValidationResult with crossing_count as the key scalar metric.
    """
    try:
        if not isinstance(solution, dict):
            return failure("Invalid format: expected dict with 'drawings' key")

        drawings = solution.get("drawings", [])
        if not drawings:
            return failure(
                "Missing or empty 'drawings' list. Provide at least one drawing "
                "as {'n': <int>, 'points': [[x1,y1], [x2,y2], ...]}."
            )
    except (ValueError, TypeError) as e:
        return failure(f"Failed to parse solution: {e}")

    best_crossings = None
    best_idx = None
    drawing_results = []

    for idx, drawing in enumerate(drawings):
        try:
            n = int(drawing["n"])
            raw_points = drawing["points"]
        except (KeyError, ValueError, TypeError) as e:
            return failure(f"Drawing {idx}: invalid format — {e}")

        if n != TARGET_N:
            return failure(f"Drawing {idx}: expected n={TARGET_N}, got n={n}")

        if n > MAX_N:
            return failure(f"Drawing {idx}: n={n} exceeds maximum {MAX_N}")

        try:
            points = [(float(p[0]), float(p[1])) for p in raw_points]
        except (ValueError, TypeError, IndexError) as e:
            return failure(f"Drawing {idx}: invalid point coordinates — {e}")

        if len(points) != n:
            return failure(f"Drawing {idx}: expected {n} points, got {len(points)}")

        # Reject NaN/Inf and pathological magnitudes (prevents trivial exploits)
        for j, (x, y) in enumerate(points):
            if not (math.isfinite(x) and math.isfinite(y)):
                return failure(f"Drawing {idx}: point {j} has non-finite coordinate(s)")
            if abs(x) > COORD_BOUND or abs(y) > COORD_BOUND:
                return failure(f"Drawing {idx}: point {j} exceeds coordinate bound {COORD_BOUND:g}")

        # Check for duplicate points
        if len(set(points)) < n:
            return failure(
                f"Drawing {idx}: has duplicate points (all vertices must be distinct)"
            )

        # Check general position (no 3 collinear)
        gp, collinear = points_in_general_position(points)
        if not gp:
            i, j, k = collinear
            return failure(
                f"Drawing {idx}: points {i}, {j}, {k} are collinear "
                f"(vertices must be in general position)"
            )

        crossings = count_crossings(points)

        drawing_results.append(
            {
                "n": n,
                "crossings": crossings,
                "baseline": BASELINE,
                "improves_baseline": crossings < BASELINE,
            }
        )

        if best_crossings is None or crossings < best_crossings:
            best_crossings = crossings
            best_idx = idx

    delta = best_crossings - BASELINE
    msg = (
        f"Best crossing_count={best_crossings} for K_{TARGET_N} "
        f"(baseline={BASELINE}, delta={delta})"
    )

    return success(
        msg,
        crossing_count=best_crossings,
        baseline=BASELINE,
        delta=delta,
        improves_baseline=(best_crossings < BASELINE),
        best_drawing_index=best_idx,
        num_drawings=len(drawing_results),
        drawing_results=drawing_results,
    )


def main():
    parser = argparse.ArgumentParser(description="Validate rectilinear drawings for K_99")
    parser.add_argument("solution", help="Solution as JSON string or path to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == "__main__":
    main()