#!/usr/bin/env python3
"""
Validator for problem 066: Parametric Family of Spherical Codes

Test a parametric family of spherical codes at multiple (n, d) parameters.

A spherical code is a set of points on S^(d-1) with minimum angular separation.

Baseline (Kerdock codes): N = 2^(4k) + 2^(2k+1) points in d = 2^(2k) dimensions
for 2 <= k <= 5. For d=16, N=288, cos θ = 1/4.

Expected input format:
    {
        "family": "description",
        "test_cases": [
            {"n": num_points, "dimension": d, "points": [[...], ...], "min_distance": dist},
            ...
        ]
    }
"""

import argparse
import math
from typing import Any

import numpy as np

from . import ValidationResult, load_solution, output_result, success, failure


TOLERANCE = 1e-9

# Baseline: Kerdock codes for k=2..5
# dimension d = 2^(2k), baseline N = 2^(4k) + 2^(2k+1)
# Kerdock codes use mutually unbiased bases: cross-basis |inner product| = 1/sqrt(d),
# so min Euclidean distance = sqrt(2 - 2/sqrt(d)).
KERDOCK_BASELINES = {}
for _k in range(2, 6):
    _d = 2 ** (2 * _k)
    _N = 2 ** (4 * _k) + 2 ** (2 * _k + 1)
    _min_dist = math.sqrt(2 - 2 / math.sqrt(_d))
    KERDOCK_BASELINES[_d] = {'N': _N, 'min_dist': _min_dist}
# {16: {N: 288, min_dist: 1.2247}, 64: {N: 4224, min_dist: 1.3229}, ...}


def validate_spherical_code(points: np.ndarray, n: int, d: int) -> tuple[bool, float, str]:
    """Validate a spherical code and return (valid, min_dist, message)."""
    # Check number of points matches claim
    if len(points) != n:
        return False, 0.0, f"Claimed n={n} but provided {len(points)} points"

    if n < 2:
        return False, 0.0, f"Need at least 2 points, got {n}"

    # Check dimension
    if points.ndim != 2 or points.shape[1] != d:
        actual_d = points.shape[1] if points.ndim == 2 else "?"
        return False, 0.0, f"Points have dimension {actual_d}, expected {d}"

    # Check all entries are finite
    if not np.all(np.isfinite(points)):
        return False, 0.0, "Points contain NaN or Inf values"

    # Check on unit sphere
    norms = np.linalg.norm(points, axis=1)
    if not np.allclose(norms, 1.0, atol=TOLERANCE):
        worst = np.argmax(np.abs(norms - 1.0))
        return False, 0.0, f"Point {worst} has norm {norms[worst]:.12g}, expected 1.0"

    # Compute minimum pairwise distance using vectorized gram matrix
    # dist^2 = 2 - 2*dot(p_i, p_j) for unit vectors
    gram = points @ points.T
    np.fill_diagonal(gram, -1.0)  # exclude self-pairs by setting diagonal low
    max_cos = gram.max()
    min_dist = math.sqrt(max(0.0, 2.0 - 2.0 * max_cos))

    if min_dist < TOLERANCE:
        # Find the duplicate/near-duplicate pair for error reporting
        idx = np.unravel_index(gram.argmax(), gram.shape)
        return False, 0.0, f"Points {idx[0]} and {idx[1]} are (near-)duplicates (dist={min_dist:.2e})"

    return True, min_dist, "Valid spherical code"


def validate(solution: Any) -> ValidationResult:
    """
    Validate a parametric family of spherical codes.

    Args:
        solution: Dict with family description and test cases

    Returns:
        ValidationResult with code properties
    """
    try:
        if not isinstance(solution, dict):
            return failure("Invalid format: expected dict")

        family = solution.get('family', 'not provided')
        test_cases = solution.get('test_cases', [])

        if not test_cases:
            return failure("Need at least one test case")

    except (ValueError, TypeError) as e:
        return failure(f"Failed to parse solution: {e}")

    results = []
    all_valid = True
    beats_baseline_count = 0
    total_baseline_count = 0

    for tc in test_cases:
        try:
            n = int(tc['n'])
            d = int(tc['dimension'])
        except (KeyError, TypeError, ValueError) as e:
            all_valid = False
            results.append({'valid': False, 'message': f"Bad test case format: {e}"})
            continue

        try:
            points = np.array(tc['points'], dtype=float)
        except (ValueError, TypeError) as e:
            all_valid = False
            results.append({'n': n, 'dimension': d, 'valid': False, 'message': f"Cannot parse points: {e}"})
            continue

        valid, min_dist, msg = validate_spherical_code(points, n, d)

        if not valid:
            all_valid = False

        # Convert to angular separation
        if valid and min_dist < 2:
            angular_sep = 2 * math.asin(min_dist / 2)
        else:
            angular_sep = math.pi if valid else 0.0

        # Check against Kerdock baseline for this dimension
        kerdock = KERDOCK_BASELINES.get(d)
        baseline_n = kerdock['N'] if kerdock else None
        baseline_min_dist = kerdock['min_dist'] if kerdock else None
        beats_baseline = None
        if kerdock is not None and valid:
            total_baseline_count += 1
            # To beat Kerdock: more points AND at least the same minimum distance
            beats_baseline = (n > kerdock['N']
                              and min_dist >= kerdock['min_dist'] * (1 - 1e-6))
            if beats_baseline:
                beats_baseline_count += 1

        results.append({
            'n': n,
            'dimension': d,
            'min_distance': float(min_dist),
            'angular_separation_rad': float(angular_sep),
            'angular_separation_deg': float(math.degrees(angular_sep)),
            'baseline_n': baseline_n,
            'baseline_min_dist': baseline_min_dist,
            'beats_baseline': beats_baseline,
            'valid': valid,
            'message': msg
        })

    if not all_valid:
        invalid = [r for r in results if not r.get('valid')]
        msg_parts = []
        for r in invalid[:3]:
            n_str = f"n={r.get('n', '?')}, d={r.get('dimension', '?')}"
            msg_parts.append(f"({n_str}): {r['message']}")
        return failure(
            f"Invalid code(s): {'; '.join(msg_parts)}",
            test_results=results,
        )

    total_points = sum(r['n'] for r in results if r['valid'])
    metrics = dict(
        family=family,
        total_points=total_points,
        num_test_cases=len(results),
        beats_baseline_count=beats_baseline_count,
        total_baseline_dimensions=total_baseline_count,
        test_results=results,
    )

    # Must include at least one Kerdock baseline dimension and beat it
    if total_baseline_count == 0:
        return failure(
            f"Valid codes but none in a Kerdock baseline dimension (d ∈ {sorted(KERDOCK_BASELINES.keys())}). "
            f"Include test cases at d=16, 64, 256, or 1024 to compare against baseline.",
            **metrics,
        )

    if beats_baseline_count == 0:
        # Show what was achieved vs needed for each baseline dimension
        baseline_details = []
        for r in results:
            if r.get('baseline_n') is not None:
                baseline_details.append(
                    f"d={r['dimension']}: n={r['n']} (need >{r['baseline_n']}), "
                    f"min_dist={r['min_distance']:.4f} (need >={r['baseline_min_dist']:.4f})"
                )
        return failure(
            f"Valid codes but none beat the Kerdock baseline "
            f"(need more points AND at least the same minimum distance). "
            f"{'; '.join(baseline_details)}",
            **metrics,
        )

    return success(
        f"Spherical code family valid for all {len(results)} test cases. "
        f"Total points: {total_points}. "
        f"Beats Kerdock baseline in {beats_baseline_count}/{total_baseline_count} applicable dimensions.",
        **metrics,
    )


def main():
    parser = argparse.ArgumentParser(description='Validate parametric spherical codes')
    parser.add_argument('solution', help='Solution as JSON string or path to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == '__main__':
    main()
