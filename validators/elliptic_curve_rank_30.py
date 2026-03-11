#!/usr/bin/env python3
"""
Validator for problem 089: Elliptic Curve with High Rank

Validates an elliptic curve over ℚ and computes the rank from provided points.

This validator uses SageMath to:
1. Verify the curve is valid (non-singular)
2. Verify provided points are on the curve
3. Check linear independence of points using saturation

Expected input format:
    {
        "curve": [a1, a2, a3, a4, a6],  # Weierstrass coefficients
        "points": [[x1, y1], [x2, y2], ...]  # Rational points
    }

Requires: SageMath
"""

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from . import (
    ValidationResult,
    load_solution,
    output_result,
    run_sage_script,
    sage_not_found_message,
    success,
    failure,
)


def run_sage_verification(curve_coeffs: list, points: list) -> ValidationResult:
    """Run SageMath code to verify elliptic curve and compute rank."""

    sage_code = f'''
from sage.all import *

curve_coeffs = {curve_coeffs}
points_data = {points}

# Create curve
E = EllipticCurve(curve_coeffs)
print(f"Curve: {{E}}")
print(f"Discriminant: {{E.discriminant()}}")

if E.discriminant() == 0:
    print("RESULT: FAIL")
    print("MESSAGE: Curve is singular")
    exit(0)

# Verify points are on curve
valid_points = []
for i, (px, py) in enumerate(points_data):
    for label, v in [("x", px), ("y", py)]:
        if not isinstance(v, (int, str)):
            print("RESULT: FAIL")
            print(f"MESSAGE: Point {{i}} {{label}}-coordinate must be an integer or ratio string 'p/q', got {{type(v).__name__}}")
            exit(0)
    try:
        P = E(QQ(px), QQ(py))
        if P.is_zero():
            print(f"Point {{i}}: identity (skipping)")
            continue
        valid_points.append(P)
    except Exception as e:
        print("RESULT: FAIL")
        print(f"MESSAGE: Point {{i}} is not on curve: {{e}}")
        exit(0)

print(f"Valid non-identity points: {{len(valid_points)}}")

if len(valid_points) == 0:
    print("RESULT: SUCCESS")
    print("MESSAGE: Valid curve with no non-identity points")
    print("RANK: 0")
    exit(0)

# Check independence using saturation
try:
    saturated, _, _ = E.saturation(valid_points)
    independent_count = len(saturated)

    print(f"Independent points after saturation: {{independent_count}}")
    print("RESULT: SUCCESS")
    print(f"MESSAGE: Valid curve with {{independent_count}} independent points")
    print(f"RANK: {{independent_count}}")

except Exception as e:
    print("RESULT: ERROR")
    print(f"MESSAGE: {{e}}")
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.sage', delete=False) as f:
        f.write(sage_code)
        temp_path = f.name

    try:
        result = run_sage_script(temp_path, timeout=3600)  # 1 hour for rank computation

        output = result.stdout + result.stderr

        # Extract rank from output
        rank = 0
        for line in output.split('\n'):
            if line.startswith('RANK:'):
                rank = int(line.split(':')[1].strip())

        if 'RESULT: SUCCESS' in output:
            msg_line = [l for l in output.split('\n') if 'MESSAGE:' in l]
            msg = msg_line[0].split('MESSAGE:')[1].strip() if msg_line else "Verified"
            return success(msg, rank=rank)
        elif 'RESULT: FAIL' in output:
            msg_line = [l for l in output.split('\n') if 'MESSAGE:' in l]
            msg = msg_line[0].split('MESSAGE:')[1].strip() if msg_line else "Failed"
            return failure(msg)
        elif 'RESULT: ERROR' in output:
            msg_line = [l for l in output.split('\n') if 'MESSAGE:' in l]
            msg = msg_line[0].split('MESSAGE:')[1].strip() if msg_line else "Error"
            return failure(f"SageMath error: {msg}")
        else:
            return failure(f"Unexpected output: {output[:500]}")

    except FileNotFoundError:
        return failure(sage_not_found_message())
    except subprocess.TimeoutExpired:
        return failure("Computation timed out (1 hour)")
    except Exception as e:
        return failure(f"Execution error: {e}")
    finally:
        Path(temp_path).unlink(missing_ok=True)


def validate(solution: Any) -> ValidationResult:
    """
    Validate an elliptic curve and compute its rank from provided points.

    Args:
        solution: Dict with 'curve' and 'points' keys

    Returns:
        ValidationResult with rank
    """
    try:
        if not isinstance(solution, dict):
            return failure("Invalid format: expected dict with 'curve' and 'points' keys")

        curve_coeffs = solution['curve']
        points = solution.get('points', [])

        if len(curve_coeffs) != 5:
            return failure(f"Curve needs 5 coefficients [a1,a2,a3,a4,a6], got {len(curve_coeffs)}")

    except (KeyError, TypeError) as e:
        return failure(f"Failed to parse solution: {e}")

    return run_sage_verification(curve_coeffs, points)


def main():
    parser = argparse.ArgumentParser(description='Validate elliptic curve and compute rank')
    parser.add_argument('solution', help='Solution as JSON string or path to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == '__main__':
    main()
