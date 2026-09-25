#!/usr/bin/env python3
"""
Validator for problem 090: High-Rank Elliptic Curve with Torsion ℤ/7ℤ

Validates an elliptic curve over ℚ with torsion subgroup ℤ/7ℤ and computes its rank.

This validator uses SageMath to:
1. Verify the curve is valid
2. Check the torsion subgroup is exactly ℤ/7ℤ
3. Verify provided points have infinite order and are independent

Expected input format:
    {
        "curve": [a1, a2, a3, a4, a6],
        "torsion_point": [x, y],  # Point of order 7
        "infinite_order_points": [[x1, y1], ...]  # Points of infinite order
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


TORSION_ORDER = 7


def run_sage_verification(curve_coeffs: list, torsion_point: list, inf_points: list) -> ValidationResult:
    """Run SageMath code to verify curve properties and compute rank."""

    sage_code = f'''
from sage.all import *

curve_coeffs = {curve_coeffs}
torsion_pt = {torsion_point}
inf_points_data = {inf_points}

E = EllipticCurve(curve_coeffs)
print(f"Curve: {{E}}")

if E.discriminant() == 0:
    print("RESULT: FAIL")
    print("MESSAGE: Curve is singular")
    exit(0)

# Verify torsion point
try:
    tx, ty = torsion_pt
    for label, v in [("x", tx), ("y", ty)]:
        if not isinstance(v, (int, str)):
            print("RESULT: FAIL")
            print(f"MESSAGE: Torsion point {{label}}-coordinate must be an integer or ratio string 'p/q', got {{type(v).__name__}}")
            exit(0)
    T = E(QQ(tx), QQ(ty))
    t_order = T.order()
    print(f"Torsion point order: {{t_order}}")

    if t_order != {TORSION_ORDER}:
        print("RESULT: FAIL")
        print(f"MESSAGE: Torsion point has order {{t_order}}, expected {TORSION_ORDER}")
        exit(0)
except Exception as e:
    print("RESULT: FAIL")
    print(f"MESSAGE: Torsion point error: {{e}}")
    exit(0)

# Check full torsion subgroup
torsion = E.torsion_subgroup()
torsion_structure = torsion.invariants()
print(f"Full torsion subgroup: {{torsion_structure}}")

if torsion_structure != ({TORSION_ORDER},):
    print("RESULT: FAIL")
    print(f"MESSAGE: Torsion subgroup is {{torsion_structure}}, expected ({TORSION_ORDER},)")
    exit(0)

# Verify infinite order points
valid_points = []
for i, (px, py) in enumerate(inf_points_data):
    for label, v in [("x", px), ("y", py)]:
        if not isinstance(v, (int, str)):
            print("RESULT: FAIL")
            print(f"MESSAGE: Point {{i}} {{label}}-coordinate must be an integer or ratio string 'p/q', got {{type(v).__name__}}")
            exit(0)
    try:
        P = E(QQ(px), QQ(py))
        if P.order() != Infinity:
            print("RESULT: FAIL")
            print(f"MESSAGE: Point {{i}} has finite order {{P.order()}}")
            exit(0)
        valid_points.append(P)
    except Exception as e:
        print("RESULT: FAIL")
        print(f"MESSAGE: Point {{i}} error: {{e}}")
        exit(0)

print(f"Valid infinite-order points: {{len(valid_points)}}")

if len(valid_points) == 0:
    print("RESULT: SUCCESS")
    print(f"MESSAGE: Valid curve with torsion Z/{TORSION_ORDER}Z and rank 0")
    print("RANK: 0")
    exit(0)

# Check independence
try:
    saturated, _, _ = E.saturation(valid_points)
    independent_count = len(saturated)

    print(f"Independent points: {{independent_count}}")
    print("RESULT: SUCCESS")
    print(f"MESSAGE: Valid curve with torsion Z/{TORSION_ORDER}Z and {{independent_count}} independent points")
    print(f"RANK: {{independent_count}}")

except Exception as e:
    print("RESULT: ERROR")
    print(f"MESSAGE: {{e}}")
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.sage', delete=False) as f:
        f.write(sage_code)
        temp_path = f.name

    try:
        result = run_sage_script(temp_path, timeout=3600)

        output = result.stdout + result.stderr

        # Extract rank from output
        rank = 0
        for line in output.split('\n'):
            if line.startswith('RANK:'):
                rank = int(line.split(':')[1].strip())

        if 'RESULT: SUCCESS' in output:
            msg_line = [l for l in output.split('\n') if 'MESSAGE:' in l]
            msg = msg_line[0].split('MESSAGE:')[1].strip() if msg_line else "Verified"
            return success(msg, torsion_order=TORSION_ORDER, rank=rank)
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
    Validate elliptic curve with torsion ℤ/7ℤ and compute its rank.

    Args:
        solution: Dict with curve, torsion_point, infinite_order_points

    Returns:
        ValidationResult with torsion verification and rank
    """
    try:
        if not isinstance(solution, dict):
            return failure("Invalid format: expected dict")

        curve_coeffs = solution['curve']
        torsion_point = solution['torsion_point']
        inf_points = solution.get('infinite_order_points', [])

        if len(curve_coeffs) != 5:
            return failure(f"Curve needs 5 coefficients, got {len(curve_coeffs)}")

    except (KeyError, TypeError) as e:
        return failure(f"Failed to parse solution: {e}")

    return run_sage_verification(curve_coeffs, torsion_point, inf_points)


def main():
    parser = argparse.ArgumentParser(description='Validate elliptic curve with torsion Z/7Z and compute rank')
    parser.add_argument('solution', help='Solution as JSON string or path to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == '__main__':
    main()
