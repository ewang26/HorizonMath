#!/usr/bin/env python3
"""
Validator for problem 088: Inverse Galois Problem for Suzuki Group ²B₂(8)

The goal is to find a polynomial over ℚ whose Galois group is the
Suzuki group Sz(8) = ²B₂(8) (order 29,120).

This validator uses SageMath to:
1. Check the polynomial is irreducible over ℚ
2. Compute the Galois group
3. Verify it has the correct order and properties

Expected input format:
    {"coefficients": [a₀, a₁, ..., aₙ]}  for polynomial a₀ + a₁x + ... + aₙxⁿ
    or [a₀, a₁, ..., aₙ]

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


SZ8_ORDER = 29120


def run_sage_verification(coefficients: list) -> ValidationResult:
    """Run SageMath code to verify Galois group."""

    sage_code = f'''
from sage.all import *

coeffs = {coefficients}
x = polygen(QQ)
f = sum(c * x^i for i, c in enumerate(coeffs))

print(f"Polynomial degree: {{f.degree()}}")

if not f.is_irreducible():
    print("RESULT: FAIL")
    print("MESSAGE: Polynomial is not irreducible over Q")
    exit(0)

print("Polynomial is irreducible over Q")

try:
    G = f.galois_group(pari_group=True)
    group_order = G.order()
    print(f"Galois group order: {{group_order}}")

    if group_order == {SZ8_ORDER}:
        print("RESULT: SUCCESS")
        print(f"MESSAGE: Galois group has order {SZ8_ORDER}, consistent with Sz(8)")
    else:
        print("RESULT: FAIL")
        print(f"MESSAGE: Galois group order {{group_order}} != {SZ8_ORDER} (Sz(8))")

except Exception as e:
    print("RESULT: ERROR")
    print(f"MESSAGE: {{e}}")
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.sage', delete=False) as f:
        f.write(sage_code)
        temp_path = f.name

    try:
        result = run_sage_script(temp_path, timeout=1800)

        output = result.stdout + result.stderr

        if 'RESULT: SUCCESS' in output:
            msg_line = [l for l in output.split('\n') if 'MESSAGE:' in l]
            msg = msg_line[0].split('MESSAGE:')[1].strip() if msg_line else "Verified"
            return success(msg, galois_group_order=SZ8_ORDER)
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
        return failure("Computation timed out (30 minutes)")
    except Exception as e:
        return failure(f"Execution error: {e}")
    finally:
        Path(temp_path).unlink(missing_ok=True)


def validate(solution: Any) -> ValidationResult:
    """
    Validate a polynomial has Galois group Sz(8).

    Args:
        solution: Dict with 'coefficients' key or list of coefficients

    Returns:
        ValidationResult with Galois group verification
    """
    try:
        if isinstance(solution, dict) and 'coefficients' in solution:
            coeffs = solution['coefficients']
        elif isinstance(solution, list):
            coeffs = solution
        else:
            return failure("Invalid format: expected dict with 'coefficients' or list")

        coeffs = [int(c) for c in coeffs]
    except (ValueError, TypeError) as e:
        return failure(f"Failed to parse coefficients: {e}")

    if len(coeffs) < 2:
        return failure("Polynomial must have degree at least 1")

    # Sz(8) acts on 65 points
    degree = len(coeffs) - 1
    if degree != 65:
        return failure(f"Polynomial has degree {degree}, expected 65 for Sz(8)")

    return run_sage_verification(coeffs)


def main():
    parser = argparse.ArgumentParser(description='Validate polynomial with Galois group Sz(8)')
    parser.add_argument('solution', help='Solution as JSON string or path to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == '__main__':
    main()
