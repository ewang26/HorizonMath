#!/usr/bin/env python3
"""
Validator for problem 087: Inverse Galois Problem for M23

Goal: verify a submitted integer-coefficient polynomial of degree 23 has
Galois group isomorphic to the Mathieu group M23 (order 10,200,960).

This validator uses SageMath to:
1) Confirm polynomial degree is exactly 23
2) Check irreducibility over Q
3) Compute the Galois group (as a permutation group) and identify it as
   transitive group 23T5 (= M23)

Expected input format:
    {"coefficients": [a0, a1, ..., a23]}  for a0 + a1*x + ... + a23*x^23
    or [a0, a1, ..., a23]

Requires: SageMath (with GAP available, as in standard Sage installs)
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

M23_ORDER = 10200960
M23_TRANSITIVE_NUMBER = 5  # 23T5 in the transitive group database

# Guardrails to keep Sage computations tractable in a benchmark setting.
SAGE_TIMEOUT_SECONDS = 300  # tune as needed (e.g., 120–600)
COEFF_ABS_MAX = 10**6       # tune as needed; smaller => faster/more robust


def run_sage_verification(coefficients: list[int]) -> ValidationResult:
    """Run SageMath code to verify the submitted polynomial has Gal ≅ M23."""

    sage_code = f"""
from sage.all import *

coeffs = {coefficients}
x = polygen(QQ)
f = sum(QQ(c) * x^i for i, c in enumerate(coeffs))

print(f"Polynomial degree (Sage): {{f.degree()}}")

# Hard degree check in Sage as a sanity check.
if f.degree() != 23:
    print("RESULT: FAIL")
    print("MESSAGE: Polynomial degree in Sage is not 23")
    quit()

# Check irreducibility over Q
if not f.is_irreducible():
    print("RESULT: FAIL")
    print("MESSAGE: Polynomial is not irreducible over Q")
    quit()

print("Polynomial is irreducible over Q")

# Compute Galois group using GAP backend (PARI polgalois does not support degree 23).
try:
    try:
        G = f.galois_group(algorithm='gap')
    except TypeError:
        # Fallback for older Sage signatures that may not accept algorithm=...
        G = f.galois_group()

    group_order = int(G.order())
    print(f"Galois group order: {{group_order}}")

    # Identify the group by its transitive label number.
    # For irreducible degree-23 polynomials, Gal group is transitive.
    tn = None
    try:
        tn = int(G.transitive_number())
        print(f"Transitive number: {{tn}}")
    except Exception as e:
        print(f"Transitive number: unavailable ({{e}})")

    # Primary identification: 23T5 is M23.
    if tn == {M23_TRANSITIVE_NUMBER}:
        # Optional consistency check on order (should match M23).
        if group_order != {M23_ORDER}:
            print("RESULT: FAIL")
            print(f"MESSAGE: Transitive group 23T5 but order {{group_order}} != {M23_ORDER}")
        else:
            print("RESULT: SUCCESS")
            print("MESSAGE: Verified Gal(f) is transitive group 23T5 (M23)")
        quit()

    # Fallback: if transitive_number() is unavailable, try explicit isomorphism check.
    if tn is None:
        try:
            H = TransitiveGroup(23, {M23_TRANSITIVE_NUMBER})
            if G.is_isomorphic(H):
                if group_order != {M23_ORDER}:
                    print("RESULT: FAIL")
                    print(f"MESSAGE: Isomorphic to 23T5 but order {{group_order}} != {M23_ORDER}")
                else:
                    print("RESULT: SUCCESS")
                    print("MESSAGE: Verified Gal(f) is isomorphic to TransitiveGroup(23,5) (M23)")
            else:
                print("RESULT: FAIL")
                print(f"MESSAGE: Could not identify transitive number; computed order {{group_order}}")
        except Exception as e:
            print("RESULT: FAIL")
            print(f"MESSAGE: Could not identify transitive number or test isomorphism ({{e}})")
        quit()

    # If we got a transitive number but it's not 5, fail.
    print("RESULT: FAIL")
    print(f"MESSAGE: Transitive group is 23T{{tn}}, not 23T5 (M23)")

except Exception as e:
    print("RESULT: ERROR")
    print(f"MESSAGE: {{e}}")
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sage", delete=False) as f:
        f.write(sage_code)
        temp_path = f.name

    try:
        result = run_sage_script(temp_path, timeout=SAGE_TIMEOUT_SECONDS)
        output = (result.stdout or "") + (result.stderr or "")

        if "RESULT: SUCCESS" in output:
            msg_line = [l for l in output.split("\n") if "MESSAGE:" in l]
            msg = msg_line[0].split("MESSAGE:", 1)[1].strip() if msg_line else "Verified"
            return success(
                msg,
                galois_group_order=M23_ORDER,
                transitive_number=M23_TRANSITIVE_NUMBER,
            )

        if "RESULT: FAIL" in output:
            msg_line = [l for l in output.split("\n") if "MESSAGE:" in l]
            msg = msg_line[0].split("MESSAGE:", 1)[1].strip() if msg_line else "Failed"
            return failure(msg)

        if "RESULT: ERROR" in output:
            msg_line = [l for l in output.split("\n") if "MESSAGE:" in l]
            msg = msg_line[0].split("MESSAGE:", 1)[1].strip() if msg_line else "Error"
            return failure(f"SageMath error: {msg}")

        return failure(f"Unexpected output: {output[:500]}")

    except FileNotFoundError:
        return failure(sage_not_found_message())
    except subprocess.TimeoutExpired:
        return failure(f"Computation timed out ({SAGE_TIMEOUT_SECONDS} seconds)")
    except Exception as e:
        return failure(f"Execution error: {e}")
    finally:
        Path(temp_path).unlink(missing_ok=True)


def validate(solution: Any) -> ValidationResult:
    """
    Validate a polynomial has Galois group M23.

    Args:
        solution: Dict with 'coefficients' key or list of coefficients

    Returns:
        ValidationResult with Galois group verification
    """
    try:
        if isinstance(solution, dict) and "coefficients" in solution:
            coeffs = solution["coefficients"]
        elif isinstance(solution, list):
            coeffs = solution
        else:
            return failure("Invalid format: expected dict with 'coefficients' or list")

        coeffs = [int(c) for c in coeffs]
    except (ValueError, TypeError) as e:
        return failure(f"Failed to parse coefficients: {e}")

    # Require exactly 24 coefficients for degree-23 polynomial.
    if len(coeffs) != 24:
        return failure(f"Expected 24 coefficients [a0..a23], got {len(coeffs)}")

    # Leading coefficient must be nonzero to truly have degree 23.
    if coeffs[-1] == 0:
        return failure("Leading coefficient a23 must be nonzero (degree must be exactly 23)")

    # Guardrail: cap coefficient magnitudes to keep computations tractable.
    max_abs = max(abs(c) for c in coeffs) if coeffs else 0
    if max_abs > COEFF_ABS_MAX:
        return failure(
            f"Coefficient magnitude too large: max |ai| = {max_abs} > {COEFF_ABS_MAX}"
        )

    return run_sage_verification(coeffs)


def main():
    parser = argparse.ArgumentParser(description="Validate polynomial with Galois group M23")
    parser.add_argument("solution", help="Solution as JSON string or path to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == "__main__":
    main()