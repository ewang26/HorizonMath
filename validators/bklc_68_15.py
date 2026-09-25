#!/usr/bin/env python3
"""
Validator for problem 077: Improve Minimum Distance of a Binary Linear [68,15] Code

A candidate submits a generator matrix G over GF(2) with shape 15 x 68.
We verify:
  1) Proper format
  2) Rank(G) = 15 over GF(2)
  3) Compute exact minimum distance by enumerating all 2^15 codewords

Metric:
  - min_distance (to be maximized)

Expected input format (output of proposed_solution()):
    {
        "generator_matrix": [
            "0101... (68 bits)",
            ... (15 rows total)
        ]
    }
"""

import argparse
from typing import Any, List

from . import (
    ValidationResult,
    load_solution,
    output_result,
    success,
    failure,
)

N = 68
K = 15


def _row_to_mask(row: str) -> int:
    s = row.strip().replace(" ", "")
    if len(s) != N:
        raise ValueError(f"Row has length {len(s)} but expected {N}")
    mask = 0
    # Use bit N-1 as leftmost character, bit 0 as rightmost.
    for i, ch in enumerate(s):
        if ch == "1":
            mask |= 1 << (N - 1 - i)
        elif ch == "0":
            continue
        else:
            raise ValueError("Row contains non-binary character")
    return mask


def _gf2_rank(row_masks: List[int]) -> int:
    # Gaussian elimination in GF(2) using pivot dictionary keyed by leading bit index.
    pivots = {}
    for r in row_masks:
        x = r
        while x:
            b = x.bit_length() - 1
            if b in pivots:
                x ^= pivots[b]
            else:
                pivots[b] = x
                break
    return len(pivots)


def _min_distance_gray(row_masks: List[int]) -> int:
    # Enumerate all nonzero linear combinations via Gray code.
    k = len(row_masks)
    codeword = 0
    dmin = N + 1
    prev_gray = 0

    for i in range(1, 1 << k):
        gray = i ^ (i >> 1)
        diff = gray ^ prev_gray  # exactly one bit differs
        idx = diff.bit_length() - 1
        codeword ^= row_masks[idx]
        w = codeword.bit_count()
        if w < dmin:
            dmin = w
            if dmin == 1:
                break
        prev_gray = gray

    return dmin if dmin <= N else 0


def validate(solution: Any) -> ValidationResult:
    try:
        if not isinstance(solution, dict):
            return failure("Invalid format: expected a dict")

        G = solution.get("generator_matrix", None)
        if not isinstance(G, list):
            return failure("Missing or invalid 'generator_matrix' (expected list of 15 bitstrings)")

        if len(G) != K:
            return failure(f"generator_matrix must have exactly {K} rows")

        # Parse rows
        row_masks: List[int] = []
        for j, row in enumerate(G):
            if not isinstance(row, str):
                return failure(f"Row {j} is not a string")
            row_masks.append(_row_to_mask(row))

        # Rank check
        rnk = _gf2_rank(row_masks)
        if rnk != K:
            return failure(f"Rank(G) is {rnk}, expected {K}")

        # Distance computation
        dmin = _min_distance_gray(row_masks)

        return success(
            f"Valid [68,15] binary linear code verified; min distance = {dmin}",
            min_distance=int(dmin),
            n=N,
            k=K,
        )

    except Exception as e:
        return failure(f"Validation error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Validate a binary linear [68,15] code and compute min distance")
    parser.add_argument("solution", help="Solution as JSON string or path to JSON file")
    args = parser.parse_args()

    sol = load_solution(args.solution)
    result = validate(sol)
    output_result(result)


if __name__ == "__main__":
    main()