#!/usr/bin/env python3
"""
Validator for problem: Hadamard Matrix of Order 716 via Goethals-Seidel construction

Validates that four ±1 sequences of length 179 define circulant matrices A, B, C, D
satisfying AA^T + BB^T + CC^T + DD^T = 716·I, which yields a Hadamard matrix of
order 716 via the Goethals-Seidel array.

Expected input format:
    {"rows": [[...], [...], [...], [...]]}   # four sequences of length 179
"""

import argparse
from typing import Any

import numpy as np
from scipy.linalg import circulant

from . import ValidationResult, load_solution, output_result, success, failure


TARGET_ORDER = 716
BLOCK_ORDER = 179  # 716 / 4


def validate(solution: Any) -> ValidationResult:
    """
    Validate a Goethals-Seidel certificate for a Hadamard matrix of order 716.

    The solution must provide four ±1 sequences of length 179 (first rows of
    circulant matrices A, B, C, D) such that AA^T + BB^T + CC^T + DD^T = 716·I.

    The validator then assembles the full 716×716 Hadamard matrix via the
    Goethals-Seidel array and verifies H·H^T = 716·I.

    Args:
        solution: Dict with 'rows' key containing four lists of length 179

    Returns:
        ValidationResult with success/failure
    """
    # --- Parse input ---
    try:
        if isinstance(solution, dict) and 'rows' in solution:
            rows = solution['rows']
        elif isinstance(solution, list) and len(solution) == 4:
            rows = solution
        else:
            return failure(
                "Invalid format: expected {\"rows\": [a, b, c, d]} "
                "where a, b, c, d are ±1 sequences of length 179"
            )

        if len(rows) != 4:
            return failure(f"Expected exactly 4 sequences, got {len(rows)}")

        for i, row in enumerate(rows):
            if len(row) != BLOCK_ORDER:
                return failure(
                    f"Sequence {i} has length {len(row)}, expected {BLOCK_ORDER}"
                )

        seqs = [np.array(row, dtype=np.int64) for row in rows]
    except (ValueError, TypeError) as e:
        return failure(f"Failed to parse sequences: {e}")

    # --- Check entries are ±1 ---
    for i, seq in enumerate(seqs):
        if not np.all((seq == 1) | (seq == -1)):
            invalid_count = int(np.sum((seq != 1) & (seq != -1)))
            return failure(
                f"Sequence {i} must have entries ±1, found {invalid_count} invalid entries"
            )

    # --- Build circulant matrices ---
    n = BLOCK_ORDER
    A, B, C, D = [circulant(seq) for seq in seqs]

    # --- Check core condition: AA^T + BB^T + CC^T + DD^T = 4n·I ---
    gram_sum = A @ A.T + B @ B.T + C @ C.T + D @ D.T
    expected = TARGET_ORDER * np.eye(n, dtype=np.int64)

    if not np.array_equal(gram_sum, expected):
        diff_mask = gram_sum != expected
        diff_count = int(np.sum(diff_mask))
        idx = np.argwhere(diff_mask)[0]
        i, j = idx
        return failure(
            f"AA^T + BB^T + CC^T + DD^T ≠ {TARGET_ORDER}·I. "
            f"Found {diff_count} incorrect entries. "
            f"Example: position ({i},{j}) has {gram_sum[i,j]}, expected {expected[i,j]}",
            differences=diff_count
        )

    # --- Assemble full Hadamard matrix via Goethals-Seidel array ---
    # R is the back-circulant (reversal) matrix: R[i,j] = delta(i+j, n-1)
    R = np.fliplr(np.eye(n, dtype=np.int64))

    BR = B @ R
    CR = C @ R
    DR = D @ R
    BtR = B.T @ R
    CtR = C.T @ R
    DtR = D.T @ R

    H = np.block([
        [ A,   BR,   CR,   DR ],
        [-BR,  A,    DtR, -CtR],
        [-CR, -DtR,  A,    BtR],
        [-DR,  CtR, -BtR,  A  ]
    ])

    # --- Final verification: H·H^T = 716·I ---
    HHT = H @ H.T
    full_expected = TARGET_ORDER * np.eye(TARGET_ORDER, dtype=np.int64)

    if not np.array_equal(HHT, full_expected):
        diff_mask = HHT != full_expected
        diff_count = int(np.sum(diff_mask))
        idx = np.argwhere(diff_mask)[0]
        i, j = idx
        return failure(
            f"Assembled H·H^T ≠ {TARGET_ORDER}·I. "
            f"Found {diff_count} incorrect entries. "
            f"Example: position ({i},{j}) has {HHT[i,j]}, expected {full_expected[i,j]}",
            differences=diff_count
        )

    return success(
        f"Verified: Goethals-Seidel construction yields {TARGET_ORDER}×{TARGET_ORDER} "
        f"Hadamard matrix with H·H^T = {TARGET_ORDER}·I",
        order=TARGET_ORDER,
        block_order=BLOCK_ORDER
    )


def main():
    parser = argparse.ArgumentParser(
        description='Validate Goethals-Seidel certificate for Hadamard matrix of order 716'
    )
    parser.add_argument('solution', help='Solution as JSON string or path to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == '__main__':
    main()