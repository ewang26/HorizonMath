#!/usr/bin/env python3
"""
Validator for problem 105: Three Mutually Orthogonal Latin Squares of Order 10

Validates that three 10×10 Latin squares L1, L2, L3 are:
1. Each a valid Latin square (each row/column contains each symbol exactly once)
2. Mutually orthogonal (superimposing any two gives all n² ordered pairs)

Expected input format:
    {"squares": [L1, L2, L3]} where each Li is a 10×10 matrix with entries 0-9
    or [L1, L2, L3]
"""

import argparse
from typing import Any

import numpy as np

from . import ValidationResult, load_solution, output_result, success, failure


TARGET_ORDER = 10
NUM_SQUARES = 3


def is_latin_square(L: np.ndarray, n: int) -> tuple[bool, str]:
    """Check if L is a valid n×n Latin square."""
    if L.shape != (n, n):
        return False, f"Wrong shape: {L.shape}, expected ({n}, {n})"

    # Check all entries are in valid range
    if not np.all((L >= 0) & (L < n)):
        return False, "Entries must be in range [0, n-1]"

    # Check each row has all symbols
    for i in range(n):
        if len(set(L[i, :])) != n:
            return False, f"Row {i} does not contain all symbols"

    # Check each column has all symbols
    for j in range(n):
        if len(set(L[:, j])) != n:
            return False, f"Column {j} does not contain all symbols"

    return True, "Valid Latin square"


def are_orthogonal(L1: np.ndarray, L2: np.ndarray, n: int) -> tuple[bool, str]:
    """Check if two Latin squares are orthogonal."""
    # Superimpose and check all n² ordered pairs appear
    pairs = set()
    for i in range(n):
        for j in range(n):
            pair = (int(L1[i, j]), int(L2[i, j]))
            if pair in pairs:
                return False, f"Duplicate pair {pair} found"
            pairs.add(pair)

    if len(pairs) != n * n:
        return False, f"Expected {n*n} pairs, found {len(pairs)}"

    return True, "Orthogonal"


def validate(solution: Any) -> ValidationResult:
    """
    Validate three mutually orthogonal Latin squares of order 10.

    Args:
        solution: Dict with 'squares' key or list of 3 matrices

    Returns:
        ValidationResult with success/failure
    """
    try:
        if isinstance(solution, dict) and 'squares' in solution:
            squares_data = solution['squares']
        elif isinstance(solution, list) and len(solution) == NUM_SQUARES:
            squares_data = solution
        else:
            return failure(f"Invalid format: expected dict with 'squares' or list of {NUM_SQUARES} matrices")

        if len(squares_data) != NUM_SQUARES:
            return failure(f"Expected {NUM_SQUARES} Latin squares, got {len(squares_data)}")

        squares = [np.array(s, dtype=np.int64) for s in squares_data]
    except (ValueError, TypeError) as e:
        return failure(f"Failed to parse squares: {e}")

    n = TARGET_ORDER

    # Validate each is a Latin square
    for i, L in enumerate(squares):
        valid, msg = is_latin_square(L, n)
        if not valid:
            return failure(f"Square {i+1} is not a valid Latin square: {msg}")

    # Check pairwise orthogonality
    for i in range(NUM_SQUARES):
        for j in range(i + 1, NUM_SQUARES):
            orth, msg = are_orthogonal(squares[i], squares[j], n)
            if not orth:
                return failure(f"Squares {i+1} and {j+1} are not orthogonal: {msg}")

    return success(
        f"Verified: {NUM_SQUARES} mutually orthogonal Latin squares of order {n}",
        order=n, num_squares=NUM_SQUARES
    )


def main():
    parser = argparse.ArgumentParser(description='Validate 3 MOLS of order 10')
    parser.add_argument('solution', help='Solution as JSON string or path to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == '__main__':
    main()
