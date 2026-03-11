#!/usr/bin/env python3
"""
Validator for problem 041: Improve Upper Bound on Difference Basis Constant

For any natural number n, Δ(n) is the size of the smallest set B of integers such
that every k in {1, ..., n} is expressible as |a-b| for some a,b ∈ B.

We validate a proposed (n, B) and compute ratio = |B|^2 / n, which is a certified
upper bound on C^6.7 = inf_{n>=1} Δ(n)^2/n.

Expected input format:
    {"n": <int>, "basis": [b0, b1, b2, ...]}
"""

import argparse
from typing import Any

from . import ValidationResult, load_solution, parse_integer, output_result, success, failure

MAX_BASIS_SIZE = 20000
MAX_N = 50_000_000  # memory guard for bytearray(n+1)


def validate(solution: Any) -> ValidationResult:
    try:
        if not isinstance(solution, dict):
            return failure("Invalid format: expected dict with 'n' and 'basis' keys")

        n = parse_integer(solution["n"])
        raw_basis = solution["basis"]
        if not isinstance(raw_basis, (list, tuple)):
            return failure("Invalid format: 'basis' must be a list of integers")
        basis = [parse_integer(b) for b in raw_basis]
    except (KeyError, ValueError, TypeError) as e:
        return failure(f"Failed to parse solution: {e}")

    if n < 1:
        return failure(f"n must be positive, got {n}")

    B = sorted(set(basis))
    m = len(B)

    if m == 0:
        return failure("Basis must contain at least one integer")

    if m > MAX_BASIS_SIZE:
        return failure(f"Basis too large: |B|={m} exceeds MAX_BASIS_SIZE={MAX_BASIS_SIZE}")

    if n > MAX_N:
        return failure(f"n too large: n={n} exceeds MAX_N={MAX_N}")

    # Necessary condition: at most m*(m-1)/2 distinct positive differences exist.
    if m * (m - 1) // 2 < n:
        return failure(
            f"Impossible coverage: |B|={m} allows at most {m*(m-1)//2} distinct positive differences "
            f"but need to cover n={n} values (1..n).",
            n=n,
            basis_size=m,
        )

    # Normalize by translation invariance (differences unchanged by shifting).
    shift = B[0]
    B = [x - shift for x in B]

    covered = bytearray(n + 1)  # covered[d] = 1 iff difference d is achieved

    for i in range(m):
        a = B[i]
        j = i + 1
        while j < m:
            d = B[j] - a
            if d > n:
                break
            covered[d] = 1
            j += 1

    missing_count = 0
    examples = []
    for d in range(1, n + 1):
        if covered[d] == 0:
            missing_count += 1
            if len(examples) < 5:
                examples.append(d)

    if missing_count:
        return failure(
            f"Not a difference basis for {{1..{n}}}: missing {missing_count} values. Examples: {examples}",
            missing_count=missing_count,
            missing_examples=examples,
            n=n,
            basis_size=m,
        )

    ratio = (m * m) / n
    return success(
        f"Verified difference basis for n={n}: |B|={m}, |B|^2/n = {ratio:.6f}",
        n=n,
        basis_size=m,
        ratio=ratio,
    )


def main():
    parser = argparse.ArgumentParser(description="Validate difference basis construction")
    parser.add_argument("solution", help="Solution as JSON string or path to JSON file")
    parser.add_argument("--verbose", "-v', action='store_true", help="Verbose output")
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == "__main__":
    main()