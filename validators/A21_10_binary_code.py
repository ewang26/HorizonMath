#!/usr/bin/env python3
"""
Validator for problem 079: Binary Code A(21,10)

We validate a binary code C ⊆ {0,1}^21 with minimum Hamming distance ≥ 10.

Expected input format:
    {"codewords": [int|str, ...]}

- int codewords are interpreted as 21-bit vectors (0 <= x < 2^21)
- str codewords must be length 21 over {'0','1'}

Metric:
- number_of_codewords = |C|   (to be maximized)
"""

import argparse
from typing import Any, List

from . import ValidationResult, load_solution, output_result, success, failure

N = 21
D = 10
MAX_WORD = 1 << N


def _parse_codeword(w: Any) -> int:
    if isinstance(w, int):
        if 0 <= w < MAX_WORD:
            return w
        raise ValueError(f"Integer codeword out of range [0, 2^{N}): {w}")

    if isinstance(w, str):
        if len(w) != N:
            raise ValueError(f"String codeword must have length {N}: got {len(w)}")
        if any(c not in "01" for c in w):
            raise ValueError("String codeword must contain only '0' and '1'")
        return int(w, 2)

    raise ValueError(f"Unsupported codeword type: {type(w)}")


def validate(solution: Any) -> ValidationResult:
    if not isinstance(solution, dict) or "codewords" not in solution:
        return failure("Invalid format: expected dict with key 'codewords'")

    raw = solution["codewords"]
    if not isinstance(raw, list):
        return failure("Invalid format: 'codewords' must be a list")

    try:
        words: List[int] = [_parse_codeword(w) for w in raw]
    except ValueError as e:
        return failure(f"Failed to parse codewords: {e}")

    # Enforce uniqueness
    uniq = list(dict.fromkeys(words))
    if len(uniq) != len(words):
        return failure(f"Duplicate codewords detected: {len(words) - len(uniq)} duplicates")

    m = len(uniq)
    if m == 0:
        return failure("Empty code is not allowed")

    # Check minimum distance (pairwise)
    # Use XOR + popcount (int.bit_count) for speed.
    min_dist = N + 1
    for i in range(m):
        wi = uniq[i]
        for j in range(i + 1, m):
            dist = (wi ^ uniq[j]).bit_count()
            if dist < D:
                return failure(
                    f"Distance violation: codewords {i} and {j} have distance {dist} < {D}",
                    number_of_codewords=m,
                    min_distance=dist
                )
            if dist < min_dist:
                min_dist = dist

    if min_dist == N + 1:
        min_dist = N  # single-word code case (but we disallow empty only)

    return success(
        f"Valid code of length {N} with min distance >= {D}. Size = {m}, min distance = {min_dist}.",
        number_of_codewords=m,
        min_distance=min_dist,
        n=N,
        d=D
    )


def main():
    parser = argparse.ArgumentParser(description="Validate binary code A(21,10)")
    parser.add_argument("solution", help="Solution as JSON string or path to JSON file")
    args = parser.parse_args()

    sol = load_solution(args.solution)
    result = validate(sol)
    output_result(result)


if __name__ == "__main__":
    main()