#!/usr/bin/env python3
"""
Validator for problem 080: Constant-Weight Code A(29,8,5)

A solution is a collection of 5-subsets ("blocks") of {0,...,28}
such that no unordered pair {i,j} appears in more than one block.
Equivalently, any two blocks intersect in at most one point.

We maximize the number of blocks.
"""

import argparse
from itertools import combinations
from typing import Any, Dict, List, Tuple

from . import ValidationResult, load_solution, output_result, success, failure

V = 29
K = 5  # block size
PAIR_LIMIT = 1  # each pair may appear in at most one block


def _parse_blocks(solution: Any) -> List[Tuple[int, int, int, int, int]]:
    if not isinstance(solution, dict) or "blocks" not in solution:
        raise ValueError("Expected a dict with key 'blocks'")

    blocks = solution["blocks"]
    if not isinstance(blocks, list):
        raise ValueError("'blocks' must be a list")

    parsed: List[Tuple[int, int, int, int, int]] = []
    for idx, b in enumerate(blocks):
        if not isinstance(b, list) or len(b) != K:
            raise ValueError(f"Block {idx} must be a list of length {K}")
        if any((not isinstance(x, int)) for x in b):
            raise ValueError(f"Block {idx} contains a non-integer")
        if any((x < 0 or x >= V) for x in b):
            raise ValueError(f"Block {idx} has element outside [0,{V-1}]")
        if len(set(b)) != K:
            raise ValueError(f"Block {idx} has repeated elements")
        t = tuple(sorted(b))
        parsed.append(t)

    return parsed


def validate(solution: Any) -> ValidationResult:
    try:
        blocks = _parse_blocks(solution)
    except Exception as e:
        return failure(f"Failed to parse solution: {e}")

    # No duplicate blocks
    if len(set(blocks)) != len(blocks):
        return failure("Duplicate blocks are not allowed")

    # Enforce packing constraint: no pair appears in two different blocks
    pair_owner: Dict[Tuple[int, int], int] = {}
    for bi, b in enumerate(blocks):
        for i, j in combinations(b, 2):
            p = (i, j)
            if p in pair_owner:
                bj = pair_owner[p]
                return failure(
                    f"Repeated pair {p} appears in blocks {bj} and {bi}"
                )
            pair_owner[p] = bi

    num_blocks = len(blocks)
    num_pairs_covered = len(pair_owner)  # each block contributes 10 pairs if valid

    return success(
        f"Valid packing on v={V} with {num_blocks} blocks.",
        num_blocks=num_blocks,
        v=V,
        block_size=K,
        num_pairs_covered=num_pairs_covered,
    )


def main():
    parser = argparse.ArgumentParser(description="Validate A(29,8,5) packing (pairs by quintuples)")
    parser.add_argument("solution", help="Solution as JSON string or path to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    sol = load_solution(args.solution)
    result = validate(sol)
    output_result(result)


if __name__ == "__main__":
    main()