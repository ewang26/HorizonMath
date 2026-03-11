#!/usr/bin/env python3
"""
Validator for 078_covering_C13_k7_t4

We validate a covering design for C(13,7,4):
- Universe: {0,1,...,12}
- Blocks: 7-subsets
- Coverage: every 4-subset is contained in at least one block

Metric (MINIMIZE):
- num_blocks = number of blocks

Solution format:
{
  "blocks": [[...7 ints...], [...], ...]
}
"""

import argparse
from itertools import combinations
from typing import Any, List, Set, Tuple

from . import (
    ValidationResult,
    load_solution,
    output_result,
    success,
    failure,
)

V = 13
K = 7
T = 4


def _block_to_mask(block: List[int]) -> int:
    m = 0
    for x in block:
        m |= 1 << x
    return m


def validate(solution: Any) -> ValidationResult:
    if not isinstance(solution, dict):
        return failure("Invalid format: expected dict")

    blocks = solution.get("blocks", None)
    if not isinstance(blocks, list) or len(blocks) == 0:
        return failure("Missing or empty 'blocks' list")

    seen_blocks: Set[Tuple[int, ...]] = set()
    masks: List[int] = []

    for idx, b in enumerate(blocks):
        if not isinstance(b, list):
            return failure(f"Block {idx} is not a list")

        if len(b) != K:
            return failure(f"Block {idx} must have exactly {K} elements")

        if not all(isinstance(x, int) for x in b):
            return failure(f"Block {idx} must contain integers only")

        if any(x < 0 or x >= V for x in b):
            return failure(f"Block {idx} has element outside 0..{V-1}")

        if len(set(b)) != K:
            return failure(f"Block {idx} has duplicate elements")

        bt = tuple(sorted(b))
        if bt in seen_blocks:
            return failure(f"Duplicate block detected at index {idx}")
        seen_blocks.add(bt)

        masks.append(_block_to_mask(list(bt)))

    # Check coverage of all 4-subsets
    uncovered = 0
    for comb4 in combinations(range(V), T):
        target = 0
        for x in comb4:
            target |= 1 << x
        if not any((m & target) == target for m in masks):
            uncovered += 1
            # Early exit after finding a few uncovered sets
            if uncovered >= 5:
                return failure("Not a valid cover: found uncovered 4-subsets")

    if uncovered != 0:
        return failure(f"Not a valid cover: {uncovered} uncovered 4-subsets")

    metrics = {
        "v": V,
        "k": K,
        "t": T,
        "num_blocks": len(masks),
    }
    return success("Valid covering design.", **metrics)


def main():
    parser = argparse.ArgumentParser(description="Validate C(13,7,4) covering design")
    parser.add_argument("solution", help="Solution as JSON string or path to JSON file")
    args = parser.parse_args()

    sol = load_solution(args.solution)
    res = validate(sol)
    output_result(res)


if __name__ == "__main__":
    main()