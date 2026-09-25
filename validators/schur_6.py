#!/usr/bin/env python3
"""
Validator for schur_6: Maximum 6-Coloring with No Monochromatic x+y=z

Expected input: a list `colors` of length N+1 where
  colors[0] = 0  (sentinel)
  colors[i] in {0,1,2,3,4,5} for i = 1..N

Validity: for every color c and all x,y with 1<=x<=y<=N,
  if colors[x] = colors[y] = c and x+y <= N, then colors[x+y] != c.

Metric: N (maximize).
"""

import argparse
from typing import Any

from . import ValidationResult, load_solution, output_result, success, failure

# Hard safety cap: proven upper bound S(6) <= 1836, so 10000 is generous.
MAX_N = 10000


def validate(sol: Any) -> ValidationResult:
    try:
        if not isinstance(sol, list):
            return failure(f"Expected a list, got {type(sol).__name__}.")

        if len(sol) < 2:
            return failure("colors must have length at least 2 (N >= 1).")

        N = len(sol) - 1

        if N > MAX_N:
            return failure(f"N={N} exceeds safety cap MAX_N={MAX_N}.")

        if sol[0] != 0:
            return failure("colors[0] must be 0.")

        # Validate entries and collect positions per color.
        pos = [[] for _ in range(6)]
        for i in range(1, N + 1):
            ci = sol[i]
            if not isinstance(ci, int):
                return failure(f"colors[{i}] is not an int (got {type(ci).__name__}).")
            if ci < 0 or ci > 5:
                return failure(f"colors[{i}]={ci} is out of range; must be in {{0,...,5}}.")
            pos[ci].append(i)

        # Check sum-free constraint: for each color c, for all x<=y in that color,
        # if x+y<=N then colors[x+y] must not equal c.
        for c in range(6):
            lst = pos[c]
            m = len(lst)
            for a in range(m):
                x = lst[a]
                for b in range(a, m):
                    y = lst[b]
                    s = x + y
                    if s > N:
                        break
                    if sol[s] == c:
                        return failure(
                            f"Monochromatic violation in color {c}: "
                            f"{x} + {y} = {s} and all have color {c}."
                        )

        return success(
            f"Valid 6-coloring of {{1,...,{N}}} with no monochromatic x+y=z.",
            N=N,
            color_sizes=[len(pos[c]) for c in range(6)],
        )

    except Exception as e:
        return failure(f"Exception during validation: {e}")


def main():
    parser = argparse.ArgumentParser(description="Validate a 6-coloring for Schur number S(6)")
    parser.add_argument("solution", help="Solution as JSON string or path to JSON file")
    args = parser.parse_args()

    sol = load_solution(args.solution)
    result = validate(sol)
    output_result(result)


if __name__ == "__main__":
    main()
