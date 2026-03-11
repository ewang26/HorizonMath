#!/usr/bin/env python3
"""
Validator for dts_7_5_min_scope: Minimum-Scope (7,5)-Difference Triangle Set

Expected input:
{
  "n": 7,
  "k": 5,
  "rows": [
    [0, ..., ..., ..., ..., ...],
    ...
    (7 rows total)
  ]
}

Validity:
- n == 7, k == 5
- Each row length == k+1 == 6
- Each row is strictly increasing and starts with 0
- All positive within-row differences are distinct across ALL rows

Metric:
- scope = max entry in the array (minimize)
"""

import argparse
from typing import Any

from . import ValidationResult, load_solution, output_result, success, failure


def validate(sol: Any) -> ValidationResult:
    try:
        if not isinstance(sol, dict):
            return failure("Invalid format: expected dict.")

        n = int(sol.get("n", -1))
        k = int(sol.get("k", -1))
        rows = sol.get("rows", None)

        if n != 7 or k != 5:
            return failure("This benchmark requires n=7 and k=5 exactly.")

        if not isinstance(rows, list) or len(rows) != n:
            return failure(f"'rows' must be a list of length {n}.")

        # Check rows and collect differences
        seen_diffs = set()
        scope = 0

        for i, row in enumerate(rows):
            if not isinstance(row, list) or len(row) != k + 1:
                return failure(f"Row {i} must be a list of length {k+1}.")

            # Check integers and increasing
            try:
                r = [int(x) for x in row]
            except Exception:
                return failure(f"Row {i} contains non-integer values.")

            if r[0] != 0:
                return failure(f"Row {i} must start with 0 (normalized).")
            for j in range(1, k + 1):
                if r[j] <= r[j - 1]:
                    return failure(f"Row {i} must be strictly increasing.")

            scope = max(scope, r[-1])

            # Positive differences within this row
            for a in range(k + 1):
                for b in range(a):
                    d = r[a] - r[b]  # positive since increasing and a>b
                    if d <= 0:
                        return failure("Non-positive difference encountered (should be impossible).")
                    if d in seen_diffs:
                        return failure(f"Duplicate difference {d} found (violates DTS property).")
                    seen_diffs.add(d)

        metrics = {"scope": scope}
        return success("Valid (7,5)-DTS.", metrics=metrics)

    except Exception as e:
        return failure(f"Exception during validation: {e}")


def main():
    parser = argparse.ArgumentParser(description="Validate a (7,5)-Difference Triangle Set")
    parser.add_argument("solution", help="Solution as JSON string or path to JSON file")
    args = parser.parse_args()

    sol = load_solution(args.solution)
    result = validate(sol)
    output_result(result)


if __name__ == "__main__":
    main()
