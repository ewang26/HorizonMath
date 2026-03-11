#!/usr/bin/env python3
"""
Validator for problem 075: 2-Coloring of Kₙ Without Monochromatic K₅

Find a 2-coloring of the edges of Kₙ (complete graph on n vertices)
such that there is no monochromatic K₅ (clique of size 5).

This is related to Ramsey numbers: R(5,5) = 43-48 (bounds).

Expected input format:
    {
        "n": number of vertices,
        "coloring": [[0, 1, 0, ...], ...]  # nxn matrix, colors 0 and 1
    }
    or
    {
        "n": number of vertices,
        "red_edges": [[u, v], ...],  # edges of color 0/red
        "blue_edges": [[u, v], ...]  # edges of color 1/blue (optional, complement)
    }
"""

import argparse
from itertools import combinations
from typing import Any

import numpy as np

from . import ValidationResult, load_solution, output_result, success, failure


CLIQUE_SIZE = 5


def has_monochromatic_clique(adj: np.ndarray, n: int, k: int) -> tuple[bool, int]:
    """
    Check if adjacency matrix has a clique of size k.

    Returns (has_clique, color_with_clique or -1).
    """
    for vertices in combinations(range(n), k):
        # Check if all edges present (clique)
        is_clique = True
        for i, v1 in enumerate(vertices):
            for v2 in vertices[i+1:]:
                if not adj[v1, v2]:
                    is_clique = False
                    break
            if not is_clique:
                break
        if is_clique:
            return True, -1  # Found clique

    return False, -1


def validate(solution: Any) -> ValidationResult:
    """
    Validate a 2-coloring of Kₙ has no monochromatic K₅.

    Args:
        solution: Dict with 'n' and coloring information

    Returns:
        ValidationResult with verification status
    """
    try:
        if not isinstance(solution, dict):
            return failure("Invalid format: expected dict")

        n = int(solution.get('n', 0))
        if n < CLIQUE_SIZE:
            return success(
                f"K_{n} trivially has no K_{CLIQUE_SIZE}",
                num_vertices=n
            )

        if 'coloring' in solution:
            coloring = np.array(solution['coloring'], dtype=int)
            if coloring.shape != (n, n):
                return failure(f"Coloring matrix must be {n}x{n}")

            # Validate off-diagonal entries are binary
            mask = ~np.eye(n, dtype=bool)
            if not np.all((coloring[mask] == 0) | (coloring[mask] == 1)):
                return failure("Coloring matrix must have entries 0 or 1 on off-diagonal")

            # Enforce symmetry on off-diagonal
            if not np.all(coloring[mask] == coloring.T[mask]):
                return failure("Coloring matrix must be symmetric (coloring[i][j] == coloring[j][i])")

            # Build red and blue adjacency from off-diagonal entries
            red_adj = (coloring == 0) & mask
            blue_adj = (coloring == 1) & mask

        elif 'red_edges' in solution:
            red_edges = solution['red_edges']
            red_adj = np.zeros((n, n), dtype=bool)
            for u, v in red_edges:
                if not (isinstance(u, (int, np.integer)) and isinstance(v, (int, np.integer))):
                    return failure(f"Edge ({u}, {v}) must be a pair of integers")
                if u < 0 or u >= n or v < 0 or v >= n:
                    return failure(f"Edge ({u}, {v}) has vertex out of range [0, {n-1}]")
                if u == v:
                    return failure(f"Self-loop at vertex {u}")
                red_adj[u, v] = red_adj[v, u] = True

            # Blue is complement
            blue_adj = np.ones((n, n), dtype=bool)
            np.fill_diagonal(blue_adj, False)
            blue_adj = blue_adj & ~red_adj

        else:
            return failure("Need 'coloring' matrix or 'red_edges' list")

    except (ValueError, TypeError, IndexError) as e:
        return failure(f"Failed to parse solution: {e}")

    # Check for monochromatic K₅ in red
    has_red_clique, _ = has_monochromatic_clique(red_adj, n, CLIQUE_SIZE)
    if has_red_clique:
        return failure(f"Found monochromatic K_{CLIQUE_SIZE} in red")

    # Check for monochromatic K₅ in blue
    has_blue_clique, _ = has_monochromatic_clique(blue_adj, n, CLIQUE_SIZE)
    if has_blue_clique:
        return failure(f"Found monochromatic K_{CLIQUE_SIZE} in blue")

    red_edge_count = np.sum(red_adj) // 2
    blue_edge_count = np.sum(blue_adj) // 2
    total_edges = n * (n - 1) // 2

    return success(
        f"Valid 2-coloring of K_{n} with no monochromatic K_{CLIQUE_SIZE}",
        num_vertices=n,
        red_edges=int(red_edge_count),
        blue_edges=int(blue_edge_count),
        total_edges=total_edges
    )


def main():
    parser = argparse.ArgumentParser(description='Validate Ramsey coloring avoiding monochromatic K_5')
    parser.add_argument('solution', help='Solution as JSON string or path to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    solution = load_solution(args.solution)
    result = validate(solution)
    output_result(result)


if __name__ == '__main__':
    main()
