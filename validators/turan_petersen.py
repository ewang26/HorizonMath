#!/usr/bin/env python3
"""
Validator for problem 074_turan_petersen: Petersen Graph Turán Problem (n=50).

Checks:
- solution is a dict with fields {"n": int, "edges": [[u,v], ...]}
- enforces n == 50 exactly
- simple undirected graph: no self-loops, vertices in range, duplicates ignored
- forbids the Petersen graph as a (non-induced) subgraph
Metrics:
- number_of_edges

Notes on Petersen-free checking strategy:
1) Fast certificates (always safe):
   - If the graph is bipartite => Petersen-free (Petersen is non-bipartite).
   - If the graph is exactly K2 ∇ K_{a,b} on the remaining vertices => Petersen-free
     (this includes the standard 673-edge construction K2 ∇ K_{24,24}).
2) Otherwise, run an exact backtracking subgraph search with a strict time limit.
   If it times out, we reject rather than risk a false accept.
"""

import argparse
import time
from typing import Any, List, Tuple

from . import ValidationResult, load_solution, output_result, success, failure

N_REQUIRED = 50

# Time budget (seconds) for the exact Petersen-subgraph search when no certificate applies.
PETERSEN_SEARCH_TIME_LIMIT = 3.0

# Petersen graph edges under the common labeling used by NetworkX:
# Outer cycle: 0-1-2-3-4-0
# Spokes: 0-5,1-6,2-7,3-8,4-9
# Inner cycle: 5-7-9-6-8-5
PETERSEN_EDGES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 4),
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
    (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
]


def _popcount(x: int) -> int:
    return x.bit_count()


def _build_adj_bitsets(n: int, edges: List[List[int]]):
    """Return (adj_masks, degs) for a simple undirected graph on n vertices."""
    adj = [0] * n
    deg = [0] * n
    for e in edges:
        if not (isinstance(e, (list, tuple)) and len(e) == 2):
            raise TypeError(f"Edge {e!r} is not a length-2 pair")
        u, v = e
        u = int(u)
        v = int(v)
        if u == v:
            raise ValueError(f"Self-loop at vertex {u}")
        if u < 0 or u >= n or v < 0 or v >= n:
            raise ValueError(f"Edge ({u}, {v}) has vertex out of range for n={n}")
        if u > v:
            u, v = v, u
        # ignore duplicates by checking bit
        if (adj[u] >> v) & 1:
            continue
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        deg[u] += 1
        deg[v] += 1
    return adj, deg


def _is_bipartite_bitset(adj: List[int]) -> bool:
    """Bipartite test via BFS 2-coloring on bitset adjacency (n is small)."""
    n = len(adj)
    color = [-1] * n
    for s in range(n):
        if color[s] != -1:
            continue
        color[s] = 0
        queue = [s]
        while queue:
            u = queue.pop()
            neigh_mask = adj[u]
            # iterate neighbors
            m = neigh_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return False
    return True


def _is_complete_bipartite_on_subset(adj: List[int], subset_mask: int) -> bool:
    """
    Check whether the induced subgraph on subset_mask is exactly complete bipartite K_{a,b}
    (connectedness not required, but will fail if empty/one-sided in a way that violates completeness).
    """
    # Extract subset vertices
    verts = []
    m = subset_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        m ^= lsb
        verts.append(v)
    if len(verts) == 0:
        return False

    # 2-coloring on induced subgraph
    color = {v: -1 for v in verts}
    for s in verts:
        if color[s] != -1:
            continue
        color[s] = 0
        q = [s]
        while q:
            u = q.pop()
            neigh = adj[u] & subset_mask
            mm = neigh
            while mm:
                lsb = mm & -mm
                v = lsb.bit_length() - 1
                mm ^= lsb
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    q.append(v)
                elif color[v] == color[u]:
                    return False

    A_mask = 0
    B_mask = 0
    for v in verts:
        if color[v] == 0:
            A_mask |= 1 << v
        else:
            B_mask |= 1 << v

    # Must be a bipartition (both parts non-empty) for K_{a,b} with edges present
    if A_mask == 0 or B_mask == 0:
        return False

    # Completeness: vertices in A connect to all in B and none in A; vice versa
    for v in verts:
        neigh_in_subset = adj[v] & subset_mask
        if (A_mask >> v) & 1:
            if neigh_in_subset != B_mask:
                return False
        else:
            if neigh_in_subset != A_mask:
                return False

    return True


def _is_K2_join_complete_bipartite(adj: List[int], deg: List[int]) -> bool:
    """
    Detect whether G is exactly K2 ∇ K_{a,b} for some a+b = n-2:
    - two universal vertices u,v (degree n-1),
    - u-v is an edge,
    - induced graph on remaining vertices is complete bipartite.
    """
    n = len(adj)
    universals = [i for i, d in enumerate(deg) if d == n - 1]
    if len(universals) < 2:
        return False
    u, v = universals[0], universals[1]
    if ((adj[u] >> v) & 1) == 0:
        return False

    rem_mask = ((1 << n) - 1) & ~(1 << u) & ~(1 << v)
    return _is_complete_bipartite_on_subset(adj, rem_mask)


def _contains_petersen_subgraph_exact(adj: List[int], deg: List[int], time_limit: float) -> bool | None:
    """
    Exact (non-induced) Petersen subgraph detection by backtracking with bitset adjacency.
    Returns:
      True  if Petersen found,
      False if proven Petersen-free,
      None  if timed out.
    """
    n = len(adj)
    if n < 10:
        return False
    # Quick necessary condition: must have at least 15 edges in total (not sufficient).
    if sum(deg) // 2 < 15:
        return False

    # Pattern adjacency
    m = 10
    padj = [0] * m
    pnei = [[] for _ in range(m)]
    for a, b in PETERSEN_EDGES:
        padj[a] |= 1 << b
        padj[b] |= 1 << a
    for u in range(m):
        mm = padj[u]
        while mm:
            lsb = mm & -mm
            w = lsb.bit_length() - 1
            mm ^= lsb
            pnei[u].append(w)

    # Candidates (degree >= 3 since Petersen is 3-regular)
    cand0 = 0
    for v in range(n):
        if deg[v] >= 3:
            cand0 |= 1 << v
    if _popcount(cand0) < 10:
        return False

    cand = [cand0] * m
    mapping = [-1] * m
    used = 0

    start = time.perf_counter()

    def choose_next():
        """Pick next pattern vertex with most assigned neighbors, then smallest feasible domain."""
        best_u = None
        best_key = None
        best_domain = 0

        for u in range(m):
            if mapping[u] != -1:
                continue

            req = None
            assigned = 0
            for w in pnei[u]:
                vw = mapping[w]
                if vw != -1:
                    assigned += 1
                    req = adj[vw] if req is None else (req & adj[vw])

            dom = (cand[u] if req is None else (cand[u] & req)) & ~used
            c = _popcount(dom)
            if c == 0:
                return None, 0
            key = (-assigned, c)
            if best_key is None or key < best_key:
                best_key = key
                best_u = u
                best_domain = dom

        return best_u, best_domain

    def backtrack(k: int) -> bool:
        nonlocal used
        if time.perf_counter() - start > time_limit:
            raise TimeoutError

        if k == m:
            return True

        u, dom = choose_next()
        if u is None:
            return False

        while dom:
            lsb = dom & -dom
            v = lsb.bit_length() - 1
            dom ^= lsb

            # adjacency constraints to already-mapped pattern neighbors
            ok = True
            for w in pnei[u]:
                vw = mapping[w]
                if vw != -1 and ((adj[v] >> vw) & 1) == 0:
                    ok = False
                    break
            if not ok:
                continue

            mapping[u] = v
            used_before = used
            used |= 1 << v

            if backtrack(k + 1):
                return True

            used = used_before
            mapping[u] = -1

        return False

    try:
        return backtrack(0)
    except TimeoutError:
        return None


def validate(solution: Any) -> ValidationResult:
    try:
        if not isinstance(solution, dict):
            return failure("Invalid format: expected dict with 'n' and 'edges'")

        if "n" not in solution:
            return failure("Missing required field 'n'")

        n = int(solution.get("n"))
        if n != N_REQUIRED:
            return failure(f"Invalid n: expected n={N_REQUIRED}, got n={n}")

        edges = solution.get("edges", [])
        if not isinstance(edges, list):
            return failure("Invalid 'edges': expected a list of [u,v] pairs")

        adj, deg = _build_adj_bitsets(n, edges)
        num_edges = sum(deg) // 2

    except (ValueError, TypeError) as e:
        return failure(f"Failed to parse graph: {e}")

    # Fast certificates of Petersen-freeness
    if _is_bipartite_bitset(adj):
        return success(
            f"Valid bipartite graph on {n} vertices (thus Petersen-free) with {num_edges} edges",
            num_vertices=n,
            number_of_edges=int(num_edges),
        )

    if _is_K2_join_complete_bipartite(adj, deg):
        return success(
            f"Graph matches K2 ∇ K_{{a,b}} form (Petersen-free) with {num_edges} edges",
            num_vertices=n,
            number_of_edges=int(num_edges),
        )

    # Exact (non-induced) Petersen subgraph check with time limit
    found = _contains_petersen_subgraph_exact(adj, deg, PETERSEN_SEARCH_TIME_LIMIT)
    if found is None:
        return failure(
            f"Petersen-subgraph check timed out after {PETERSEN_SEARCH_TIME_LIMIT:.1f}s; "
            f"unable to certify Petersen-freeness.",
            number_of_edges=int(num_edges),
            num_vertices=n,
        )

    if found:
        return failure(
            "Graph contains the Petersen graph as a (non-induced) subgraph",
            num_vertices=n,
            number_of_edges=int(num_edges),
        )

    return success(
        f"Valid Petersen-free graph on {n} vertices with {num_edges} edges",
        num_vertices=n,
        number_of_edges=int(num_edges),
    )


def main():
    parser = argparse.ArgumentParser(description="Validate Petersen-free graph (n=50)")
    parser.add_argument("solution", help="Solution as JSON string or path to JSON file")
    args = parser.parse_args()

    sol = load_solution(args.solution)
    result = validate(sol)
    output_result(result)


if __name__ == "__main__":
    main()
