"""Validators for the conjecture classes in https://docs.google.com/spreadsheets/d/1wwkpiFf_e8sonJ4M6LDEuj1CsWHvI3MIVAsCa_n5YCA/edit?gid=1284609381#gid=1284609381.

The module follows the style of ``validators_11.py``:
- exact by default whenever the property is exact;
- dependency-light (standard library + sympy + networkx);
- witness-format tolerant for the common natural encodings of graphs, matrices,
  operation tables, hypergraphs, designs, and sequences;
- proof/certificate friendly where a direct bare witness would otherwise force an
  expensive global search.

Supported problem families
--------------------------
1. Turyn-type sequences TT(46).
2. Skew-Hadamard matrices of order 356.
3. Cocyclic Hadamard matrices of order 188.
4. Finite magma: E677 holds while E255 fails.
5. Line-graph inertia conjecture.
6. Lovasz deletion checker for r=5,6 (retained for diagnostic use, but
   excluded from the benchmark because realistic witnesses are impractical
   to verify with the available exact certificates).
7. Dual finite magma implication: dual(E677) holds while dual(E255) fails.
8. Max Laplacian eigenvalue upper bounds: open conjectures 44 and 46,
   plus refuted instances 11, 40, and 56.
9. Distance-spectral independence/inertia conjecture.
10. Erdos 97 convex polygon equidistance problem.
11. The finite Er91/Er96 C6 challenge associated with Erdos 811: a balanced
    6-edge-coloring with no rainbow C6.
12. Symmetric conference matrix of order 86 / SRG(85,42,20,21).
13. RSHCD(196, -1).
14. Steiner systems S(3,5,41) and S(3,6,46).
15. Seymour's second-neighborhood conjecture for oriented graphs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import functools
import itertools
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import networkx as nx
import sympy as sp

try:  # package import
    from .common import (
        ValidationResult,
        VerificationError,
        _as_nonstring_list,
        _expr_is_zero,
        _hashable_key,
        _is_nonstring_sequence,
        _is_square_matrix,
        _mapping_get_first,
        _parse_graph,
        _parse_int_matrix,
        _to_exact_sympy,
        _to_fraction,
        _to_int,
    )
except ImportError:  # standalone-file import
    from common import (
        ValidationResult,
        VerificationError,
        _as_nonstring_list,
        _expr_is_zero,
        _hashable_key,
        _is_nonstring_sequence,
        _is_square_matrix,
        _mapping_get_first,
        _parse_graph,
        _parse_int_matrix,
        _to_exact_sympy,
        _to_fraction,
        _to_int,
    )

X = sp.symbols("x")


# ---------------------------------------------------------------------------
# Small exact arithmetic helpers
# ---------------------------------------------------------------------------


def _element_labels_or_none(value: Any, *, name: str) -> list[Any] | None:
    if value is None:
        return None
    labels = [
        _hashable_key(label)
        for label in _as_nonstring_list(value, name=f"{name} element labels")
    ]
    try:
        set(labels)
    except TypeError as exc:
        raise VerificationError(f"{name}: element labels must be hashable") from exc
    return labels


def _sign_of_exact_real(expr: sp.Expr, *, name: str = "expression") -> int:
    """Return the exact sign of a real SymPy expression when SymPy can prove it."""
    simplified = sp.simplify(expr)
    if simplified == 0:
        return 0
    if simplified.is_Rational is True:
        return 1 if simplified > 0 else -1
    if simplified.is_positive is True:
        return 1
    if simplified.is_negative is True:
        return -1
    sign_expr = sp.sign(simplified).doit()
    if sign_expr == 1:
        return 1
    if sign_expr == -1:
        return -1
    if sign_expr == 0:
        return 0
    raise VerificationError(f"could not determine the exact sign of {name}")


def _fraction_from_sympy_rational(value: Any) -> Fraction:
    expr = sp.Rational(value)
    return Fraction(int(expr.p), int(expr.q))


# ---------------------------------------------------------------------------
# Generic graph parsing and exact spectral helpers
# ---------------------------------------------------------------------------


def _adjacency_rows(G: nx.Graph) -> list[list[int]]:
    n = G.number_of_nodes()
    rows = [[0] * n for _ in range(n)]
    for u, v in G.edges():
        rows[u][v] = 1
        rows[v][u] = 1
    return rows


def _laplacian_rows(G: nx.Graph) -> list[list[int]]:
    n = G.number_of_nodes()
    rows = [[0] * n for _ in range(n)]
    degrees = dict(G.degree())
    for i in range(n):
        rows[i][i] = degrees[i]
    for u, v in G.edges():
        rows[u][v] = -1
        rows[v][u] = -1
    return rows


def _sympy_matrix_from_rows(rows: Sequence[Sequence[int]]) -> sp.Matrix:
    return sp.Matrix([[sp.Integer(v) for v in row] for row in rows])


def _charpoly_from_integer_symmetric_rows(rows: Sequence[Sequence[int]]) -> sp.Poly:
    if not rows:
        raise VerificationError("spectral validation needs a nonempty matrix")
    if not _is_square_matrix(rows):
        raise VerificationError("spectral validation needs a square matrix")
    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            if rows[i][j] != rows[j][i]:
                raise VerificationError("spectral validation matrix must be symmetric")
    return _sympy_matrix_from_rows(rows).charpoly(X)


def _inertia_from_integer_symmetric_rows(rows: Sequence[Sequence[int]]) -> tuple[int, int, int]:
    """Return (positive, negative, zero) inertia, counting algebraic multiplicity exactly."""
    P = _charpoly_from_integer_symmetric_rows(rows)
    intervals = P.intervals(eps=sp.Rational(1, 2**40))
    positive = negative = zero = 0
    unresolved: list[tuple[Any, int]] = []
    for (lo_raw, hi_raw), mult in intervals:
        lo = sp.Rational(lo_raw)
        hi = sp.Rational(hi_raw)
        if hi < 0:
            negative += int(mult)
        elif lo > 0:
            positive += int(mult)
        elif lo == 0 and hi == 0:
            zero += int(mult)
        else:
            unresolved.append(((lo, hi), int(mult)))
    if unresolved:
        # This should happen only if 0 is a root but was not isolated as (0,0), or if a
        # nonzero root interval straddles 0. Refine before giving up.
        for bits in (80, 160, 320, 640):
            positive = negative = zero = 0
            unresolved = []
            for (lo_raw, hi_raw), mult in P.intervals(eps=sp.Rational(1, 2**bits)):
                lo = sp.Rational(lo_raw)
                hi = sp.Rational(hi_raw)
                if hi < 0:
                    negative += int(mult)
                elif lo > 0:
                    positive += int(mult)
                elif lo == 0 and hi == 0:
                    zero += int(mult)
                else:
                    unresolved.append(((lo, hi), int(mult)))
            if not unresolved:
                break
        if unresolved:
            raise VerificationError("could not isolate eigenvalue signs exactly")
    return positive, negative, zero


@dataclass(frozen=True)
class RadicalBound:
    """A real bound of the form rational_part + sqrt(radicand), or just rational_part."""

    rational_part: Fraction
    radicand: Fraction | None = None

    def __post_init__(self) -> None:
        if self.radicand is not None and self.radicand < 0:
            raise VerificationError("a square-root bound has negative radicand")


def _compare_fraction_to_radical_bound(value: Fraction, bound: RadicalBound) -> int:
    """Compare an exact rational ``value`` with ``bound``.

    Returns 1, 0, -1 according as value is greater than, equal to, or less than the bound.
    """
    a = bound.rational_part
    if bound.radicand is None:
        return 1 if value > a else (-1 if value < a else 0)
    s = bound.radicand
    diff = value - a
    if s == 0:
        return 1 if diff > 0 else (-1 if diff < 0 else 0)
    if diff <= 0:
        return -1
    sq = diff * diff
    return 1 if sq > s else (-1 if sq < s else 0)


def _largest_root_interval(poly: sp.Poly, *, bits: int) -> tuple[Fraction, Fraction]:
    intervals = poly.intervals(eps=sp.Rational(1, 2**bits))
    if not intervals:
        raise VerificationError("matrix characteristic polynomial has no real roots")
    max_interval = max(intervals, key=lambda item: sp.Rational(item[0][1]))[0]
    lo = _fraction_from_sympy_rational(max_interval[0])
    hi = _fraction_from_sympy_rational(max_interval[1])
    return lo, hi


def _rayleigh_quotient_for_rows(rows: Sequence[Sequence[int]], vector: Any) -> Fraction:
    n = len(rows)
    values = _as_nonstring_list(vector, name="Rayleigh vector")
    if len(values) != n:
        raise VerificationError("Rayleigh vector has the wrong length")
    x = [
        _to_fraction(v, name=f"rayleigh_vector[{i}]")
        for i, v in enumerate(values)
    ]
    denom = sum(v * v for v in x)
    if denom == 0:
        raise VerificationError("Rayleigh vector must be nonzero")
    numer = Fraction(0)
    for i in range(n):
        row_sum = Fraction(0)
        for j in range(n):
            if rows[i][j]:
                row_sum += Fraction(rows[i][j]) * x[j]
        numer += x[i] * row_sum
    return numer / denom


# ---------------------------------------------------------------------------
# Generic exact matrix parsing/checking helpers
# ---------------------------------------------------------------------------


def _parse_sign(value: Any, *, name: str, allow_zero: bool = False) -> int:
    if isinstance(value, str):
        s = value.strip()
        if s in {"+", "+1", "1"}:
            return 1
        if s in {"-", "-1", "−", "−1"}:
            return -1
        if allow_zero and s == "0":
            return 0
        raise VerificationError(f"{name} must be a sign")
    z = _to_int(value, name=name)
    if z in (1, -1):
        return z
    if allow_zero and z == 0:
        return 0
    raise VerificationError(f"{name} must be {'0 or ' if allow_zero else ''}±1")


def _parse_pm1_sequence(seq_data: Any, *, length: int | None = None, name: str = "sequence") -> list[int]:
    """Parse a ±1 sequence.

    Accepted forms include lists/tuples of ±1, strings such as '+-++', and dict wrappers:
    - {"sequence": ...}
    - {"minus_positions": [...], "length": n} with 0-based indices of -1 entries.
    """
    if isinstance(seq_data, Mapping):
        if "sequence" in seq_data:
            embedded_length_raw = _mapping_get_first(seq_data, "length", "n")
            embedded_length = (
                None
                if embedded_length_raw is None
                else _to_int(embedded_length_raw, name=f"{name}.length")
            )
            if (
                length is not None
                and embedded_length is not None
                and length != embedded_length
            ):
                raise VerificationError(
                    f"{name}: supplied length conflicts with the required length"
                )
            target_length = length if length is not None else embedded_length
            return _parse_pm1_sequence(
                seq_data["sequence"], length=target_length, name=name
            )
        for key in ("minus_positions", "negative_positions", "negatives", "minus_indices"):
            if key in seq_data:
                embedded_length_raw = _mapping_get_first(
                    seq_data, "length", "n"
                )
                embedded_length = (
                    None
                    if embedded_length_raw is None
                    else _to_int(embedded_length_raw, name=f"{name}.length")
                )
                if (
                    length is not None
                    and embedded_length is not None
                    and length != embedded_length
                ):
                    raise VerificationError(
                        f"{name}: supplied length conflicts with the required length"
                    )
                n_raw = length if length is not None else embedded_length
                if n_raw is None:
                    raise VerificationError(f"{name}: bitset format needs an explicit length")
                n = _to_int(n_raw, name=f"{name}.length")
                if n < 0:
                    raise VerificationError(f"{name}.length must be nonnegative")
                out = [1] * n
                seen_positions: set[int] = set()
                for idx_raw in _as_nonstring_list(
                    seq_data[key], name=f"{name}.{key}"
                ):
                    idx = _to_int(idx_raw, name=f"{name}.{key}")
                    if not 0 <= idx < n:
                        raise VerificationError(f"{name}: negative-position index out of range")
                    if idx in seen_positions:
                        raise VerificationError(f"{name}: duplicate negative-position index")
                    seen_positions.add(idx)
                    out[idx] = -1
                return out
        raise VerificationError(f"unsupported {name} dict format")

    if isinstance(seq_data, str):
        s = seq_data.strip()
        if not s:
            raise VerificationError(f"{name} cannot be empty")
        if all(ch in "+-−" for ch in s):
            out = [1 if ch == "+" else -1 for ch in s]
        elif all(ch in "01" for ch in s):
            # Common bitstring convention: 0 encodes +1, 1 encodes -1.
            out = [1 if ch == "0" else -1 for ch in s]
        else:
            tokens = [tok for tok in s.replace(",", " ").split() if tok]
            if not tokens:
                raise VerificationError(f"{name} cannot be empty")
            out = [_parse_sign(tok, name=name) for tok in tokens]
    else:
        values = _as_nonstring_list(seq_data, name=name)
        out = [
            _parse_sign(v, name=f"{name}[{i}]") for i, v in enumerate(values)
        ]
    if length is not None and len(out) != length:
        raise VerificationError(f"{name} must have length {length}")
    return out


def _parse_pm1_matrix(data: Any, *, n: int | None = None, name: str = "matrix") -> list[list[int]]:
    """Parse a ±1 matrix.

    Accepted forms:
    - raw list of rows, with entries ±1 or '+'/'-';
    - dict wrappers {"matrix": ...}, {"H": ...};
    - {"negative_positions": [(i,j), ...], "n": n};
    - {"row_negative_positions": [[...], ...], "n": n};
    - {"bit_rows": ["0101", ...]}, with 0 -> +1 and 1 -> -1.
    """
    if isinstance(data, Mapping):
        for key in ("matrix", "H", "M", "array"):
            if key in data:
                embedded_n_raw = _mapping_get_first(data, "n", "order", "size")
                embedded_n = (
                    None
                    if embedded_n_raw is None
                    else _to_int(embedded_n_raw, name=f"{name}.n")
                )
                if n is not None and embedded_n is not None and n != embedded_n:
                    raise VerificationError(
                        f"{name}: supplied order conflicts with the required order"
                    )
                target_n = n if n is not None else embedded_n
                return _parse_pm1_matrix(data[key], n=target_n, name=name)
        for key in ("negative_positions", "minus_positions"):
            if key in data:
                embedded_n_raw = _mapping_get_first(data, "n", "order", "size")
                embedded_n = (
                    None
                    if embedded_n_raw is None
                    else _to_int(embedded_n_raw, name=f"{name}.n")
                )
                if n is not None and embedded_n is not None and n != embedded_n:
                    raise VerificationError(
                        f"{name}: supplied order conflicts with the required order"
                    )
                n_raw = n if n is not None else embedded_n
                if n_raw is None:
                    raise VerificationError(f"{name}: negative_positions format needs n/order")
                m = _to_int(n_raw, name=f"{name}.n")
                if m < 0:
                    raise VerificationError(f"{name}.n must be nonnegative")
                rows = [[1] * m for _ in range(m)]
                seen_positions: set[tuple[int, int]] = set()
                for pos_raw in _as_nonstring_list(
                    data[key], name=f"{name}.{key}"
                ):
                    pos = _as_nonstring_list(
                        pos_raw, name=f"{name} negative position"
                    )
                    if len(pos) != 2:
                        raise VerificationError(f"{name}: each negative position must be a pair")
                    i = _to_int(pos[0], name=f"{name}.negative_positions row")
                    j = _to_int(pos[1], name=f"{name}.negative_positions col")
                    if not (0 <= i < m and 0 <= j < m):
                        raise VerificationError(f"{name}: negative position out of range")
                    if (i, j) in seen_positions:
                        raise VerificationError(f"{name}: duplicate negative position")
                    seen_positions.add((i, j))
                    rows[i][j] = -1
                return rows
        for key in ("row_negative_positions", "rows_negative_positions", "negative_positions_by_row"):
            if key in data:
                row_sets = _as_nonstring_list(
                    data[key], name=f"{name}.{key}"
                )
                embedded_n_raw = _mapping_get_first(data, "n", "order", "size")
                embedded_n = (
                    None
                    if embedded_n_raw is None
                    else _to_int(embedded_n_raw, name=f"{name}.n")
                )
                if n is not None and embedded_n is not None and n != embedded_n:
                    raise VerificationError(
                        f"{name}: supplied order conflicts with the required order"
                    )
                m_raw = (
                    n
                    if n is not None
                    else embedded_n if embedded_n is not None else len(row_sets)
                )
                m = _to_int(m_raw, name=f"{name}.n")
                if len(row_sets) != m:
                    raise VerificationError(f"{name}: row_negative_positions has wrong number of rows")
                if m < 0:
                    raise VerificationError(f"{name}.n must be nonnegative")
                rows = [[1] * m for _ in range(m)]
                for i, cols in enumerate(row_sets):
                    seen_cols: set[int] = set()
                    for c_raw in _as_nonstring_list(
                        cols, name=f"{name}.{key}[{i}]"
                    ):
                        j = _to_int(c_raw, name=f"{name}.row_negative_positions col")
                        if not 0 <= j < m:
                            raise VerificationError(f"{name}: negative column out of range")
                        if j in seen_cols:
                            raise VerificationError(f"{name}: duplicate negative column in row {i}")
                        seen_cols.add(j)
                        rows[i][j] = -1
                return rows
        if "bit_rows" in data:
            bit_rows = _as_nonstring_list(
                data["bit_rows"], name=f"{name}.bit_rows"
            )
            rows = [_parse_pm1_sequence(row, name=f"{name}.bit_rows[{i}]") for i, row in enumerate(bit_rows)]
            if not rows or any(len(row) != len(rows[0]) for row in rows):
                raise VerificationError(f"{name}.bit_rows must form a nonempty rectangular matrix")
            embedded_n_raw = _mapping_get_first(data, "n", "order", "size")
            embedded_n = (
                None
                if embedded_n_raw is None
                else _to_int(embedded_n_raw, name=f"{name}.n")
            )
            if n is not None and embedded_n is not None and n != embedded_n:
                raise VerificationError(
                    f"{name}: supplied order conflicts with the required order"
                )
            target_n = n if n is not None else embedded_n
            if target_n is not None and (
                len(rows) != target_n
                or any(len(row) != target_n for row in rows)
            ):
                raise VerificationError(f"{name} must be {target_n}x{target_n}")
            return rows
        raise VerificationError(f"unsupported {name} dict format")

    raw_rows = _as_nonstring_list(data, name=name)
    rows = [
        _parse_pm1_sequence(row, name=f"{name}[{i}]")
        for i, row in enumerate(raw_rows)
    ]
    if n is not None and len(rows) != n:
        raise VerificationError(f"{name} must have {n} rows")
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise VerificationError(f"{name} must be a nonempty rectangular matrix")
    if n is not None and any(len(row) != n for row in rows):
        raise VerificationError(f"{name} must be {n}x{n}")
    return rows


def _check_square_shape(rows: Sequence[Sequence[Any]], n: int, *, name: str) -> None:
    if len(rows) != n or any(len(row) != n for row in rows):
        raise VerificationError(f"{name} must be {n}x{n}")


def _dot(row1: Sequence[int], row2: Sequence[int]) -> int:
    return sum(a * b for a, b in zip(row1, row2))


def _check_gram_equals_scalar_identity(rows: Sequence[Sequence[int]], scalar: int, *, name: str) -> None:
    n = len(rows)
    for i in range(n):
        d = _dot(rows[i], rows[i])
        if d != scalar:
            raise VerificationError(f"{name}: row {i} has norm-squared {d}, expected {scalar}")
        for j in range(i + 1, n):
            val = _dot(rows[i], rows[j])
            if val != 0:
                raise VerificationError(f"{name}: rows {i} and {j} have dot product {val}, expected 0")


# ---------------------------------------------------------------------------
# Binary operation tables: magmas and groups
# ---------------------------------------------------------------------------


def _parse_binary_operation_table_with_elements(data: Any, *, name: str = "operation table") -> tuple[list[list[int]], list[Any]]:
    """Parse a finite binary operation table and return (table, normalized labels)."""
    elements = None
    payload = data
    if isinstance(data, Mapping):
        for key in ("table", "operation_table", "cayley_table", "multiplication_table"):
            if key in data:
                payload = data[key]
                elements = _element_labels_or_none(data.get("elements", data.get("labels")), name=name)
                break
        else:
            for key in ("operation", "op", "products"):
                if key in data:
                    payload = data[key]
                    elements = _element_labels_or_none(data.get("elements", data.get("labels")), name=name)
                    break
            else:
                # Bare pair-key dictionary.
                if data and all(_is_nonstring_sequence(k) and len(k) == 2 for k in data.keys()):
                    payload = data
                else:
                    raise VerificationError(f"unsupported {name} dict format")

    if isinstance(payload, Mapping):
        pairs = list(payload.items())
        normalized_pairs = []
        if elements is None:
            seen: list[Any] = []
            for key, value in pairs:
                if not _is_nonstring_sequence(key) or len(key) != 2:
                    raise VerificationError(f"{name}: operation keys must be pairs")
                a = _hashable_key(key[0])
                b = _hashable_key(key[1])
                c = _hashable_key(value)
                try:
                    hash(a)
                    hash(b)
                    hash(c)
                except TypeError as exc:
                    raise VerificationError(
                        f"{name}: element labels must be hashable"
                    ) from exc
                normalized_pairs.append(((a, b), c))
                for z in (a, b, c):
                    if z not in seen:
                        seen.append(z)
            elements = seen
        else:
            normalized_pairs = []
            for key, value in pairs:
                if not _is_nonstring_sequence(key) or len(key) != 2:
                    raise VerificationError(f"{name}: operation keys must be pairs")
                a = _hashable_key(key[0])
                b = _hashable_key(key[1])
                c = _hashable_key(value)
                try:
                    hash(a)
                    hash(b)
                    hash(c)
                except TypeError as exc:
                    raise VerificationError(
                        f"{name}: element labels must be hashable"
                    ) from exc
                normalized_pairs.append(((a, b), c))
        if len(set(elements)) != len(elements):
            raise VerificationError(f"{name}: element labels must be distinct")
        index = {e: i for i, e in enumerate(elements)}
        n = len(elements)
        table: list[list[int | None]] = [[None] * n for _ in range(n)]
        for (a, b), value in normalized_pairs:
            if a not in index or b not in index or value not in index:
                raise VerificationError(f"{name}: operation uses an element outside the element list")
            if table[index[a]][index[b]] is not None:
                raise VerificationError(f"{name}: duplicate operation value for a pair")
            table[index[a]][index[b]] = index[value]
        if any(cell is None for row in table for cell in row):
            raise VerificationError(f"{name}: operation dictionary is incomplete")
        return [[int(cell) for cell in row] for row in table], list(elements)  # type: ignore[arg-type]

    rows = _as_nonstring_list(payload, name=name)
    if not rows:
        raise VerificationError(f"{name} must be a nonempty square table")
    normalized_rows = []
    for i, row in enumerate(rows):
        normalized_rows.append(
            _as_nonstring_list(row, name=f"{name} row {i}")
        )
    rows = normalized_rows
    n = len(rows)
    if any(len(row) != n for row in rows):
        raise VerificationError(f"{name} must be square")
    if elements is not None:
        if len(elements) != n or len(set(elements)) != n:
            raise VerificationError(f"{name}: element labels must be distinct and match table size")
        index = {e: i for i, e in enumerate(elements)}
        table = []
        for i, row in enumerate(rows):
            parsed_row = []
            for j, value in enumerate(row):
                key = _hashable_key(value)
                try:
                    hash(key)
                except TypeError as exc:
                    raise VerificationError(
                        f"{name}[{i}][{j}] is not a hashable element label"
                    ) from exc
                if key not in index:
                    raise VerificationError(f"{name}[{i}][{j}] is not in the element list")
                parsed_row.append(index[key])
            table.append(parsed_row)
        return table, list(elements)
    table = [[_to_int(value, name=f"{name}[{i}][{j}]") for j, value in enumerate(row)] for i, row in enumerate(rows)]
    if any(not 0 <= value < n for row in table for value in row):
        raise VerificationError(f"{name} entries must be integers in 0..n-1 when no labels are supplied")
    return table, list(range(n))


def _parse_binary_operation_table(data: Any, *, name: str = "operation table") -> list[list[int]]:
    """Parse a finite binary operation table with entries normalized to 0..n-1.

    Accepted forms:
    - raw n x n table using integer entries 0..n-1;
    - {"elements": [...], "table": [[... labels ...], ...]};
    - {"operation": {(a,b): c, ...}, "elements": [...]} or a bare pair-key dict.
    """
    table, _ = _parse_binary_operation_table_with_elements(data, name=name)
    return table


def _verify_group_table(table: Sequence[Sequence[int]]) -> tuple[int, list[int]]:
    """Verify a finite group table. Return (identity, inverses)."""
    n = len(table)
    if n == 0 or any(len(row) != n for row in table):
        raise VerificationError("group operation table must be nonempty and square")
    if any(not 0 <= x < n for row in table for x in row):
        raise VerificationError("group operation table is not closed")

    identity = None
    for e in range(n):
        if all(table[e][x] == x and table[x][e] == x for x in range(n)):
            identity = e
            break
    if identity is None:
        raise VerificationError("group table has no two-sided identity")

    inverses = [-1] * n
    for x in range(n):
        for y in range(n):
            if table[x][y] == identity and table[y][x] == identity:
                inverses[x] = y
                break
        if inverses[x] < 0:
            raise VerificationError("some group element has no two-sided inverse")

    for x in range(n):
        tx = table[x]
        for y in range(n):
            xy = tx[y]
            ty = table[y]
            for z in range(n):
                if table[xy][z] != tx[ty[z]]:
                    raise VerificationError("group operation is not associative")
    return identity, inverses


# ---------------------------------------------------------------------------
# 1. Turyn-type sequences TT(46)
# ---------------------------------------------------------------------------


def _nonperiodic_autocorrelation(seq: Sequence[int], shift: int) -> int:
    if shift < 0:
        raise VerificationError("autocorrelation shift must be nonnegative")
    if shift >= len(seq):
        return 0
    return sum(seq[i] * seq[i + shift] for i in range(len(seq) - shift))


def _parse_turyn_payload(candidate: Any, *, n: int) -> tuple[list[int], list[int], list[int], list[int]]:
    if isinstance(candidate, Mapping):
        if "sequences" in candidate:
            return _parse_turyn_payload(candidate["sequences"], n=n)
        keys = None
        for possible in (("A", "B", "C", "D"), ("X", "Y", "Z", "W"), ("a", "b", "c", "d"), ("x", "y", "z", "w")):
            if all(key in candidate for key in possible):
                keys = possible
                break
        if keys is None:
            raise VerificationError("Turyn witness dict must contain A,B,C,D or X,Y,Z,W")
        A = _parse_pm1_sequence(candidate[keys[0]], length=n, name=keys[0])
        B = _parse_pm1_sequence(candidate[keys[1]], length=n, name=keys[1])
        C = _parse_pm1_sequence(candidate[keys[2]], length=n, name=keys[2])
        D = _parse_pm1_sequence(candidate[keys[3]], length=n - 1, name=keys[3])
        return A, B, C, D
    sequences = _as_nonstring_list(candidate, name="Turyn witness")
    if len(sequences) == 4:
        A = _parse_pm1_sequence(sequences[0], length=n, name="A")
        B = _parse_pm1_sequence(sequences[1], length=n, name="B")
        C = _parse_pm1_sequence(sequences[2], length=n, name="C")
        D = _parse_pm1_sequence(sequences[3], length=n - 1, name="D")
        return A, B, C, D
    raise VerificationError("Turyn witness must be four ±1 sequences")


def verify_turyn_type_tt46(candidate: Any) -> ValidationResult:
    """Verify a Turyn-type sequence TT(46).

    Accepted formats:
    - [A, B, C, D]
    - {"A": A, "B": B, "C": C, "D": D} or {"X": X, ...}
    - each sequence may be a ±1 list, '+-...' string, bitstring, or minus-position dict.
    """
    n = 46
    A, B, C, D = _parse_turyn_payload(candidate, n=n)
    for s in range(1, n):
        val = (
            _nonperiodic_autocorrelation(A, s)
            + _nonperiodic_autocorrelation(B, s)
            + 2 * _nonperiodic_autocorrelation(C, s)
            + 2 * _nonperiodic_autocorrelation(D, s)
        )
        if val != 0:
            return ValidationResult(False, "Turyn-type TT(46)", f"autocorrelation condition fails at shift {s}")
    return ValidationResult(True, "Turyn-type TT(46)", "valid counterexample witness", {"n": n})


# ---------------------------------------------------------------------------
# 2. Skew-Hadamard matrix of order 356
# ---------------------------------------------------------------------------


def _parse_strict_upper_triangle_pm1(data: Any, *, n: int, name: str) -> list[list[int]]:
    """Parse the strict upper triangle of a skew-Hadamard matrix as ±1 signs."""
    payload = data
    if isinstance(data, Mapping):
        for key in ("strict_upper_triangle", "upper_triangle", "upper_triangle_signs", "signs", "sequence", "entries", "data", "bitstring"):
            if key in data:
                payload = data[key]
                break
    signs = _parse_pm1_sequence(payload, length=n * (n - 1) // 2, name=name)
    H = [[1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = signs[idx]
            idx += 1
            H[i][j] = s
            H[j][i] = -s
    return H


def _parse_tournament_adjacency(data: Any, *, n: int, name: str) -> list[list[int]]:
    """Parse an oriented complete graph as a 0/1 adjacency matrix."""
    try:
        D = _parse_oriented_graph(data)
    except VerificationError as exc:
        raise VerificationError(f"{name}: {exc}") from exc
    if D.number_of_nodes() != n:
        raise VerificationError(f"{name} must have order {n}")
    rows = [[0] * n for _ in range(n)]
    for u, v in D.edges():
        rows[u][v] = 1

    for i in range(n):
        for j in range(i + 1, n):
            if rows[i][j] + rows[j][i] != 1:
                raise VerificationError(f"{name}: every unordered pair must have exactly one orientation")
    return rows


def _check_doubly_regular_tournament(rows: Sequence[Sequence[int]]) -> None:
    v = len(rows)
    if v % 4 != 3:
        raise VerificationError("a doubly regular tournament must have order congruent to 3 mod 4")
    outdegree = (v - 1) // 2
    common = (v - 3) // 4
    masks = []
    for i, row in enumerate(rows):
        mask = 0
        for j, value in enumerate(row):
            if value not in (0, 1):
                raise VerificationError("tournament adjacency entries must be 0/1")
            if value:
                mask |= 1 << j
        if mask & (1 << i):
            raise VerificationError("tournament adjacency matrix must have zero diagonal")
        if mask.bit_count() != outdegree:
            raise VerificationError(f"tournament vertex {i} has outdegree {mask.bit_count()}, expected {outdegree}")
        masks.append(mask)
    for i in range(v):
        for j in range(i + 1, v):
            if (masks[i] >> j) & 1 == (masks[j] >> i) & 1:
                raise VerificationError("input is not a tournament orientation")
            c = (masks[i] & masks[j]).bit_count()
            if c != common:
                raise VerificationError(f"vertices {i},{j} have {c} common out-neighbors, expected {common}")


def _skew_hadamard_from_core(core: Sequence[Sequence[int]]) -> list[list[int]]:
    m = len(core)
    H = [[1] * (m + 1)]
    for row in core:
        H.append([-1] + list(row))
    return H


def _skew_hadamard_from_tournament_adjacency(rows: Sequence[Sequence[int]]) -> list[list[int]]:
    # Core convention: C_ij = +1 when i -> j and -1 otherwise, with C_ii = +1.
    core = []
    for i, row in enumerate(rows):
        core_row = []
        for j, value in enumerate(row):
            if i == j:
                core_row.append(1)
            else:
                core_row.append(1 if value == 1 else -1)
        core.append(core_row)
    return _skew_hadamard_from_core(core)


def _verify_skew_hadamard_rows(H: Sequence[Sequence[int]], *, n: int, source: str) -> ValidationResult:
    _check_square_shape(H, n, name="skew Hadamard matrix")
    if any(value not in (-1, 1) for row in H for value in row):
        raise VerificationError("skew Hadamard matrix entries must be ±1")
    for i in range(n):
        if H[i][i] != 1:
            return ValidationResult(False, "Skew-Hadamard order 356", "diagonal entries must all be +1")
        for j in range(i + 1, n):
            if H[i][j] + H[j][i] != 0:
                return ValidationResult(False, "Skew-Hadamard order 356", "H-I is not skew-symmetric")
    try:
        _check_gram_equals_scalar_identity(H, n, name="skew Hadamard matrix")
    except VerificationError as exc:
        return ValidationResult(False, "Skew-Hadamard order 356", str(exc))
    return ValidationResult(True, "Skew-Hadamard order 356", "valid counterexample witness", {"order": n, "source": source})


def verify_skew_hadamard_356(candidate: Any) -> ValidationResult:
    """Verify a skew-Hadamard matrix of order 356.

    Accepted encodings include raw ±1 rows, {"matrix": ...}, {"H": ...},
    negative-position encodings, row-negative-position encodings, and compact
    proof-carrying forms:
    - {"strict_upper_triangle": signs} for the strict upper triangle of H;
    - {"normalized_core": C} or {"skew_core": C} for the 355x355 normalized core;
    - {"doubly_regular_tournament": A} (or construction/type="doubly_regular_tournament")
      where A is a 355-vertex tournament adjacency matrix or directed edge list.
    """
    n = 356
    if isinstance(candidate, Mapping):
        kind = str(candidate.get("construction", candidate.get("type", ""))).strip().lower().replace("-", "_").replace(" ", "_")
        if kind in {"strict_upper_triangle", "upper_triangle", "upper_triangle_signs"}:
            H = _parse_strict_upper_triangle_pm1(candidate, n=n, name="skew Hadamard strict upper triangle")
            return _verify_skew_hadamard_rows(H, n=n, source="strict_upper_triangle")
        if kind in {"normalized_core", "skew_core", "core"}:
            core_payload = candidate.get("core", candidate.get("normalized_core", candidate.get("skew_core", candidate.get("matrix"))))
            C = _parse_pm1_matrix(core_payload, n=n - 1, name="skew Hadamard normalized core")
            return _verify_skew_hadamard_rows(_skew_hadamard_from_core(C), n=n, source="normalized_core")
        if kind in {"doubly_regular_tournament", "drt", "tournament"}:
            A = _parse_tournament_adjacency(candidate, n=n - 1, name="doubly regular tournament")
            try:
                _check_doubly_regular_tournament(A)
            except VerificationError as exc:
                return ValidationResult(False, "Skew-Hadamard order 356", str(exc))
            return _verify_skew_hadamard_rows(_skew_hadamard_from_tournament_adjacency(A), n=n, source="doubly_regular_tournament")
        for key in ("strict_upper_triangle", "upper_triangle", "upper_triangle_signs"):
            if key in candidate:
                H = _parse_strict_upper_triangle_pm1(candidate[key], n=n, name="skew Hadamard strict upper triangle")
                return _verify_skew_hadamard_rows(H, n=n, source="strict_upper_triangle")
        for key in ("normalized_core", "skew_core", "core"):
            if key in candidate:
                C = _parse_pm1_matrix(candidate[key], n=n - 1, name="skew Hadamard normalized core")
                return _verify_skew_hadamard_rows(_skew_hadamard_from_core(C), n=n, source="normalized_core")
        for key in ("doubly_regular_tournament", "drt", "tournament", "tournament_matrix"):
            if key in candidate:
                A = _parse_tournament_adjacency(candidate[key], n=n - 1, name="doubly regular tournament")
                try:
                    _check_doubly_regular_tournament(A)
                except VerificationError as exc:
                    return ValidationResult(False, "Skew-Hadamard order 356", str(exc))
                return _verify_skew_hadamard_rows(_skew_hadamard_from_tournament_adjacency(A), n=n, source="doubly_regular_tournament")
    H = _parse_pm1_matrix(candidate, n=n, name="skew Hadamard matrix")
    return _verify_skew_hadamard_rows(H, n=n, source="matrix")


# ---------------------------------------------------------------------------
# 3. Cocyclic Hadamard matrix of order 188
# ---------------------------------------------------------------------------


def _parse_sign_matrix_or_pair_dict(
    data: Any,
    *,
    n: int,
    name: str,
    labels: Sequence[Any] | None = None,
) -> list[list[int]]:
    """Parse a ±1 matrix, optionally from a complete pair dictionary using labels."""
    label_to_index: dict[Any, int] | None = None
    if labels is not None:
        normalized_labels = [_hashable_key(label) for label in labels]
        try:
            distinct_labels = set(normalized_labels)
        except TypeError as exc:
            raise VerificationError(f"{name}: labels must be hashable") from exc
        if len(normalized_labels) != n or len(distinct_labels) != n:
            raise VerificationError(f"{name}: labels must be distinct and have length {n}")
        label_to_index = {label: i for i, label in enumerate(normalized_labels)}

    def parse_pair_index(value: Any, *, role: str) -> int:
        key = _hashable_key(value)
        try:
            hash(key)
        except TypeError as exc:
            raise VerificationError(f"{name}: pair labels must be hashable") from exc
        if label_to_index is not None and key in label_to_index:
            return label_to_index[key]
        idx = _to_int(value, name=f"{name} {role} index")
        if not 0 <= idx < n:
            raise VerificationError(f"{name}: pair index out of range")
        return idx

    matrix_wrapper_keys = {
        "matrix", "H", "M", "array", "negative_positions", "minus_positions",
        "row_negative_positions", "rows_negative_positions", "negative_positions_by_row", "bit_rows",
    }
    if isinstance(data, Mapping) and not any(key in data for key in matrix_wrapper_keys):
        rows: list[list[int | None]] = [[None] * n for _ in range(n)]
        if "entries" in data or "values" in data:
            entries = _as_nonstring_list(
                _mapping_get_first(data, "entries", "values"),
                name=f"{name} entries/values",
            )
            for entry_raw in entries:
                entry = _as_nonstring_list(
                    entry_raw, name=f"{name} entry"
                )
                if len(entry) == 2:
                    pair = _as_nonstring_list(
                        entry[0], name=f"{name} entry pair"
                    )
                    if len(pair) != 2:
                        raise VerificationError(
                            f"{name}: each entry must be ((row,col), sign) or (row,col,sign)"
                        )
                    value = entry[1]
                    a_raw, b_raw = pair
                elif len(entry) == 3:
                    a_raw, b_raw, value = entry
                else:
                    raise VerificationError(f"{name}: each entry must be ((row,col), sign) or (row,col,sign)")
                i = parse_pair_index(a_raw, role="row")
                j = parse_pair_index(b_raw, role="col")
                if rows[i][j] is not None:
                    raise VerificationError(f"{name}: duplicate entry for a pair")
                rows[i][j] = _parse_sign(value, name=f"{name}[{i}][{j}]")
        else:
            for key, value in data.items():
                if not _is_nonstring_sequence(key) or len(key) != 2:
                    raise VerificationError(f"{name}: dict keys must be pairs of indices or labels")
                i = parse_pair_index(key[0], role="row")
                j = parse_pair_index(key[1], role="col")
                if rows[i][j] is not None:
                    raise VerificationError(f"{name}: duplicate entry for a pair")
                rows[i][j] = _parse_sign(value, name=f"{name}[{i}][{j}]")
        if any(cell is None for row in rows for cell in row):
            raise VerificationError(f"{name}: pair dictionary is incomplete")
        return [[int(cell) for cell in row] for row in rows]  # type: ignore[arg-type]
    return _parse_pm1_matrix(data, n=n, name=name)


def verify_cocyclic_hadamard_188(candidate: Any) -> ValidationResult:
    """Verify a cocyclic Hadamard matrix of order 188.

    A raw Hadamard matrix alone does not prove cocyclicity. The accepted proof-carrying
    formats are:
    - {"group_table": table, "cocycle": psi}
    - {"operation_table": table, "cocycle_matrix": psi}
    - {"group_table": table, "derived_matrix": H} where H itself is used as psi.

    The table must define a group of order 188; psi must be a ±1-valued 2-cocycle, and
    the matrix (psi(g,h)) must be Hadamard.
    """
    if not isinstance(candidate, Mapping):
        raise VerificationError("cocyclic Hadamard witnesses must include a group table and a cocycle")
    group_payload = None
    for key in ("group_table", "operation_table", "cayley_table", "multiplication_table", "group"):
        if key in candidate:
            group_payload = candidate[key]
            break
    if group_payload is None:
        raise VerificationError("cocyclic witness needs a finite group operation table")
    outer_labels = candidate.get("elements", candidate.get("labels"))
    group_data = {"table": group_payload, "elements": outer_labels} if outer_labels is not None else group_payload
    table, elements = _parse_binary_operation_table_with_elements(group_data, name="group table")
    n = len(table)
    if n != 188:
        raise VerificationError("cocyclic Hadamard group must have order 188")
    _verify_group_table(table)

    cocycle_payload = None
    for key in ("cocycle", "psi", "cocycle_matrix", "derived_matrix", "matrix", "H"):
        if key in candidate:
            cocycle_payload = candidate[key]
            break
    if cocycle_payload is None:
        raise VerificationError("cocyclic witness needs a ±1-valued cocycle matrix")
    psi = _parse_sign_matrix_or_pair_dict(cocycle_payload, n=n, name="cocycle", labels=elements)

    for x in range(n):
        for y in range(n):
            xy = table[x][y]
            for z in range(n):
                yz = table[y][z]
                if psi[x][y] * psi[xy][z] != psi[x][yz] * psi[y][z]:
                    return ValidationResult(False, "Cocyclic Hadamard order 188", "2-cocycle identity fails")

    try:
        _check_gram_equals_scalar_identity(psi, n, name="cocyclic matrix")
    except VerificationError as exc:
        return ValidationResult(False, "Cocyclic Hadamard order 188", str(exc))
    return ValidationResult(True, "Cocyclic Hadamard order 188", "valid counterexample witness", {"order": n})


# ---------------------------------------------------------------------------
# 4 and 7. Finite magma implications E677/E255 and their duals
# ---------------------------------------------------------------------------


def _magma_op(table: Sequence[Sequence[int]], x: int, y: int) -> int:
    return table[x][y]


def _magma_satisfies_e677(table: Sequence[Sequence[int]]) -> bool:
    n = len(table)
    op = _magma_op
    for x in range(n):
        for y in range(n):
            rhs = op(table, y, op(table, x, op(table, op(table, y, x), y)))
            if rhs != x:
                return False
    return True


def _magma_violates_e255(table: Sequence[Sequence[int]]) -> bool:
    n = len(table)
    op = _magma_op
    for x in range(n):
        rhs = op(table, op(table, op(table, x, x), x), x)
        if rhs != x:
            return True
    return False


def _magma_satisfies_dual_e677(table: Sequence[Sequence[int]]) -> bool:
    n = len(table)
    op = _magma_op
    for x in range(n):
        for y in range(n):
            # dual of y*(x*((y*x)*y)) is (((y*(x*y))*x)*y)
            rhs = op(table, op(table, op(table, y, op(table, x, y)), x), y)
            if rhs != x:
                return False
    return True


def _magma_violates_dual_e255(table: Sequence[Sequence[int]]) -> bool:
    n = len(table)
    op = _magma_op
    for x in range(n):
        # dual of (((x*x)*x)*x) is x*(x*(x*x))
        rhs = op(table, x, op(table, x, op(table, x, x)))
        if rhs != x:
            return True
    return False


def verify_finite_magma_e677_not_e255(candidate: Any) -> ValidationResult:
    """Verify a finite magma satisfying E677 and violating E255.

    E677: x = y * (x * ((y * x) * y)).
    E255: x = (((x * x) * x) * x).
    """
    table = _parse_binary_operation_table(candidate, name="finite magma operation table")
    if not _magma_satisfies_e677(table):
        return ValidationResult(False, "Finite magma E677 not E255", "E677 does not hold for all x,y")
    if not _magma_violates_e255(table):
        return ValidationResult(False, "Finite magma E677 not E255", "E255 also holds, so this is not a counterexample")
    return ValidationResult(True, "Finite magma E677 not E255", "valid counterexample witness", {"order": len(table)})


def verify_dual_finite_magma(candidate: Any) -> ValidationResult:
    """Verify a finite magma satisfying dual(E677) and violating dual(E255)."""
    table = _parse_binary_operation_table(candidate, name="finite magma operation table")
    if not _magma_satisfies_dual_e677(table):
        return ValidationResult(False, "Dual finite magma", "dual(E677) does not hold for all x,y")
    if not _magma_violates_dual_e255(table):
        return ValidationResult(False, "Dual finite magma", "dual(E255) also holds, so this is not a counterexample")
    return ValidationResult(True, "Dual finite magma", "valid counterexample witness", {"order": len(table)})


# ---------------------------------------------------------------------------
# 5. Line-graph inertia
# ---------------------------------------------------------------------------


def verify_line_graph_inertia(candidate: Any) -> ValidationResult:
    """Verify n_+(A(L(G))) > n_-(A(L(G))) + 1 for a connected simple graph G."""
    G = _parse_graph(candidate)
    if G.number_of_nodes() == 0:
        raise VerificationError("graph must be nonempty")
    if not nx.is_connected(G):
        raise VerificationError("graph must be connected")
    if G.number_of_edges() == 0:
        raise VerificationError("line graph has no vertices; supply a connected graph with at least one edge")
    L = nx.convert_node_labels_to_integers(nx.line_graph(G), ordering="sorted")
    rows = _adjacency_rows(L)
    p_pos, p_neg, p_zero = _inertia_from_integer_symmetric_rows(rows)
    ok = p_pos > p_neg + 1
    return ValidationResult(
        ok,
        "Line-graph inertia",
        "valid counterexample witness" if ok else "line graph inertia inequality is not violated",
        {"vertices": G.number_of_nodes(), "edges": G.number_of_edges(), "p_plus": p_pos, "p_minus": p_neg, "p_zero": p_zero},
    )


# ---------------------------------------------------------------------------
# 6. Lovasz deletion counterexample checker, r=5 and r=6
# Diagnostic only: intentionally omitted from VALIDATORS and __all__.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedHypergraph:
    r: int
    num_vertices: int
    parts: tuple[int, ...]
    edges: tuple[frozenset[int], ...]
    original_vertices: tuple[Any, ...]


def _normalize_vertex_key(vertex: Any) -> Any:
    key = _hashable_key(vertex)
    try:
        hash(key)
    except TypeError as exc:
        raise VerificationError("hypergraph vertex labels must be hashable") from exc
    return key


def _parse_hypergraph(candidate: Any, *, r: int) -> tuple[ParsedHypergraph, Mapping[str, Any] | None]:
    metadata = candidate if isinstance(candidate, Mapping) else None
    if isinstance(candidate, Mapping):
        edges_payload = candidate.get("edges", candidate.get("blocks", candidate.get("hyperedges")))
        if edges_payload is None:
            raise VerificationError("hypergraph witness must contain edges/hyperedges")
        parts_payload = candidate.get("parts", candidate.get("vertex_parts"))
        part_of_vertex_payload = candidate.get("part_of_vertex", candidate.get("vertex_to_part"))
    else:
        edges_payload = candidate
        parts_payload = None
        part_of_vertex_payload = None

    edge_values = _as_nonstring_list(edges_payload, name="hypergraph edges")
    raw_edges = [
        _as_nonstring_list(edge, name=f"hypergraph edge {i}")
        for i, edge in enumerate(edge_values)
    ]
    if not raw_edges:
        raise VerificationError("hypergraph must have at least one edge")
    if any(len(edge) != r for edge in raw_edges):
        raise VerificationError(f"hypergraph must be {r}-uniform")

    vertex_to_part_raw: dict[Any, int] = {}
    if parts_payload is not None:
        part_values = _as_nonstring_list(parts_payload, name="hypergraph parts")
        if len(part_values) != r:
            raise VerificationError(f"parts must be a length-{r} sequence")
        part_label_lists: list[list[Any]] = []
        for part_idx, part in enumerate(part_values):
            labels = [
                _normalize_vertex_key(vertex)
                for vertex in _as_nonstring_list(
                    part, name=f"hypergraph part {part_idx}"
                )
            ]
            if len(set(labels)) != len(labels):
                raise VerificationError("a part contains a repeated vertex label")
            part_label_lists.append(labels)
        flat_labels = [label for labels in part_label_lists for label in labels]
        use_part_local_labels = len(set(flat_labels)) != len(flat_labels)
        for part_idx, labels in enumerate(part_label_lists):
            for label in labels:
                key = (part_idx, label) if use_part_local_labels else label
                vertex_to_part_raw[key] = part_idx
    elif part_of_vertex_payload is not None:
        if not isinstance(part_of_vertex_payload, Mapping):
            raise VerificationError("part_of_vertex must be a mapping")
        for vertex, part_raw in part_of_vertex_payload.items():
            part = _to_int(part_raw, name="vertex part")
            if not 0 <= part < r:
                raise VerificationError("vertex part out of range")
            vertex_to_part_raw[_normalize_vertex_key(vertex)] = part
    else:
        # Position-coded r-partite format: edge[j] is a vertex in part j.  Use (j,label)
        # as the actual vertex key so labels can safely repeat across parts.
        vertex_to_part_raw = {}
        converted_edges = []
        for edge in raw_edges:
            converted = []
            for part_idx, vertex in enumerate(edge):
                key = (part_idx, _normalize_vertex_key(vertex))
                vertex_to_part_raw.setdefault(key, part_idx)
                converted.append(key)
            converted_edges.append(converted)
        raw_edges = converted_edges

    vertex_order: list[Any] = []
    vertex_index: dict[Any, int] = {}
    parts: list[int] = []

    def add_vertex(key: Any, part: int) -> int:
        if key not in vertex_index:
            vertex_index[key] = len(vertex_order)
            vertex_order.append(key)
            parts.append(part)
        else:
            if parts[vertex_index[key]] != part:
                raise VerificationError("a vertex is assigned inconsistent parts")
        return vertex_index[key]

    normalized_edges: list[frozenset[int]] = []
    for raw_edge in raw_edges:
        edge_indices: list[int] = []
        seen_parts: set[int] = set()
        seen_vertices: set[int] = set()
        for pos, vertex in enumerate(raw_edge):
            key = _normalize_vertex_key(vertex)
            if key not in vertex_to_part_raw:
                positional_key = (pos, key)
                if positional_key in vertex_to_part_raw:
                    key = positional_key
                # If vertices are explicitly encoded as (part,label), accept that too.
                elif _is_nonstring_sequence(vertex) and len(vertex) == 2:
                    part = _to_int(vertex[0], name="vertex part")
                    key = (part, _normalize_vertex_key(vertex[1]))
                else:
                    raise VerificationError("edge uses a vertex whose part is unknown")
            part = vertex_to_part_raw.get(key, None)
            if part is None:
                part = _to_int(vertex[0], name="vertex part")  # type: ignore[index]
            if not 0 <= part < r:
                raise VerificationError("vertex part out of range")
            idx = add_vertex(key, part)
            if idx in seen_vertices:
                raise VerificationError("an edge repeats a vertex")
            if part in seen_parts:
                raise VerificationError("an edge has two vertices from the same part")
            seen_vertices.add(idx)
            seen_parts.add(part)
            edge_indices.append(idx)
        if seen_parts != set(range(r)):
            raise VerificationError("each edge must contain exactly one vertex from each part")
        normalized_edges.append(frozenset(edge_indices))

    edge_set = set(normalized_edges)
    if len(edge_set) != len(normalized_edges):
        raise VerificationError("duplicate hyperedges are not allowed")
    return ParsedHypergraph(r, len(vertex_order), tuple(parts), tuple(normalized_edges), tuple(vertex_order)), metadata


def _edge_masks(edges: Iterable[frozenset[int]]) -> tuple[int, ...]:
    masks = []
    for edge in edges:
        mask = 0
        for v in edge:
            mask |= 1 << v
        masks.append(mask)
    return tuple(sorted(set(masks)))


def _max_matching_size_from_masks(edge_masks: tuple[int, ...]) -> int:
    edge_masks = tuple(sorted(set(edge_masks), key=lambda m: (m.bit_count(), m)))

    @functools.lru_cache(maxsize=None)
    def rec(remaining: tuple[int, ...]) -> int:
        if not remaining:
            return 0
        if len(remaining) == 1:
            return 1
        first = remaining[0]
        without = rec(remaining[1:])
        compatible = tuple(mask for mask in remaining[1:] if mask & first == 0)
        with_first = 1 + rec(compatible)
        return max(without, with_first)

    return rec(edge_masks)


def _has_matching_of_size(edge_masks: tuple[int, ...], k: int) -> bool:
    if k <= 0:
        return True
    edge_masks = tuple(sorted(set(edge_masks), key=lambda m: m.bit_count()))

    @functools.lru_cache(maxsize=None)
    def rec(start: int, used: int, need: int) -> bool:
        if need == 0:
            return True
        if len(edge_masks) - start < need:
            return False
        for idx in range(start, len(edge_masks)):
            mask = edge_masks[idx]
            if mask & used == 0 and rec(idx + 1, used | mask, need - 1):
                return True
        return False

    return rec(0, 0, k)


def _parse_hyperedge_reference(edge_ref: Any, H: ParsedHypergraph) -> frozenset[int]:
    if isinstance(edge_ref, int) and not isinstance(edge_ref, bool):
        if not 0 <= edge_ref < len(H.edges):
            raise VerificationError("hyperedge index out of range")
        return H.edges[edge_ref]
    edge_vertices = _as_nonstring_list(
        edge_ref, name="hyperedge reference"
    )
    vertex_index = {v: i for i, v in enumerate(H.original_vertices)}
    out = []
    for pos, vertex in enumerate(edge_vertices):
        key = _normalize_vertex_key(vertex)
        if key in vertex_index:
            out.append(vertex_index[key])
        elif (pos, key) in vertex_index:
            # Natural edge-tuple format for position-coded r-partite hypergraphs.
            out.append(vertex_index[(pos, key)])
        elif _is_nonstring_sequence(vertex) and tuple(vertex) in vertex_index:
            out.append(vertex_index[tuple(vertex)])
        else:
            # Also allow already-normalized integer vertices.
            idx = _to_int(vertex, name="hyperedge vertex")
            if not 0 <= idx < H.num_vertices:
                raise VerificationError("hyperedge vertex index out of range")
            out.append(idx)
    edge = frozenset(out)
    if edge not in set(H.edges):
        raise VerificationError("referenced hyperedge is not in the hypergraph")
    return edge


def _verify_matching_certificate(matching_payload: Any, H: ParsedHypergraph, *, expected_size: int | None = None, forbidden_mask: int = 0) -> int:
    matching = _as_nonstring_list(
        matching_payload, name="matching certificate"
    )
    used = 0
    size = 0
    for edge_ref in matching:
        edge = _parse_hyperedge_reference(edge_ref, H)
        mask = next(mask for mask in _edge_masks([edge]))
        if mask & forbidden_mask:
            raise VerificationError("matching certificate uses a deleted/forbidden vertex")
        if used & mask:
            raise VerificationError("matching certificate edges are not disjoint")
        used |= mask
        size += 1
    if expected_size is not None and size != expected_size:
        raise VerificationError("matching certificate has the wrong size")
    return size


def _verify_vertex_cover_certificate(cover_payload: Any, H: ParsedHypergraph, *, expected_size: int | None = None) -> int:
    cover_vertices = _as_nonstring_list(
        cover_payload, name="vertex cover certificate"
    )
    vertex_index = {v: i for i, v in enumerate(H.original_vertices)}
    cover: set[int] = set()
    for vertex in cover_vertices:
        key = _normalize_vertex_key(vertex)
        if key in vertex_index:
            idx = vertex_index[key]
        else:
            idx = _to_int(vertex, name="cover vertex")
            if not 0 <= idx < H.num_vertices:
                raise VerificationError("cover vertex index out of range")
        if idx in cover:
            raise VerificationError("vertex cover certificate contains a duplicate")
        cover.add(idx)
    if expected_size is not None and len(cover) != expected_size:
        raise VerificationError("vertex cover certificate has the wrong size")
    for edge in H.edges:
        if not (set(edge) & cover):
            raise VerificationError("vertex cover certificate misses an edge")
    return len(cover)


def _deleted_set_mask(raw_deleted: Any, H: ParsedHypergraph) -> int:
    deleted_vertices = _as_nonstring_list(
        raw_deleted, name="deleted vertex set"
    )
    vertex_index = {v: i for i, v in enumerate(H.original_vertices)}
    mask = 0
    count = 0
    for vertex in deleted_vertices:
        key = _normalize_vertex_key(vertex)
        if key in vertex_index:
            idx = vertex_index[key]
        else:
            idx = _to_int(vertex, name="deleted vertex")
            if not 0 <= idx < H.num_vertices:
                raise VerificationError("deleted vertex index out of range")
        if mask & (1 << idx):
            raise VerificationError("deleted vertex set contains a duplicate")
        mask |= 1 << idx
        count += 1
    if count != H.r - 1:
        raise VerificationError(f"deleted vertex set must have size {H.r - 1}")
    return mask


def verify_lovasz_deletion(candidate: Any, *, r: int) -> ValidationResult:
    """Verify a counterexample to Lovasz's deletion conjecture for r-partite r-uniform hypergraphs.

    The counterexample condition checked is: if k = nu(H), then for every set S of r-1
    vertices, nu(H - S) = k. Since deletion cannot increase matching number, it suffices
    to verify a k-matching remains after every such deletion.

    Bare hypergraphs are accepted and checked by exact branch-and-bound matching. Optional
    certificates can make validation much faster:
    - "matching": a k-matching in H;
    - "vertex_cover": a k-vertex cover in H, proving nu(H) <= k;
    - "deletion_matchings": list of {"deleted": [...], "matching": [...]} entries.

    This exact checker is retained for diagnostic and small-instance use. The r=5,6
    problems are not benchmark-selected: absent a new compact proof format, checking
    every one of the C(|V|, r-1) deletions (or listing a certificate for each of them)
    is impractical at realistic witness sizes.
    """
    if r not in (5, 6):
        raise VerificationError("this checker supports Lovasz deletion only at r=5 or r=6")
    H, meta = _parse_hypergraph(candidate, r=r)
    all_masks = _edge_masks(H.edges)

    k: int
    if meta is not None and "matching" in meta:
        k = _verify_matching_certificate(meta["matching"], H)
        if "vertex_cover" in meta:
            _verify_vertex_cover_certificate(
                meta["vertex_cover"], H, expected_size=k
            )
        else:
            # A valid matching certificate is useful as a lower bound, but if it is
            # not maximum we should not reject an otherwise valid bare hypergraph.
            k = _max_matching_size_from_masks(all_masks)
    else:
        k = _max_matching_size_from_masks(all_masks)
    if k <= 0:
        return ValidationResult(False, f"Lovasz deletion r={r}", "hypergraph has matching number 0")

    deletion_certs: dict[int, Any] = {}
    if meta is not None and "deletion_matchings" in meta:
        certs = _as_nonstring_list(
            meta["deletion_matchings"], name="deletion_matchings"
        )
        for item in certs:
            if not isinstance(item, Mapping) or "deleted" not in item or "matching" not in item:
                raise VerificationError("each deletion matching certificate must have deleted and matching fields")
            mask = _deleted_set_mask(item["deleted"], H)
            if mask in deletion_certs:
                raise VerificationError("duplicate deletion matching certificate")
            _verify_matching_certificate(item["matching"], H, expected_size=k, forbidden_mask=mask)
            deletion_certs[mask] = item["matching"]

    vertices = range(H.num_vertices)
    for deleted in itertools.combinations(vertices, r - 1):
        deleted_mask = 0
        for v in deleted:
            deleted_mask |= 1 << v
        if deleted_mask in deletion_certs:
            continue
        remaining_edges = tuple(mask for mask in all_masks if mask & deleted_mask == 0)
        if not _has_matching_of_size(remaining_edges, k):
            return ValidationResult(False, f"Lovasz deletion r={r}", "some deletion reduces the matching number")

    return ValidationResult(True, f"Lovasz deletion r={r}", "valid counterexample witness", {"r": r, "vertices": H.num_vertices, "edges": len(H.edges), "matching_number": k})


def verify_lovasz_deletion_r5(candidate: Any) -> ValidationResult:
    return verify_lovasz_deletion(candidate, r=5)


def verify_lovasz_deletion_r6(candidate: Any) -> ValidationResult:
    return verify_lovasz_deletion(candidate, r=6)


# ---------------------------------------------------------------------------
# 8. Max Laplacian eigenvalue upper bounds
# ---------------------------------------------------------------------------


def _local_degree_data(G: nx.Graph) -> tuple[list[int], list[Fraction]]:
    degrees = [G.degree(v) for v in range(G.number_of_nodes())]
    if any(d == 0 for d in degrees):
        raise VerificationError("all vertices must have positive degree for m_v to be defined")
    m_values = []
    for v in range(G.number_of_nodes()):
        m_values.append(Fraction(sum(degrees[u] for u in G.neighbors(v)), degrees[v]))
    return degrees, m_values


def _laplacian_bound_values(G: nx.Graph, conjecture_id: int) -> list[RadicalBound]:
    """
    Following arXiv:2606.14550, a term with a negative square-root argument is
    treated as negative infinity and therefore omitted from the maximum.
    """
    d, m = _local_degree_data(G)
    values: list[RadicalBound] = []
    if conjecture_id == 11:
        for v in range(G.number_of_nodes()):
            values.append(RadicalBound(2 * m[v] ** 3 / (d[v] ** 2)))
    elif conjecture_id == 40:
        for u, v in G.edges():
            rad = (
                2 * ((m[u] - 1) ** 2 + (m[v] - 1) ** 2)
                + d[u] ** 2
                + d[v] ** 2
                - d[u] * m[u]
                - d[v] * m[v]
            )
            if rad >= 0:
                values.append(RadicalBound(Fraction(2), rad))
    elif conjecture_id == 44:
        for u, v in G.edges():
            rad = 2 * (
                (d[u] - 1) ** 2
                + (d[v] - 1) ** 2
                + m[u] * m[v]
                - d[u] * d[v]
            )
            if rad >= 0:
                values.append(RadicalBound(Fraction(2), rad))
    elif conjecture_id == 46:
        for u, v in G.edges():
            rad = (
                2 * (d[u] ** 2 + d[v] ** 2)
                - Fraction(16 * d[u] * d[v], 1) / (m[u] + m[v])
                + 4
            )
            if rad >= 0:
                values.append(RadicalBound(Fraction(2), rad))
    elif conjecture_id == 56:
        for u, v in G.edges():
            values.append(
                RadicalBound(
                    ((d[u] ** 2 + d[v] ** 2) * (m[u] + m[v]))
                    / (2 * d[u] * d[v])
                )
            )
    else:
        raise VerificationError(
            "supported max-Laplacian conjecture ids are 11, 40, 44, 46, 56"
        )
    return values


def _extract_graph_and_optional_vector(candidate: Any) -> tuple[nx.Graph, Any | None, int | None]:
    if isinstance(candidate, Mapping):
        vector = _mapping_get_first(
            candidate,
            "rayleigh_vector",
            "vector",
            "eigenvector_certificate",
        )
        cid_raw = _mapping_get_first(
            candidate, "conjecture_id", "id", "instance"
        )
        cid = None if cid_raw is None else _to_int(cid_raw, name="conjecture id")
        return _parse_graph(candidate), vector, cid
    return _parse_graph(candidate), None, None


def verify_max_laplacian_eigenvalue(candidate: Any, *, conjecture_id: int | None = None) -> ValidationResult:
    """Verify a counterexample to a supported max-Laplacian upper bound.

    Bounds 44 and 46 are open. Bounds 11, 40, and 56 have published
    counterexamples and are retained as level-0 instances.
    Status source: Damnjanovic--Ha--Stevanovic (2026), arXiv:2606.14550.

    Witnesses may supply just a graph; optionally they may also
    supply a rational Rayleigh vector proving a lower bound on the largest Laplacian
    eigenvalue, which avoids characteristic-polynomial computation.
    """
    G, vector, embedded_id = _extract_graph_and_optional_vector(candidate)
    fixed_id = (
        None
        if conjecture_id is None
        else _to_int(conjecture_id, name="conjecture id")
    )
    if fixed_id is not None and embedded_id is not None and fixed_id != embedded_id:
        raise VerificationError("embedded conjecture_id conflicts with the selected validator")
    cid = fixed_id if fixed_id is not None else embedded_id
    if cid is None:
        raise VerificationError("max-Laplacian validator needs conjecture_id in {11,40,44,46,56}")
    if cid not in {11, 40, 44, 46, 56}:
        raise VerificationError("supported max-Laplacian conjecture ids are 11, 40, 44, 46, 56")
    if G.number_of_nodes() < 2:
        raise VerificationError("graph must have at least two vertices")
    if not nx.is_connected(G):
        raise VerificationError("graph must be connected")
    bounds = _laplacian_bound_values(G, cid)
    if not bounds:
        # Every local square-root term is undefined, so the paper's convention
        # makes the maximum negative infinity.
        return ValidationResult(
            True,
            f"Max Laplacian eigenvalue conjecture {cid}",
            "valid counterexample witness: every local bound term is undefined",
            {"vertices": G.number_of_nodes(), "defined_bound_terms": 0},
        )
    L_rows = _laplacian_rows(G)

    if vector is not None:
        q = _rayleigh_quotient_for_rows(L_rows, vector)
        if all(_compare_fraction_to_radical_bound(q, bound) > 0 for bound in bounds):
            return ValidationResult(
                True,
                f"Max Laplacian eigenvalue conjecture {cid}",
                "valid counterexample witness via Rayleigh certificate",
                {
                    "vertices": G.number_of_nodes(),
                    "rayleigh_quotient": str(q),
                },
            )
        # Fall through to exact eigenvalue computation; the supplied vector may simply be weak.

    P = _charpoly_from_integer_symmetric_rows(L_rows)
    for bits in (40, 80, 160, 320, 640, 1280):
        lo, hi = _largest_root_interval(P, bits=bits)
        if all(_compare_fraction_to_radical_bound(lo, bound) > 0 for bound in bounds):
            return ValidationResult(
                True,
                f"Max Laplacian eigenvalue conjecture {cid}",
                "valid counterexample witness",
                {
                    "vertices": G.number_of_nodes(),
                    "lambda_max_lower_bound": str(lo),
                },
            )
        if any(_compare_fraction_to_radical_bound(hi, bound) <= 0 for bound in bounds):
            return ValidationResult(
                False,
                f"Max Laplacian eigenvalue conjecture {cid}",
                "largest Laplacian eigenvalue does not exceed the bound",
            )
    raise VerificationError(
        "could not decide the strict Laplacian eigenvalue comparison exactly; "
        "provide a Rayleigh certificate"
    )


def verify_max_laplacian_11(candidate: Any) -> ValidationResult:
    """Verify a witness for refuted Bound 11, retained as a level-0 calibration."""
    return verify_max_laplacian_eigenvalue(candidate, conjecture_id=11)


def verify_max_laplacian_40(candidate: Any) -> ValidationResult:
    """Verify a witness for refuted Bound 40, retained as a level-0 calibration."""
    return verify_max_laplacian_eigenvalue(candidate, conjecture_id=40)


def verify_max_laplacian_44(candidate: Any) -> ValidationResult:
    """Verify a counterexample witness for open Bound 44."""
    return verify_max_laplacian_eigenvalue(candidate, conjecture_id=44)


def verify_max_laplacian_46(candidate: Any) -> ValidationResult:
    """Verify a counterexample witness for open Bound 46."""
    return verify_max_laplacian_eigenvalue(candidate, conjecture_id=46)


def verify_max_laplacian_56(candidate: Any) -> ValidationResult:
    """Verify a witness for refuted Bound 56, retained as a level-0 calibration."""
    return verify_max_laplacian_eigenvalue(candidate, conjecture_id=56)


# ---------------------------------------------------------------------------
# 9. Distance spectral independence/inertia conjecture
# ---------------------------------------------------------------------------


def _graph_girth(G: nx.Graph) -> int | None:
    """Return girth, or None for forests."""
    best = math.inf
    for start in G.nodes():
        dist = {start: 0}
        parent = {start: None}
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in G.neighbors(u):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    best = min(best, dist[u] + dist[v] + 1)
    return None if best == math.inf else int(best)


def _distance_matrix_rows(G: nx.Graph) -> list[list[int]]:
    n = G.number_of_nodes()
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    return [[int(lengths[i][j]) for j in range(n)] for i in range(n)]


def _graph_vertex_label_map_from_candidate(
    candidate: Mapping[str, Any], *, expected_order: int
) -> dict[Any, int] | None:
    """Recover the common label->normalized-index map when the graph supplies vertices."""
    graph_payload = None
    for key in ("undirected_graph", "graph", "G"):
        if key in candidate:
            graph_payload = candidate[key]
            break
    if graph_payload is None:
        graph_payload = candidate
    metadata = graph_payload if isinstance(graph_payload, Mapping) else candidate
    if isinstance(metadata, Mapping):
        vertices_raw = metadata.get(
            "vertices",
            metadata.get(
                "nodes",
                metadata.get(
                    "vertex_labels", metadata.get("labels")
                ),
            ),
        )
        if vertices_raw is not None:
            labels = []
            for value in _as_nonstring_list(
                vertices_raw, name="graph vertex labels"
            ):
                label = _hashable_key(value)
                try:
                    hash(label)
                except TypeError as exc:
                    raise VerificationError(
                        "graph vertex labels must be hashable"
                    ) from exc
                labels.append(label)
            if len(set(labels)) != len(labels):
                raise VerificationError("graph vertex labels must be distinct")
            if len(labels) != expected_order:
                raise VerificationError(
                    "graph vertex label list does not match the graph order"
                )
            return {label: i for i, label in enumerate(labels)}
    if isinstance(graph_payload, nx.Graph):
        try:
            labels = sorted(graph_payload.nodes())
        except TypeError:
            labels = list(graph_payload.nodes())
        if len(labels) != expected_order:  # pragma: no cover - defensive
            raise VerificationError("graph label map does not match the graph order")
        return {_hashable_key(label): i for i, label in enumerate(labels)}
    return None


def _parse_independent_set(candidate: Mapping[str, Any], G: nx.Graph) -> list[int] | None:
    label_to_index = _graph_vertex_label_map_from_candidate(
        candidate, expected_order=G.number_of_nodes()
    )
    for key in ("independent_set", "stable_set", "alpha_set", "S"):
        if key in candidate:
            payload = candidate[key]
            vertices = []
            for item in _as_nonstring_list(
                payload, name="independent set"
            ):
                item_key = _hashable_key(item)
                if label_to_index is not None and item_key in label_to_index:
                    v = label_to_index[item_key]
                else:
                    v = _to_int(item, name="independent set vertex")
                if not 0 <= v < G.number_of_nodes():
                    raise VerificationError("independent set vertex out of range after normalization")
                vertices.append(v)
            if len(set(vertices)) != len(vertices):
                raise VerificationError("independent set contains duplicate vertices")
            for u, v in itertools.combinations(vertices, 2):
                if G.has_edge(u, v):
                    raise VerificationError("claimed independent set is not independent")
            return vertices
    return None


def _exact_independence_number(G: nx.Graph) -> int:
    complement = nx.complement(G)
    best = 0
    for clique in nx.find_cliques(complement):
        if len(clique) > best:
            best = len(clique)
    return best


def verify_distance_spectral_independence_inertia(candidate: Any) -> ValidationResult:
    """Verify alpha(G) > p_-(D(G)) + p_0(D(G)) under connected, girth >= 5, m >= n."""
    if isinstance(candidate, Mapping):
        G = _parse_graph(candidate)
        independent = _parse_independent_set(candidate, G)
    else:
        G = _parse_graph(candidate)
        independent = None
    n = G.number_of_nodes()
    m_edges = G.number_of_edges()
    if n < 3:
        raise VerificationError("distance-spectral conjecture assumes n >= 3")
    if not nx.is_connected(G):
        raise VerificationError("graph must be connected")
    if m_edges < n:
        raise VerificationError("graph must satisfy m >= n")
    girth = _graph_girth(G)
    if girth is not None and girth < 5:
        raise VerificationError("graph must have girth at least 5")
    D_rows = _distance_matrix_rows(G)
    p_pos, p_neg, p_zero = _inertia_from_integer_symmetric_rows(D_rows)
    threshold = p_neg + p_zero
    if independent is not None:
        alpha_lower = len(independent)
        if alpha_lower > threshold:
            return ValidationResult(True, "Distance spectral independence/inertia", "valid counterexample witness via independent-set certificate", {"n": n, "m": m_edges, "independent_set_size": alpha_lower, "p_minus": p_neg, "p_zero": p_zero})
        # The optional certificate was valid but too weak; fall back to exact alpha(G)
        # rather than rejecting a graph witness that may still be a counterexample.
    alpha = _exact_independence_number(G)
    ok = alpha > threshold
    return ValidationResult(
        ok,
        "Distance spectral independence/inertia",
        "valid counterexample witness" if ok else "independence/inertia inequality is not violated",
        {"n": n, "m": m_edges, "alpha": alpha, "p_plus": p_pos, "p_minus": p_neg, "p_zero": p_zero},
    )


# ---------------------------------------------------------------------------
# 10. Erdos 97: convex polygon equidistance
# ---------------------------------------------------------------------------


def _quadruples_as_list(payload: Any, *, n: int) -> list[Any]:
    if isinstance(payload, Mapping):
        out: list[Any | None] = [None] * n
        for key, quad in payload.items():
            i = _to_int(key, name="quadruple vertex index")
            if not 0 <= i < n:
                raise VerificationError("quadruple vertex index out of range")
            if out[i] is not None:
                raise VerificationError("duplicate quadruple for a vertex")
            out[i] = quad
        if any(quad is None for quad in out):
            raise VerificationError("quadruple dictionary must give one quadruple for every vertex")
        return list(out)
    values = _as_nonstring_list(payload, name="quadruples")
    if len(values) != n:
        raise VerificationError("quadruples must contain one 4-tuple/list for each vertex")
    return values


def _remap_quadruples_for_order(quadruples: Sequence[Any], old_order: Sequence[int]) -> list[list[int]]:
    old_to_new = {old: new for new, old in enumerate(old_order)}
    reordered_quads: list[list[int]] = []
    for old_i in old_order:
        quad = _as_nonstring_list(
            quadruples[old_i], name="equidistant quadruple"
        )
        new_quad = []
        for raw_j in quad:
            old_j = _to_int(raw_j, name="quadruple index")
            if old_j not in old_to_new:
                raise VerificationError("quadruple index out of range")
            new_quad.append(old_to_new[old_j])
        reordered_quads.append(new_quad)
    return reordered_quads


def _exact_real_cuberoot(value: Any) -> sp.Expr:
    return sp.real_root(value, 3)


_POLYGON_COORDINATE_LOCALS = {
    "pi": sp.pi,
    "cos": sp.cos,
    "sin": sp.sin,
    "tan": sp.tan,
    "cbrt": _exact_real_cuberoot,
}


def _parse_points(candidate: Any) -> tuple[list[tuple[sp.Expr, sp.Expr]], Any | None]:
    quadruples = None
    payload = candidate
    if isinstance(candidate, Mapping):
        payload = _mapping_get_first(
            candidate, "points", "vertices", "coordinates"
        )
        quadruples = _mapping_get_first(
            candidate, "quadruples", "equidistant_quadruples"
        )
        if "cyclic_order" in candidate and payload is not None:
            raw = _as_nonstring_list(payload, name="point list")
            order_payload = _as_nonstring_list(
                candidate["cyclic_order"], name="cyclic_order"
            )
            order = [_to_int(i, name="cyclic_order") for i in order_payload]
            if sorted(order) != list(range(len(raw))):
                raise VerificationError("cyclic_order must be a permutation of point indices")
            if quadruples is not None:
                quadruples = _remap_quadruples_for_order(_quadruples_as_list(quadruples, n=len(raw)), order)
            payload = [raw[i] for i in order]
    point_values = _as_nonstring_list(payload, name="point list")
    points = []
    for i, point_raw in enumerate(point_values):
        point = _as_nonstring_list(point_raw, name=f"point {i}")
        if len(point) != 2:
            raise VerificationError("each point must be a pair of exact coordinates")
        x = _to_exact_sympy(point[0], name=f"point[{i}].x", extra_locals=_POLYGON_COORDINATE_LOCALS)
        y = _to_exact_sympy(point[1], name=f"point[{i}].y", extra_locals=_POLYGON_COORDINATE_LOCALS)
        if (
            x.is_real is not True
            or y.is_real is not True
            or x.is_finite is not True
            or y.is_finite is not True
        ):
            raise VerificationError("polygon coordinates must be finite real values")
        points.append((x, y))
    return points, quadruples



def _points_are_pairwise_distinct(points: Sequence[tuple[sp.Expr, sp.Expr]]) -> bool:
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if _expr_is_zero(points[i][0] - points[j][0]) and _expr_is_zero(points[i][1] - points[j][1]):
                return False
    return True


def _cross(o: tuple[sp.Expr, sp.Expr], a: tuple[sp.Expr, sp.Expr], b: tuple[sp.Expr, sp.Expr]) -> sp.Expr:
    return sp.expand((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _strictly_convex_in_given_order(points: Sequence[tuple[sp.Expr, sp.Expr]]) -> bool:
    n = len(points)
    if n < 5:
        return False
    signs = []
    for i in range(n):
        s = _sign_of_exact_real(_cross(points[i], points[(i + 1) % n], points[(i + 2) % n]), name="polygon turn")
        if s == 0:
            return False
        signs.append(s)
    if not all(s == signs[0] for s in signs):
        return False
    # Check non-adjacent edges do not cross by verifying all other vertices lie strictly on
    # the same side of every oriented edge. This catches simple star-shaped orderings.
    orientation = signs[0]
    for i in range(n):
        a = points[i]
        b = points[(i + 1) % n]
        for j in range(n):
            if j in (i, (i + 1) % n):
                continue
            s = _sign_of_exact_real(_cross(a, b, points[j]), name="convexity side test")
            if s != orientation:
                return False
    return True


def _angle_sort_points_with_order(points: Sequence[tuple[sp.Expr, sp.Expr]]) -> tuple[list[tuple[sp.Expr, sp.Expr]], list[int]]:
    """Cyclically order an unordered strictly-convex point set using exact signs."""
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)

    enriched = [(i, p, sp.expand(p[0] - cx), sp.expand(p[1] - cy)) for i, p in enumerate(points)]

    def half(dx: sp.Expr, dy: sp.Expr) -> int:
        sy = _sign_of_exact_real(dy, name="polar-angle y displacement")
        if sy > 0:
            return 0
        if sy < 0:
            return 1
        sx = _sign_of_exact_real(dx, name="polar-angle x displacement")
        return 0 if sx >= 0 else 1

    def cmp(a: tuple[int, tuple[sp.Expr, sp.Expr], sp.Expr, sp.Expr], b: tuple[int, tuple[sp.Expr, sp.Expr], sp.Expr, sp.Expr]) -> int:
        i, _, ax, ay = a
        j, _, bx, by = b
        ha = half(ax, ay)
        hb = half(bx, by)
        if ha != hb:
            return -1 if ha < hb else 1
        cross_sign = _sign_of_exact_real(sp.expand(ax * by - ay * bx), name="polar-angle comparison")
        if cross_sign > 0:
            return -1
        if cross_sign < 0:
            return 1
        # Same ray from the centroid.  This cannot happen for the vertex set of a
        # strictly convex polygon, but tie-break exactly by distance for diagnostics.
        da = sp.expand(ax * ax + ay * ay)
        db = sp.expand(bx * bx + by * by)
        dist_sign = _sign_of_exact_real(da - db, name="polar-angle equal-ray tie-break")
        if dist_sign != 0:
            return -1 if dist_sign < 0 else 1
        return (i > j) - (i < j)

    ordered = sorted(enriched, key=functools.cmp_to_key(cmp))
    return [p for _, p, _, _ in ordered], [i for i, _, _, _ in ordered]


def _squared_distance(p: tuple[sp.Expr, sp.Expr], q: tuple[sp.Expr, sp.Expr]) -> sp.Expr:
    return sp.expand((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


def _check_distance_quadruple(points: Sequence[tuple[sp.Expr, sp.Expr]], i: int, quad: Sequence[Any]) -> bool:
    n = len(points)
    indices = [
        _to_int(v, name="quadruple index")
        for v in _as_nonstring_list(quad, name="equidistant quadruple")
    ]
    if len(indices) != 4 or len(set(indices)) != 4:
        raise VerificationError("each equidistant quadruple must contain four distinct indices")
    if any(not 0 <= j < n for j in indices):
        raise VerificationError("quadruple index out of range")
    if i in indices:
        raise VerificationError("equidistant quadruple cannot include the base vertex")
    d0 = _squared_distance(points[i], points[indices[0]])
    return all(_expr_is_zero(_squared_distance(points[i], points[j]) - d0) for j in indices[1:])


def _has_four_equidistant_others(points: Sequence[tuple[sp.Expr, sp.Expr]], i: int) -> bool:
    distances = [(j, _squared_distance(points[i], points[j])) for j in range(len(points)) if j != i]
    used = [False] * len(distances)
    for idx, (_, d) in enumerate(distances):
        if used[idx]:
            continue
        count = 1
        used[idx] = True
        for k in range(idx + 1, len(distances)):
            if _expr_is_zero(distances[k][1] - d):
                used[k] = True
                count += 1
        if count >= 4:
            return True
    return False


def verify_erdos_97(candidate: Any) -> ValidationResult:
    """Verify a convex polygon in which every vertex has four other equidistant vertices.

    Points are expected in cyclic order unless a dict supplies a ``cyclic_order`` field.
    Optional quadruples may be supplied to avoid all-pairs grouping ambiguity; quadruple
    indices are interpreted in the same indexing as the supplied points and are remapped
    when the verifier reorders the points.
    """
    points, quadruples = _parse_points(candidate)
    n = len(points)
    if n < 5:
        raise VerificationError(
            "Erdos 97 witness must contain at least five polygon vertices"
        )
    if not _points_are_pairwise_distinct(points):
        raise VerificationError("points must be distinct")
    quad_list = None if quadruples is None else _quadruples_as_list(quadruples, n=n)

    if not _strictly_convex_in_given_order(points):
        sorted_points, old_order = _angle_sort_points_with_order(points)
        if _strictly_convex_in_given_order(sorted_points):
            if quad_list is not None:
                quad_list = _remap_quadruples_for_order(quad_list, old_order)
            points = sorted_points
        else:
            raise VerificationError("points are not a strictly convex polygon in the supplied or inferred cyclic order")

    if quad_list is not None:
        if all(_check_distance_quadruple(points, i, quad) for i, quad in enumerate(quad_list)):
            return ValidationResult(True, "Erdos 97", "valid counterexample witness", {"vertices": n})
        # The optional quadruple certificate was well-formed but too weak/wrong;
        # fall back to exact distance grouping on the point set itself.

    for i in range(n):
        if not _has_four_equidistant_others(points, i):
            return ValidationResult(False, "Erdos 97", f"vertex {i} does not have four equidistant other vertices")
    return ValidationResult(True, "Erdos 97", "valid counterexample witness", {"vertices": n})

# ---------------------------------------------------------------------------
# 11. Finite Er91/Er96 C6 challenge associated with Erdos 811
# ---------------------------------------------------------------------------


def _parse_color_matrix(candidate: Any) -> tuple[int, list[list[int]]]:
    t_raw = None
    payload = candidate
    if isinstance(candidate, Mapping):
        t_raw = candidate.get("t")
        payload = _mapping_get_first(
            candidate,
            "matrix",
            "color_matrix",
            "coloring",
            "colors",
            "edge_colors",
            "C",
        )
    matrix_rows = _as_nonstring_list(payload, name="color matrix")
    rows = []
    for i, row in enumerate(matrix_rows):
        if isinstance(row, str):
            stripped = row.strip()
            # Accept compact row strings such as "012345" and, for the diagonal
            # only, common single-character sentinels such as '.', '*', 'x', or '-'.
            if stripped and all(ch in "012345.*xX-" for ch in stripped):
                tokens = list(stripped)
            else:
                tokens = [tok for tok in row.replace(",", " ").split() if tok]
        else:
            tokens = _as_nonstring_list(row, name=f"color matrix row {i}")
        parsed_row = []
        for j, value in enumerate(tokens):
            if i == j and (value is None or (isinstance(value, str) and value.strip() in {"", ".", "*", "x", "X", "-"})):
                parsed_row.append(-1)
            else:
                parsed_row.append(_to_int(value, name=f"color[{i}][{j}]"))
        rows.append(parsed_row)
    if not rows or any(len(row) != len(rows) for row in rows):
        raise VerificationError("color matrix must be square")
    N = len(rows)
    if (N - 1) % 6 != 0:
        raise VerificationError("matrix size must be 6t+1")
    t = (N - 1) // 6
    if t_raw is not None and _to_int(t_raw, name="t") != t:
        raise VerificationError("supplied t does not match matrix size")
    return t, rows


def _has_rainbow_c6(C: Sequence[Sequence[int]]) -> bool:
    """Return whether the complete edge-coloring contains a rainbow 6-cycle.

    The exact search represents a C6 as two internally vertex-disjoint 3-edge
    paths between the same pair of opposite vertices.  This reduces the scan from
    explicit 6-cycle enumeration to an O(N^4) path-pair test.
    """
    N = len(C)
    all_colors = (1 << 6) - 1
    color_bit = [[0 if i == j else 1 << C[i][j] for j in range(N)] for i in range(N)]
    vertex_bit = [1 << i for i in range(N)]

    for a in range(N):
        for b in range(a + 1, N):
            buckets: list[list[int]] = [[] for _ in range(1 << 6)]
            for x in range(N):
                if x == a or x == b:
                    continue
                ax = color_bit[a][x]
                x_bit = vertex_bit[x]
                for y in range(N):
                    if y == a or y == b or y == x:
                        continue
                    xy = color_bit[x][y]
                    if ax & xy:
                        continue
                    yb = color_bit[y][b]
                    if (ax | xy) & yb:
                        continue
                    mask = ax | xy | yb
                    internal_vertices = x_bit | vertex_bit[y]
                    complement = all_colors ^ mask
                    if any((other_vertices & internal_vertices) == 0 for other_vertices in buckets[complement]):
                        return True
                    buckets[mask].append(internal_vertices)
    return False


def verify_erdos_811(candidate: Any) -> ValidationResult:
    """Verify a finite counterexample to the Er91/Er96 rainbow-C6 challenge.

    The witness is one balanced 6-coloring of K_{6t+1}, for an eligible (necessarily
    even) positive t, with no rainbow C6. Such a witness refutes the assertion that
    every eligible finite order has the C6 property. It does not settle the current
    asymptotic formulation of Erdos Problem 811, which asks whether the property
    holds for all sufficiently large orders.
    """
    t, C = _parse_color_matrix(candidate)
    if t <= 0 or t % 2 != 0:
        raise VerificationError("t must be a positive even integer")
    N = len(C)
    for i in range(N):
        if C[i][i] not in (-1, 0, 6, 9):
            # Diagonal is ignored, but requiring a sentinel catches shifted/rectangular data.
            raise VerificationError("color matrix diagonal should be an ignored sentinel such as 0, -1, 6, or 9")
        counts = [0] * 6
        for j in range(N):
            if i == j:
                continue
            if C[i][j] != C[j][i]:
                raise VerificationError("color matrix must be symmetric")
            color = C[i][j]
            if not 0 <= color < 6:
                raise VerificationError("edge colors must be in {0,1,2,3,4,5}")
            counts[color] += 1
        if counts != [t] * 6:
            return ValidationResult(False, "Erdos 811", f"vertex {i} does not see exactly t edges of each color")

    if _has_rainbow_c6(C):
        return ValidationResult(False, "Erdos 811", "a rainbow C6 exists")
    return ValidationResult(True, "Erdos 811", "valid counterexample witness", {"t": t, "vertices": N})


# ---------------------------------------------------------------------------
# 12. Symmetric conference matrix of order 86 / SRG(85,42,20,21)
# ---------------------------------------------------------------------------


def _verify_srg_85_42_20_21(G: nx.Graph) -> None:
    if G.number_of_nodes() != 85:
        raise VerificationError("SRG form must have 85 vertices")
    if any(G.has_edge(v, v) for v in G.nodes()):
        raise VerificationError("SRG graph must be simple")
    degrees = dict(G.degree())
    if any(degrees[v] != 42 for v in G.nodes()):
        raise VerificationError("SRG graph is not 42-regular")
    neighbor_sets = {v: set(G.neighbors(v)) for v in G.nodes()}
    nodes = list(G.nodes())
    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            common = len(neighbor_sets[u] & neighbor_sets[v])
            expected = 20 if G.has_edge(u, v) else 21
            if common != expected:
                raise VerificationError("SRG common-neighbor parameter check failed")


def verify_symmetric_conference_matrix_86(candidate: Any) -> ValidationResult:
    """Verify a symmetric conference matrix of order 86 or an SRG(85,42,20,21)."""
    try_graph = False
    payload = candidate
    if isinstance(candidate, Mapping):
        if any(
            key in candidate
            for key in (
                "undirected_graph",
                "graph",
                "G",
                "graph6",
                "g6",
                "sparse6",
                "s6",
                "edges",
                "edge_list",
                "adjacency_matrix",
                "adjacency",
                "A",
            )
        ):
            try_graph = True
        if "conference_matrix" in candidate or "C" in candidate:
            payload = candidate.get("conference_matrix", candidate.get("C"))
            try_graph = False
        elif "matrix" in candidate:
            payload = candidate["matrix"]
        elif try_graph:
            payload = candidate
    elif isinstance(candidate, str):
        try_graph = True

    if try_graph or isinstance(payload, str):
        G = _parse_graph(payload)
        try:
            _verify_srg_85_42_20_21(G)
        except VerificationError as exc:
            return ValidationResult(False, "Symmetric conference matrix order 86", str(exc))
        return ValidationResult(True, "Symmetric conference matrix order 86", "valid counterexample witness via SRG", {"vertices": 85})

    rows_any_size = _parse_int_matrix(payload, name="conference/SRG matrix")
    if len(rows_any_size) == 85 and all(len(row) == 85 for row in rows_any_size) and all(v in (0, 1) for row in rows_any_size for v in row):
        G = _parse_graph(rows_any_size)
        try:
            _verify_srg_85_42_20_21(G)
        except VerificationError as exc:
            return ValidationResult(False, "Symmetric conference matrix order 86", str(exc))
        return ValidationResult(True, "Symmetric conference matrix order 86", "valid counterexample witness via SRG", {"vertices": 85})

    rows = rows_any_size
    _check_square_shape(rows, 86, name="conference matrix")
    for i in range(86):
        if rows[i][i] != 0:
            return ValidationResult(False, "Symmetric conference matrix order 86", "diagonal must be zero")
        for j in range(86):
            if i != j and rows[i][j] not in (-1, 1):
                return ValidationResult(False, "Symmetric conference matrix order 86", "off-diagonal entries must be ±1")
            if rows[i][j] != rows[j][i]:
                return ValidationResult(False, "Symmetric conference matrix order 86", "matrix must be symmetric")
    try:
        _check_gram_equals_scalar_identity(rows, 85, name="conference matrix")
    except VerificationError as exc:
        return ValidationResult(False, "Symmetric conference matrix order 86", str(exc))
    return ValidationResult(True, "Symmetric conference matrix order 86", "valid counterexample witness", {"order": 86})


# ---------------------------------------------------------------------------
# 13. RSHCD order 196 type -1
# ---------------------------------------------------------------------------


def verify_rshcd_196_type_minus(candidate: Any) -> ValidationResult:
    """Verify a regular symmetric Hadamard matrix with constant diagonal of order 196 and type -1."""
    n = 196
    root_n = 14
    H = _parse_pm1_matrix(candidate, n=n, name="RSHCD matrix")
    diag = H[0][0]
    if diag not in (-1, 1):
        raise VerificationError("RSHCD diagonal sign must be ±1")
    row_sum = sum(H[0])
    if row_sum not in (-root_n, root_n):
        return ValidationResult(False, "RSHCD(196,-1)", "row sum must be ±14")
    if diag * row_sum != -root_n:
        return ValidationResult(False, "RSHCD(196,-1)", "type -1 condition diag * row_sum = -14 fails")
    for i in range(n):
        if H[i][i] != diag:
            return ValidationResult(False, "RSHCD(196,-1)", "diagonal is not constant")
        if sum(H[i]) != row_sum:
            return ValidationResult(False, "RSHCD(196,-1)", "row sums are not constant")
        for j in range(i + 1, n):
            if H[i][j] != H[j][i]:
                return ValidationResult(False, "RSHCD(196,-1)", "matrix is not symmetric")
    try:
        _check_gram_equals_scalar_identity(H, n, name="RSHCD matrix")
    except VerificationError as exc:
        return ValidationResult(False, "RSHCD(196,-1)", str(exc))
    return ValidationResult(True, "RSHCD(196,-1)", "valid counterexample witness", {"order": n, "diagonal": diag, "row_sum": row_sum})


# ---------------------------------------------------------------------------
# 14. Steiner systems S(3,5,41) and S(3,6,46)
# ---------------------------------------------------------------------------


def _parse_blocks(candidate: Any) -> tuple[list[frozenset[Any]], list[Any] | None]:
    points = None
    payload = candidate
    if isinstance(candidate, Mapping):
        payload = _mapping_get_first(
            candidate, "blocks", "block_list", "B", "design"
        )
        points_payload = _mapping_get_first(
            candidate, "points", "point_set", "vertices"
        )
        if points_payload is not None:
            points = [
                _hashable_key(point)
                for point in _as_nonstring_list(
                    points_payload, name="point list"
                )
            ]
            try:
                set(points)
            except TypeError as exc:
                raise VerificationError(
                    "point list must contain hashable point labels"
                ) from exc
    block_values = _as_nonstring_list(payload, name="Steiner block list")
    blocks = []
    for index, block in enumerate(block_values):
        block_list = [
            _hashable_key(point)
            for point in _as_nonstring_list(
                block, name=f"Steiner block {index}"
            )
        ]
        try:
            block_set = frozenset(block_list)
        except TypeError as exc:
            raise VerificationError(
                "each block must contain hashable point labels"
            ) from exc
        if len(block_set) != len(block_list):
            raise VerificationError("a block contains a repeated point")
        blocks.append(block_set)
    return blocks, points


def verify_steiner_3_design(candidate: Any, *, block_size: int, v: int) -> ValidationResult:
    """Verify a Steiner system S(3, block_size, v)."""
    if block_size < 4 or v <= block_size:
        raise VerificationError("invalid Steiner 3-design parameters")
    if isinstance(candidate, Mapping):
        supplied_block_size = _mapping_get_first(
            candidate, "block_size", "k"
        )
        supplied_v = _mapping_get_first(candidate, "v", "num_points", "order")
        if supplied_block_size is not None and _to_int(
            supplied_block_size, name="block_size"
        ) != block_size:
            raise VerificationError(
                "supplied block_size conflicts with the selected Steiner validator"
            )
        if supplied_v is not None and _to_int(
            supplied_v, name="number of points"
        ) != v:
            raise VerificationError(
                "supplied point count conflicts with the selected Steiner validator"
            )
    blocks_raw, points_payload = _parse_blocks(candidate)
    if points_payload is None:
        point_order = []
        seen = set()
        for block in blocks_raw:
            for point in block:
                if point not in seen:
                    seen.add(point)
                    point_order.append(point)
    else:
        if len(set(points_payload)) != len(points_payload):
            raise VerificationError("point list contains duplicates")
        point_order = points_payload
    if len(point_order) != v:
        raise VerificationError(f"Steiner system must have exactly {v} points")
    point_index = {p: i for i, p in enumerate(point_order)}

    blocks: list[tuple[int, ...]] = []
    for block in blocks_raw:
        if len(block) != block_size:
            raise VerificationError(f"each block must have size {block_size}")
        try:
            idx_block = tuple(sorted(point_index[p] for p in block))
        except KeyError as exc:
            raise VerificationError("block contains a point outside the point set") from exc
        blocks.append(idx_block)
    if len(set(blocks)) != len(blocks):
        raise VerificationError("duplicate blocks are not allowed")

    expected_blocks_num = math.comb(v, 3)
    expected_blocks_den = math.comb(block_size, 3)
    if expected_blocks_num % expected_blocks_den != 0:
        raise VerificationError("parameters do not satisfy the divisibility condition")
    expected_blocks = expected_blocks_num // expected_blocks_den
    if len(blocks) != expected_blocks:
        return ValidationResult(False, f"Steiner S(3,{block_size},{v})", f"expected {expected_blocks} blocks, got {len(blocks)}")

    triple_count: Counter[tuple[int, int, int]] = Counter()
    for block in blocks:
        for triple in itertools.combinations(block, 3):
            triple_count[triple] += 1
            if triple_count[triple] > 1:
                return ValidationResult(False, f"Steiner S(3,{block_size},{v})", "some triple occurs more than once")
    if len(triple_count) != math.comb(v, 3):
        return ValidationResult(False, f"Steiner S(3,{block_size},{v})", "some triple occurs zero times")
    return ValidationResult(True, f"Steiner S(3,{block_size},{v})", "valid counterexample witness", {"v": v, "block_size": block_size, "blocks": len(blocks)})


def verify_steiner_3_5_41(candidate: Any) -> ValidationResult:
    return verify_steiner_3_design(candidate, block_size=5, v=41)


def verify_steiner_3_6_46(candidate: Any) -> ValidationResult:
    return verify_steiner_3_design(candidate, block_size=6, v=46)


# ---------------------------------------------------------------------------
# 15. Seymour's second-neighborhood conjecture
# ---------------------------------------------------------------------------


def _validate_and_relabel_oriented_graph(D: nx.DiGraph) -> nx.DiGraph:
    if not D.is_directed():
        raise VerificationError("oriented graph witness must be directed")
    if D.is_multigraph():
        raise VerificationError("oriented graph witness must not contain parallel arcs")
    if any(u == v for u, v in D.edges()):
        raise VerificationError("oriented graph witness must not contain loops")
    for u, v in D.edges():
        if D.has_edge(v, u):
            raise VerificationError(
                "opposite arcs are not allowed in an oriented graph"
            )
    try:
        return nx.convert_node_labels_to_integers(D, ordering="sorted")
    except TypeError:
        return nx.convert_node_labels_to_integers(D, ordering="default")


def _oriented_graph_from_adjacency_matrix(data: Any) -> nx.DiGraph:
    if hasattr(data, "tolist"):
        try:
            data = data.tolist()
        except Exception as exc:
            raise VerificationError(
                "could not convert the oriented adjacency matrix to rows"
            ) from exc
    rows = _parse_int_matrix(data, name="oriented adjacency matrix")
    if not _is_square_matrix(rows):
        raise VerificationError("oriented adjacency matrix must be nonempty and square")
    n = len(rows)
    D = nx.DiGraph()
    D.add_nodes_from(range(n))
    for i in range(n):
        if rows[i][i] != 0:
            raise VerificationError("oriented adjacency matrix must have zero diagonal")
        for j in range(n):
            if rows[i][j] not in (0, 1):
                raise VerificationError("oriented adjacency entries must be 0 or 1")
            if rows[i][j]:
                D.add_edge(i, j)
    return _validate_and_relabel_oriented_graph(D)


def _oriented_graph_from_arcs(
    arcs_payload: Any,
    *,
    n_raw: Any = None,
    vertices_raw: Any = None,
) -> nx.DiGraph:
    arcs = _as_nonstring_list(arcs_payload, name="directed arcs")

    def label_key(value: Any) -> Any:
        key = _hashable_key(value)
        try:
            hash(key)
        except TypeError as exc:
            raise VerificationError("directed graph vertex labels must be hashable") from exc
        return key

    D = nx.DiGraph()
    label_to_node: dict[Any, int] | None = None
    n: int | None = None
    if vertices_raw is not None:
        labels = [
            label_key(value)
            for value in _as_nonstring_list(
                vertices_raw, name="vertices/nodes"
            )
        ]
        if len(set(labels)) != len(labels):
            raise VerificationError("directed graph vertex labels must be distinct")
        if n_raw is not None and _to_int(
            n_raw, name="number of directed graph vertices"
        ) != len(labels):
            raise VerificationError(
                "number of directed graph vertices does not match the vertex label list"
            )
        label_to_node = {label: i for i, label in enumerate(labels)}
        D.add_nodes_from(range(len(labels)))
    elif n_raw is not None:
        n = _to_int(n_raw, name="number of directed graph vertices")
        if n < 0:
            raise VerificationError(
                "number of directed graph vertices must be nonnegative"
            )
        D.add_nodes_from(range(n))

    for arc_raw in arcs:
        arc = _as_nonstring_list(arc_raw, name="directed arc")
        if len(arc) != 2:
            raise VerificationError("every directed arc must be a 2-element sequence")
        if label_to_node is not None:
            u_key = label_key(arc[0])
            v_key = label_key(arc[1])
            if u_key not in label_to_node or v_key not in label_to_node:
                raise VerificationError(
                    "directed arc endpoint is outside the vertex label list"
                )
            u = label_to_node[u_key]
            v = label_to_node[v_key]
        elif n is not None:
            u = _to_int(arc[0], name="directed arc endpoint")
            v = _to_int(arc[1], name="directed arc endpoint")
            if not (0 <= u < n and 0 <= v < n):
                raise VerificationError("directed arc endpoint is outside 0..n-1")
        else:
            u = label_key(arc[0])
            v = label_key(arc[1])
        if u == v:
            raise VerificationError("oriented graph witness must not contain loops")
        if D.has_edge(u, v):
            raise VerificationError("oriented graph witness contains a duplicate arc")
        if D.has_edge(v, u):
            raise VerificationError(
                "opposite arcs are not allowed in an oriented graph"
            )
        D.add_edge(u, v)
    return _validate_and_relabel_oriented_graph(D)


def _parse_oriented_graph(data: Any) -> nx.DiGraph:
    """Parse an oriented graph from a DiGraph, adjacency matrix, or directed arcs."""
    if isinstance(data, Mapping):
        for key in (
            "directed_adjacency_matrix",
            "digraph_matrix",
            "tournament_matrix",
            "adjacency_matrix",
            "adjacency",
            "matrix",
            "rows",
            "array",
            "A",
        ):
            if key in data:
                D = _oriented_graph_from_adjacency_matrix(data[key])
                n_raw = data.get(
                    "n",
                    data.get(
                        "num_vertices", data.get("num_nodes", data.get("order"))
                    ),
                )
                if n_raw is not None and _to_int(
                    n_raw, name="number of directed graph vertices"
                ) != D.number_of_nodes():
                    raise VerificationError(
                        "number of directed graph vertices does not match the matrix"
                    )
                return D
        for key in (
            "arcs",
            "arc_list",
            "directed_edges",
            "directed_edge_list",
            "edges",
            "edge_list",
        ):
            if key in data:
                return _oriented_graph_from_arcs(
                    data[key],
                    n_raw=data.get(
                        "n",
                        data.get(
                            "num_vertices",
                            data.get("num_nodes", data.get("order")),
                        ),
                    ),
                    vertices_raw=data.get(
                        "vertices",
                        data.get(
                            "nodes",
                            data.get("vertex_labels", data.get("labels")),
                        ),
                    ),
                )
        for key in (
            "oriented_graph",
            "directed_graph",
            "digraph",
            "D",
            "graph",
            "G",
        ):
            if key in data:
                D = _parse_oriented_graph(data[key])
                n_raw = _mapping_get_first(
                    data, "n", "num_vertices", "num_nodes", "order"
                )
                if n_raw is not None and _to_int(
                    n_raw, name="number of directed graph vertices"
                ) != D.number_of_nodes():
                    raise VerificationError(
                        "number of directed graph vertices does not match the graph"
                    )
                return D
        raise VerificationError("unsupported oriented graph dict format")

    if isinstance(data, (nx.Graph, nx.DiGraph)):
        if not data.is_directed():
            raise VerificationError("oriented graph witness must be directed")
        if data.is_multigraph():
            D = nx.DiGraph()
            D.add_nodes_from(data.nodes())
            for u, v in data.edges():
                if D.has_edge(u, v):
                    raise VerificationError(
                        "oriented graph witness must not contain parallel arcs"
                    )
                D.add_edge(u, v)
        else:
            D = nx.DiGraph(data)
        return _validate_and_relabel_oriented_graph(D)

    if hasattr(data, "tolist"):
        try:
            return _parse_oriented_graph(data.tolist())
        except VerificationError:
            raise
        except Exception as exc:
            raise VerificationError(
                "could not convert the oriented graph witness to nested lists"
            ) from exc

    if isinstance(data, Iterable) and not isinstance(
        data, (str, bytes, bytearray, Mapping)
    ):
        rows = _as_nonstring_list(data, name="oriented graph witness")
        normalized_rows = []
        for row in rows:
            if hasattr(row, "tolist"):
                try:
                    row = row.tolist()
                except Exception as exc:
                    raise VerificationError(
                        "could not convert oriented graph row to a list"
                    ) from exc
            elif (
                isinstance(row, Iterable)
                and not isinstance(row, (str, bytes, bytearray, Mapping))
                and not _is_nonstring_sequence(row)
            ):
                try:
                    row = list(row)
                except Exception as exc:
                    raise VerificationError(
                        "could not iterate over oriented graph row"
                    ) from exc
            normalized_rows.append(row)
        rows = normalized_rows
        parsed: list[list[int]] | None
        try:
            parsed = _parse_int_matrix(rows, name="oriented adjacency matrix")
        except VerificationError:
            parsed = None
        if (
            parsed is not None
            and _is_square_matrix(parsed)
            and all(value in (0, 1) for row in parsed for value in row)
        ):
            return _oriented_graph_from_adjacency_matrix(parsed)
        if all(_is_nonstring_sequence(arc) and len(arc) == 2 for arc in rows):
            return _oriented_graph_from_arcs(rows)
    raise VerificationError("unsupported oriented graph format")


def verify_seymour_second_neighborhood(candidate: Any) -> ValidationResult:
    """Verify a counterexample to Seymour's second-neighborhood conjecture.

    For every vertex v of an oriented graph D, let N+(v) be its out-neighbors
    and let N++(v) contain the vertices at directed distance exactly two from v
    (excluding v and its out-neighbors). The conjecture asserts that some v has
    |N++(v)| >= |N+(v)|. A counterexample must therefore satisfy the strict
    reverse inequality at every vertex.

    Accepted witnesses are NetworkX DiGraphs; raw 0/1 adjacency matrices
    (including compact string rows and objects with ``tolist``); raw directed
    arc lists; or mappings that wrap a graph, matrix, or arc list and may give
    an order or explicit vertex labels. Loops, parallel arcs, and opposite arc
    pairs are rejected.

    Status source: Bai--Li--Park (2026), arXiv:2607.18047.
    """
    D = _parse_oriented_graph(candidate)
    n = D.number_of_nodes()
    if n == 0:
        raise VerificationError("Seymour witness must contain at least one vertex")

    out_masks: list[int] = []
    for u in range(n):
        mask = 0
        for v in D.successors(u):
            mask |= 1 << v
        out_masks.append(mask)

    max_gap: int | None = None
    for u, first_mask in enumerate(out_masks):
        second_mask = 0
        pending = first_mask
        while pending:
            bit = pending & -pending
            v = bit.bit_length() - 1
            second_mask |= out_masks[v]
            pending ^= bit
        second_mask &= ~first_mask
        second_mask &= ~(1 << u)
        first_size = first_mask.bit_count()
        second_size = second_mask.bit_count()
        gap = second_size - first_size
        max_gap = gap if max_gap is None else max(max_gap, gap)
        if gap >= 0:
            return ValidationResult(
                False,
                "Seymour second-neighborhood conjecture",
                f"vertex {u} has |N++|={second_size} >= |N+|={first_size}",
            )

    return ValidationResult(
        True,
        "Seymour second-neighborhood conjecture",
        "valid counterexample witness",
        {
            "vertices": n,
            "arcs": D.number_of_edges(),
            "maximum_second_minus_first": max_gap,
        },
    )


# ---------------------------------------------------------------------------
# Convenience registry
# ---------------------------------------------------------------------------


VALIDATORS: dict[str, Callable[..., ValidationResult]] = {
    "turyn_type_tt46": verify_turyn_type_tt46,
    "skew_hadamard_356": verify_skew_hadamard_356,
    "cocyclic_hadamard_188": verify_cocyclic_hadamard_188,
    "finite_magma_e677_not_e255": verify_finite_magma_e677_not_e255,
    "line_graph_inertia": verify_line_graph_inertia,
    "dual_finite_magma": verify_dual_finite_magma,
    "max_laplacian_eigenvalue": verify_max_laplacian_eigenvalue,
    "max_laplacian_11": verify_max_laplacian_11,
    "max_laplacian_40": verify_max_laplacian_40,
    "max_laplacian_44": verify_max_laplacian_44,
    "max_laplacian_46": verify_max_laplacian_46,
    "max_laplacian_56": verify_max_laplacian_56,
    "distance_spectral_independence_inertia": verify_distance_spectral_independence_inertia,
    "erdos_97": verify_erdos_97,
    "erdos_811": verify_erdos_811,
    "symmetric_conference_matrix_86": verify_symmetric_conference_matrix_86,
    "rshcd_196_type_minus": verify_rshcd_196_type_minus,
    "steiner_3_design": verify_steiner_3_design,
    "steiner_3_5_41": verify_steiner_3_5_41,
    "steiner_3_6_46": verify_steiner_3_6_46,
    "seymour_second_neighborhood": verify_seymour_second_neighborhood,
}


__all__ = [
    "ValidationResult",
    "VerificationError",
    "verify_turyn_type_tt46",
    "verify_skew_hadamard_356",
    "verify_cocyclic_hadamard_188",
    "verify_finite_magma_e677_not_e255",
    "verify_line_graph_inertia",
    "verify_dual_finite_magma",
    "verify_max_laplacian_eigenvalue",
    "verify_max_laplacian_11",
    "verify_max_laplacian_40",
    "verify_max_laplacian_44",
    "verify_max_laplacian_46",
    "verify_max_laplacian_56",
    "verify_distance_spectral_independence_inertia",
    "verify_erdos_97",
    "verify_erdos_811",
    "verify_symmetric_conference_matrix_86",
    "verify_rshcd_196_type_minus",
    "verify_steiner_3_design",
    "verify_steiner_3_5_41",
    "verify_steiner_3_6_46",
    "verify_seymour_second_neighborhood",
    "VALIDATORS",
]
