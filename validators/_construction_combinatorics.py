"""Exact verifiers for combinatorial construction witnesses."""

from __future__ import annotations
from collections import Counter
import itertools
import math
from typing import Any, Iterable, Mapping, Sequence
import networkx as nx
from ._construction_common import (
    ValidationResult,
    VerificationError,
    _as_nonstring_list,
    _hashable_key,
    _is_nonstring_sequence,
    _is_square_matrix,
    _mapping_get_first,
    _parse_int_matrix,
    _to_int,
)


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
