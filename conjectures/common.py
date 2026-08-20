"""Shared result, parsing, and exact-arithmetic helpers for conjecture validators."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


class VerificationError(ValueError):
    """Raised when a witness is malformed or exact verification needs more data."""


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    conjecture: str
    reason: str
    normalized_witness: Any | None = None


_sympy_locals = {"I": sp.I, "sqrt": sp.sqrt, "Rational": sp.Rational, "S": sp.S}
_FLOAT_LITERAL_RE = re.compile(
    r"(?<![A-Za-z0-9_.])(?:\d+\.\d*|\.\d+|(?:\d+|\d*\.\d+)[eE][+-]?\d+)(?![A-Za-z0-9_.])"
)
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_SAFE_PARSE_GLOBALS = {
    "__builtins__": {},
    "Add": sp.Add,
    "Mul": sp.Mul,
    "Pow": sp.Pow,
    "Integer": sp.Integer,
    "Rational": sp.Rational,
    "Float": sp.Float,  # parsed floats are rejected immediately after parsing
    "Symbol": sp.Symbol,
}
_SAFE_TRANSFORMATIONS = standard_transformations + (convert_xor, implicit_multiplication_application)


def _safe_parse_expr_string(
    value: str,
    *,
    name: str,
    extra_locals: Mapping[str, Any] | None = None,
) -> sp.Expr:
    """Parse a string expression with no builtins and a very small SymPy environment."""
    if not value.isascii():
        raise VerificationError(
            f"{name}: non-ASCII characters are not accepted in exact expressions"
        )
    _reject_float_literal_string(value, name=name)
    locals_dict = dict(_sympy_locals)
    if extra_locals is not None:
        locals_dict.update(extra_locals)
    allowed_names = set(locals_dict) | (set(_SAFE_PARSE_GLOBALS) - {"__builtins__"})
    for token in set(_IDENTIFIER_RE.findall(value)):
        if token not in allowed_names:
            raise VerificationError(f"{name}: unknown name {token!r} is not accepted in exact expressions")
    try:
        expr = parse_expr(
            value,
            local_dict=locals_dict,
            global_dict=_SAFE_PARSE_GLOBALS,
            transformations=_SAFE_TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise VerificationError(f"could not parse exact expression for {name}: {value!r}") from exc
    if not isinstance(expr, sp.Basic):
        raise VerificationError(f"could not parse exact expression for {name}: {value!r}")
    return expr


def _reject_float_literal_string(value: Any, *, name: str) -> None:
    """Reject decimal/scientific string literals that SymPy could rationalize silently."""
    if isinstance(value, str) and _FLOAT_LITERAL_RE.search(value):
        raise VerificationError(f"{name} must be exact: decimal/scientific literals are not accepted")


def _is_nonstring_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _as_nonstring_list(value: Any, *, name: str) -> list[Any]:
    """Materialize a non-string iterable, including common array-like objects."""
    if hasattr(value, "tolist") and not isinstance(
        value, (str, bytes, bytearray, Mapping)
    ):
        try:
            value = value.tolist()
        except Exception as exc:
            raise VerificationError(f"could not convert {name} to a list") from exc
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Iterable
    ):
        raise VerificationError(f"{name} must be a non-string iterable")
    try:
        return list(value)
    except Exception as exc:  # pragma: no cover - defensive for unusual iterables
        raise VerificationError(f"could not iterate over {name}") from exc


def _mapping_get_first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present key, preserving false-y values such as 0 or []."""
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _hashable_key(value: Any) -> Any:
    """Convert simple nested list labels to hashable keys without changing scalars."""
    if isinstance(value, list):
        return tuple(_hashable_key(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_hashable_key(v) for v in value)
    return value


def _relabel_graph_to_consecutive_integers(G: nx.Graph) -> nx.Graph:
    """Relabel a simple graph to 0..n-1, tolerating mixed incomparable labels."""
    try:
        return nx.convert_node_labels_to_integers(G, ordering="sorted")
    except TypeError:
        return nx.convert_node_labels_to_integers(G, ordering="default")


def _to_int(value: Any, *, name: str) -> int:
    """Parse an exact integer value, rejecting booleans, floats, and non-integral rationals."""
    if isinstance(value, bool):
        raise VerificationError(f"{name} must be an integer, not bool")
    if value is None:
        raise VerificationError(f"{name} must be an integer")
    if isinstance(value, complex):
        raise VerificationError(f"{name} must be an integer, not complex")
    if isinstance(value, str):
        expr = _safe_parse_expr_string(value, name=name)
    else:
        try:
            expr = sp.sympify(value, locals=_sympy_locals, rational=True)
        except Exception as exc:  # pragma: no cover - defensive
            raise VerificationError(f"{name} must be an integer") from exc
        if not isinstance(expr, sp.Basic):
            raise VerificationError(f"{name} must be an integer")
    if expr.has(sp.Float):
        raise VerificationError(f"{name} must be an exact integer, not a floating-point value")
    if expr.is_Integer is True:
        return int(expr)
    if expr.is_Rational is True and expr.q == 1:
        return int(expr.p)
    raise VerificationError(f"{name} must be an integer")


def _to_fraction(value: Any, *, name: str) -> Fraction:
    """Parse an exact real rational value as ``fractions.Fraction``."""
    if isinstance(value, bool):
        raise VerificationError(f"{name} must be a rational number, not bool")
    if isinstance(value, float):
        raise VerificationError(f"{name} must be exact, not a floating-point value")
    if isinstance(value, complex):
        raise VerificationError(f"{name} must be real rational, not complex")
    if isinstance(value, str):
        expr = _safe_parse_expr_string(value, name=name)
    else:
        try:
            expr = sp.sympify(value, locals=_sympy_locals, rational=True)
        except Exception as exc:  # pragma: no cover - defensive
            raise VerificationError(f"{name} must be a rational number") from exc
    if not isinstance(expr, sp.Basic) or expr.has(sp.Float) or expr.has(sp.I):
        raise VerificationError(f"{name} must be an exact real rational")
    if expr.is_Rational is True:
        return Fraction(int(expr.p), int(expr.q))
    raise VerificationError(f"{name} must be a rational number")


def _to_exact_sympy(
    value: Any,
    *,
    name: str = "value",
    extra_locals: Mapping[str, Any] | None = None,
    allow_symbols: bool = False,
) -> sp.Expr:
    """Parse an exact number/expression into SymPy."""
    if isinstance(value, bool):
        raise VerificationError(f"{name}: boolean inputs are not accepted in exact symbolic formats")
    if isinstance(value, float):
        raise VerificationError(f"{name}: floating-point inputs are not accepted in exact symbolic formats")
    if isinstance(value, complex):
        if value.real.is_integer() and value.imag.is_integer():
            return sp.Integer(int(value.real)) + sp.Integer(int(value.imag)) * sp.I
        raise VerificationError(f"{name}: complex floating-point inputs are not accepted")
    if isinstance(value, str):
        expr = _safe_parse_expr_string(value, name=name, extra_locals=extra_locals)
    else:
        locals_dict = dict(_sympy_locals)
        if extra_locals is not None:
            locals_dict.update(extra_locals)
        try:
            expr = sp.sympify(value, locals=locals_dict, rational=True)
        except Exception as exc:  # pragma: no cover - defensive
            raise VerificationError(f"could not parse exact expression for {name}: {value!r}") from exc
        if not isinstance(expr, sp.Basic):
            raise VerificationError(f"could not parse exact expression for {name}: {value!r}")
    if expr.has(sp.Float):
        raise VerificationError(f"{name}: floating-point inputs are not accepted in exact symbolic formats")
    if expr.has(sp.oo, -sp.oo, sp.zoo, sp.nan) or expr.is_finite is False:
        raise VerificationError(f"{name}: non-finite exact values are not accepted")
    if not allow_symbols and expr.free_symbols:
        raise VerificationError(f"{name}: symbolic variables are not accepted in numeric exact formats")
    return expr


def _expr_is_zero(expr: sp.Expr) -> bool:
    """Return whether a symbolic expression is provably the exact zero expression."""
    return bool(sp.simplify(sp.expand(expr)) == 0)


def _is_square_matrix(rows: Sequence[Sequence[Any]]) -> bool:
    return bool(rows) and all(len(row) == len(rows) for row in rows)


def _parse_int_matrix(data: Any, *, n: int | None = None, name: str = "matrix") -> list[list[int]]:
    """Parse an integer matrix, accepting array-like objects and iterable rows."""
    if isinstance(data, Mapping):
        for key in (
            "matrix",
            "rows",
            "adjacency_matrix",
            "incidence_matrix",
            "C",
            "M",
            "A",
            "array",
        ):
            if key in data:
                embedded_n_raw = _mapping_get_first(data, "n", "order", "size")
                embedded_n = (
                    None
                    if embedded_n_raw is None
                    else _to_int(embedded_n_raw, name=f"{name} order")
                )
                if n is not None and embedded_n is not None and n != embedded_n:
                    raise VerificationError(
                        f"{name}: supplied order conflicts with the required order"
                    )
                target_n = n if n is not None else embedded_n
                return _parse_int_matrix(data[key], n=target_n, name=name)
        raise VerificationError(f"{name} must be a matrix")

    rows: list[list[int]] = []
    for i, row in enumerate(_as_nonstring_list(data, name=name)):
        if isinstance(row, str):
            s = row.strip()
            if s and all(ch in "0123456789" for ch in s):
                tokens: list[Any] = list(s)
            else:
                tokens = s.replace(",", " ").split()
        else:
            try:
                tokens = _as_nonstring_list(row, name=f"{name} row {i}")
            except VerificationError as exc:
                raise VerificationError(f"{name}: row {i} is not a valid row") from exc
        rows.append([_to_int(v, name=f"{name}[{i}][{j}]") for j, v in enumerate(tokens)])
    if n is not None and (len(rows) != n or any(len(row) != n for row in rows)):
        raise VerificationError(f"{name} must be {n}x{n}")
    return rows


def _check_networkx_graph_is_simple_undirected(graph: nx.Graph) -> None:
    if graph.is_directed():
        raise VerificationError("graphs must be undirected")
    if graph.is_multigraph():
        seen: set[tuple[Any, Any]] = set()
        for u, v in graph.edges():
            if u == v:
                raise VerificationError("graphs must be simple: self-loops are not allowed")
            key = tuple(sorted((u, v), key=repr))
            if key in seen:
                raise VerificationError("graphs must be simple: parallel edges are not allowed")
            seen.add(key)
    elif any(u == v for u, v in graph.edges()):
        raise VerificationError("graphs must be simple: self-loops are not allowed")


def _undirected_edge_pair(value: Any) -> list[Any] | None:
    """Return two endpoints from any non-string iterable edge container."""
    if (
        isinstance(value, (str, bytes, bytearray, Mapping))
        or not isinstance(value, Iterable)
    ):
        return None
    try:
        endpoints = list(value)
    except Exception as exc:  # pragma: no cover - defensive for unusual iterables
        raise VerificationError("could not iterate over graph edge") from exc
    return endpoints if len(endpoints) == 2 else None


def _undirected_graph_from_adjacency_matrix(data: Any) -> nx.Graph:
    rows = _parse_int_matrix(data, name="adjacency matrix")
    if not _is_square_matrix(rows):
        raise VerificationError("adjacency matrix must be nonempty and square")
    if any(value not in (0, 1) for row in rows for value in row):
        raise VerificationError("adjacency matrix entries must be 0 or 1")
    n = len(rows)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        if rows[i][i] != 0:
            raise VerificationError("adjacency matrix must have zero diagonal")
        for j in range(i + 1, n):
            if rows[i][j] != rows[j][i]:
                raise VerificationError("adjacency matrix must be symmetric")
            if rows[i][j]:
                G.add_edge(i, j)
    return G


def _undirected_graph_from_graph_encoding(data: Any, *, sparse: bool) -> nx.Graph:
    if isinstance(data, str):
        try:
            encoded = data.strip().encode("ascii", errors="strict")
        except UnicodeEncodeError as exc:
            raise VerificationError(
                "graph6/sparse6 strings must be ASCII"
            ) from exc
    elif isinstance(data, (bytes, bytearray)):
        encoded = bytes(data).strip()
    else:
        kind = "sparse6" if sparse else "graph6"
        raise VerificationError(f"{kind} encoding must be a string or bytes")
    if sparse:
        if not (
            encoded.startswith(b":") or encoded.startswith(b">>sparse6<<:")
        ):
            raise VerificationError("invalid sparse6 string")
    elif encoded.startswith(b":") or encoded.startswith(b">>sparse6<<"):
        raise VerificationError("invalid graph6 string")
    try:
        if sparse:
            return nx.from_sparse6_bytes(encoded)
        return nx.from_graph6_bytes(encoded)
    except Exception as exc:
        kind = "sparse6" if sparse else "graph6"
        raise VerificationError(f"invalid {kind} string") from exc


def _parse_graph(graph_data: Any) -> nx.Graph:
    """Parse a simple undirected graph from standard exact witness formats.

    Accepted forms include graph6/sparse6 strings, NetworkX Graph/MultiGraph
    objects, raw adjacency matrices or edge iterables, array-like matrices with
    ``tolist()``, and mappings that wrap those forms with common aliases and
    optional order or vertex-label fields.
    """
    if isinstance(graph_data, Mapping):
        for key in ("undirected_graph", "graph", "G"):
            if key in graph_data:
                G = _parse_graph(graph_data[key])
                n_raw = _mapping_get_first(
                    graph_data, "n", "num_vertices", "num_nodes", "order"
                )
                if n_raw is not None and _to_int(
                    n_raw, name="number of graph vertices"
                ) != G.number_of_nodes():
                    raise VerificationError(
                        "number of graph vertices does not match the graph"
                    )
                return G
        for key in ("graph6", "g6"):
            if key in graph_data:
                G = _undirected_graph_from_graph_encoding(
                    graph_data[key], sparse=False
                )
                n_raw = _mapping_get_first(
                    graph_data, "n", "num_vertices", "num_nodes", "order"
                )
                if n_raw is not None and _to_int(
                    n_raw, name="number of graph vertices"
                ) != G.number_of_nodes():
                    raise VerificationError(
                        "number of graph vertices does not match the graph encoding"
                    )
                return G
        for key in ("sparse6", "s6"):
            if key in graph_data:
                G = _undirected_graph_from_graph_encoding(
                    graph_data[key], sparse=True
                )
                n_raw = _mapping_get_first(
                    graph_data, "n", "num_vertices", "num_nodes", "order"
                )
                if n_raw is not None and _to_int(
                    n_raw, name="number of graph vertices"
                ) != G.number_of_nodes():
                    raise VerificationError(
                        "number of graph vertices does not match the graph encoding"
                    )
                return G
        for key in ("adjacency_matrix", "adjacency"):
            if key in graph_data:
                G = _undirected_graph_from_adjacency_matrix(graph_data[key])
                n_raw = _mapping_get_first(
                    graph_data, "n", "num_vertices", "num_nodes", "order"
                )
                if n_raw is not None and _to_int(
                    n_raw, name="number of graph vertices"
                ) != G.number_of_nodes():
                    raise VerificationError(
                        "number of graph vertices does not match the adjacency matrix"
                    )
                return G

        edge_key = next(
            (key for key in ("edges", "edge_list") if key in graph_data),
            None,
        )
        if edge_key is None:
            for key in ("matrix", "rows", "array", "A"):
                if key in graph_data:
                    G = _undirected_graph_from_adjacency_matrix(graph_data[key])
                    n_raw = _mapping_get_first(
                        graph_data, "n", "num_vertices", "num_nodes", "order"
                    )
                    if n_raw is not None and _to_int(
                        n_raw, name="number of graph vertices"
                    ) != G.number_of_nodes():
                        raise VerificationError(
                            "number of graph vertices does not match the adjacency matrix"
                        )
                    return G
        if edge_key is not None:
            n_raw = graph_data.get(
                "n",
                graph_data.get(
                    "num_vertices", graph_data.get("num_nodes", graph_data.get("order"))
                ),
            )
            vertices_raw = graph_data.get(
                "vertices",
                graph_data.get(
                    "nodes",
                    graph_data.get("vertex_labels", graph_data.get("labels")),
                ),
            )
            G = nx.Graph()
            label_to_node: dict[Any, int] | None = None
            if vertices_raw is not None:
                labels = []
                for value in _as_nonstring_list(
                    vertices_raw, name="vertices/nodes"
                ):
                    label = _hashable_key(value)
                    try:
                        hash(label)
                    except TypeError as exc:
                        raise VerificationError("graph vertex labels must be hashable") from exc
                    labels.append(label)
                if len(set(labels)) != len(labels):
                    raise VerificationError("graph vertex labels must be distinct")
                if n_raw is not None and _to_int(n_raw, name="number of graph vertices") != len(labels):
                    raise VerificationError("number of graph vertices does not match the vertex label list")
                label_to_node = {label: i for i, label in enumerate(labels)}
                G.add_nodes_from(range(len(labels)))
            elif n_raw is not None:
                n = _to_int(n_raw, name="number of graph vertices")
                if n < 0:
                    raise VerificationError("number of graph vertices must be nonnegative")
                G.add_nodes_from(range(n))
            edges_payload = graph_data[edge_key]
            for edge in _as_nonstring_list(edges_payload, name="edges"):
                endpoints = _undirected_edge_pair(edge)
                if endpoints is None:
                    raise VerificationError("every edge must be a 2-element sequence")
                if label_to_node is not None:
                    u_key = _hashable_key(endpoints[0])
                    v_key = _hashable_key(endpoints[1])
                    try:
                        hash(u_key)
                        hash(v_key)
                    except TypeError as exc:
                        raise VerificationError(
                            "graph vertex labels must be hashable"
                        ) from exc
                    if u_key not in label_to_node or v_key not in label_to_node:
                        raise VerificationError("edge endpoint is outside the vertex label list")
                    u = label_to_node[u_key]
                    v = label_to_node[v_key]
                elif n_raw is not None:
                    n = G.number_of_nodes()
                    u = _to_int(endpoints[0], name="edge endpoint")
                    v = _to_int(endpoints[1], name="edge endpoint")
                    if not (0 <= u < n and 0 <= v < n):
                        raise VerificationError("edge endpoint is outside 0..n-1")
                else:
                    u = _hashable_key(endpoints[0])
                    v = _hashable_key(endpoints[1])
                    try:
                        hash(u)
                        hash(v)
                    except TypeError as exc:
                        raise VerificationError("graph vertex labels must be hashable") from exc
                if u == v:
                    raise VerificationError("graphs must be simple: self-loops are not allowed")
                if G.has_edge(u, v):
                    raise VerificationError("graphs must be simple: duplicate/parallel edges are not allowed")
                G.add_edge(u, v)
            return _relabel_graph_to_consecutive_integers(G)
        raise VerificationError("unsupported graph dict format")

    if isinstance(graph_data, nx.Graph):
        _check_networkx_graph_is_simple_undirected(graph_data)
        G = nx.Graph(graph_data)
    elif isinstance(graph_data, (str, bytes, bytearray)):
        if isinstance(graph_data, str):
            prefix = graph_data.strip()
        else:
            try:
                prefix = bytes(graph_data).strip().decode("ascii")
            except UnicodeDecodeError as exc:
                raise VerificationError(
                    "graph6/sparse6 bytes must be ASCII"
                ) from exc
        sparse = prefix.startswith(":") or prefix.startswith(">>sparse6<<")
        G = _undirected_graph_from_graph_encoding(graph_data, sparse=sparse)
    elif hasattr(graph_data, "tolist"):
        try:
            return _parse_graph(graph_data.tolist())
        except VerificationError:
            raise
        except Exception as exc:
            raise VerificationError(
                "could not convert graph witness to nested lists"
            ) from exc
    elif isinstance(graph_data, Iterable) and not isinstance(
        graph_data, (str, bytes, bytearray, Mapping)
    ):
        rows = _as_nonstring_list(graph_data, name="graph witness")
        normalized_rows = []
        for row in rows:
            if hasattr(row, "tolist"):
                try:
                    row = row.tolist()
                except Exception as exc:
                    raise VerificationError(
                        "could not convert graph row to a list"
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
                        "could not iterate over graph row"
                    ) from exc
            normalized_rows.append(row)
        rows = normalized_rows
        if not rows:
            raise VerificationError("unsupported empty graph sequence")
        matrix_values: list[list[int]] | None = None
        try:
            parsed = _parse_int_matrix(rows, name="adjacency matrix")
        except VerificationError:
            parsed = None
        if parsed is not None and _is_square_matrix(parsed) and all(
            v in (0, 1) for row in parsed for v in row
        ):
            matrix_values = parsed
        if matrix_values is not None:
            G = _undirected_graph_from_adjacency_matrix(matrix_values)
        else:
            edge_pairs = [_undirected_edge_pair(edge) for edge in rows]
            if any(pair is None for pair in edge_pairs):
                raise VerificationError("unsupported graph sequence format")
            G = nx.Graph()
            for pair in edge_pairs:
                if pair is None:  # narrowed by the check above
                    raise VerificationError("unsupported graph sequence format")
                u, v = (_hashable_key(pair[0]), _hashable_key(pair[1]))
                try:
                    hash(u)
                    hash(v)
                except TypeError as exc:
                    raise VerificationError("graph vertex labels must be hashable") from exc
                if u == v:
                    raise VerificationError("graphs must be simple: self-loops are not allowed")
                if G.has_edge(u, v):
                    raise VerificationError("graphs must be simple: duplicate/parallel edges are not allowed")
                G.add_edge(u, v)
    else:
        raise VerificationError("unsupported graph format")

    if any(u == v for u, v in G.edges()):
        raise VerificationError("graphs must be simple")
    return _relabel_graph_to_consecutive_integers(G)



def _is_square(n: int) -> bool:
    """Return True iff ``n`` is a nonnegative perfect square."""
    if n < 0:
        return False
    r = int(sp.integer_nthroot(n, 2)[0])
    return r * r == n


def _ln_rational_interval(num: int, den: int, terms: int) -> tuple[Fraction, Fraction]:
    """Return a rigorous rational interval containing log(num/den).

    The implementation uses
        log(x) = 2 * sum_{j>=0} z^(2j+1)/(2j+1),  z=(x-1)/(x+1),
    and a geometric upper bound for the tail.  All endpoint arithmetic is exact.
    """
    if num <= 0 or den <= 0:
        raise VerificationError("logarithm interval endpoints must be positive")
    if terms <= 0:
        raise VerificationError("number of logarithm-series terms must be positive")
    if num == den:
        return Fraction(0), Fraction(0)

    z = Fraction(num - den, num + den)
    z2 = z * z
    term = z
    partial = Fraction(0)
    for j in range(terms):
        partial += term / (2 * j + 1)
        term *= z2

    tail_bound = 2 * abs(term) / (2 * terms + 1) / (1 - z2)
    approx = 2 * partial
    if z > 0:
        return approx, approx + tail_bound
    return approx - tail_bound, approx


def _ln_integer_interval(n: int, terms: int) -> tuple[Fraction, Fraction]:
    """Return a rigorous rational interval containing log(n)."""
    if n <= 0:
        raise VerificationError("logarithm needs a positive integer")
    if n == 1:
        return Fraction(0), Fraction(0)
    k = n.bit_length() - 1
    pow2 = 1 << k
    ln2_lo, ln2_hi = _ln_rational_interval(2, 1, terms)
    red_lo, red_hi = _ln_rational_interval(n, pow2, terms)
    return k * ln2_lo + red_lo, k * ln2_hi + red_hi


def _compare_exact_integer_powers(p: int, q: int, n: int, *, max_total_bits: int = 1_000_000) -> int | None:
    """Compare q**n and p**(n+1) directly when the integers are modest.

    Returns None rather than constructing very large integers.
    """
    estimated_bits = n * max(1, q.bit_length()) + (n + 1) * max(1, p.bit_length())
    if estimated_bits > max_total_bits:
        return None
    lhs = pow(q, n)
    rhs = pow(p, n + 1)
    return (lhs > rhs) - (lhs < rhs)


def _compare_q_pow_n_vs_p_pow_np1(p: int, q: int, n: int) -> int:
    """Return sign(q**n - p**(n+1)) exactly.

    This avoids floating-point logarithms.  It first tries direct integer-power comparison;
    for huge exponents it proves the sign of n*log(q/p) - log(p) by exact rational
    interval arithmetic, refining until the sign is separated.
    """
    if p <= 1 or q <= 1 or n <= 0:
        raise VerificationError("power comparison expects p,q > 1 and n > 0")

    direct = _compare_exact_integer_powers(p, q, n)
    if direct is not None:
        return direct

    if p == q:
        return -1

    # Defensive equality test for non-prime inputs: q^n = p^(n+1) iff p=a^n and q=a^(n+1).
    if q % p == 0:
        root, exact = sp.integer_nthroot(p, n)
        if exact and q == p * int(root):
            return 0

    for terms in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
        gap_lo, gap_hi = _ln_rational_interval(q, p, terms)
        p_lo, p_hi = _ln_integer_interval(p, terms)
        diff_lo = n * gap_lo - p_hi
        diff_hi = n * gap_hi - p_lo
        if diff_lo > 0:
            return 1
        if diff_hi < 0:
            return -1
        if diff_lo == 0 and diff_hi == 0:
            return 0
    raise VerificationError("could not determine the exact logarithmic comparison")


__all__ = [
    "ValidationResult",
    "VerificationError",
    "_as_nonstring_list",
    "_compare_q_pow_n_vs_p_pow_np1",
    "_expr_is_zero",
    "_hashable_key",
    "_is_nonstring_sequence",
    "_is_square",
    "_is_square_matrix",
    "_mapping_get_first",
    "_parse_graph",
    "_parse_int_matrix",
    "_reject_float_literal_string",
    "_sympy_locals",
    "_to_exact_sympy",
    "_to_fraction",
    "_to_int",
]
