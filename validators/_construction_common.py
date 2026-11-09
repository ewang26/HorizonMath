"""Shared parsing and result types for exact construction witnesses."""

from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Sequence
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
    if not re.fullmatch(r"[A-Za-z0-9_+*/^(),\s-]+", value):
        raise VerificationError(f"{name}: invalid characters in exact expression")
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


def _is_square(n: int) -> bool:
    """Return True iff ``n`` is a nonnegative perfect square."""
    if n < 0:
        return False
    r = int(sp.integer_nthroot(n, 2)[0])
    return r * r == n
