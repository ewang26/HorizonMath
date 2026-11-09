"""Exact verifiers for number theory and polynomial witnesses."""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence
import sympy as sp
from sympy.ntheory.primetest import _lucas_sequence
from sympy.polys.numberfields import to_number_field
from ._construction_common import (
    ValidationResult,
    VerificationError,
    _as_nonstring_list,
    _is_square,
    _mapping_get_first,
    _to_exact_sympy,
    _to_int,
)


UINT64_MAX = (1 << 64) - 1


MR64_BASES = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)


def _miller_rabin_u64(n: int) -> bool:
    """Deterministic Miller-Rabin for 64-bit integers."""
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in MR64_BASES:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def verify_pratt_certificate(cert: Mapping[str, Any]) -> bool:
    """Verify a recursive Pratt certificate.

    Base case: {"n": 2}. For odd primes p, the certificate must include
    - 'n': p,
    - 'a': a Pratt witness,
    - 'factorization': a list of prime-power factors of p-1.

    Malformed certificates return False instead of leaking parser exceptions.
    """
    if not isinstance(cert, Mapping):
        return False
    try:
        p = _to_int(cert.get("n"), name="certificate.n")
    except VerificationError:
        return False
    if p == 2:
        return True
    if p < 2 or p % 2 == 0:
        return False
    try:
        a = _to_int(cert.get("a"), name="certificate.a")
    except VerificationError:
        return False
    try:
        factors = _as_nonstring_list(
            cert.get("factorization"), name="certificate.factorization"
        )
    except VerificationError:
        return False
    if not factors:
        return False

    factor_product = 1
    prime_divisors: list[int] = []
    for item in factors:
        if not isinstance(item, Mapping):
            return False
        try:
            q = _to_int(item.get("prime"), name="factorization.prime")
            e = _to_int(item.get("exp"), name="factorization.exp")
        except VerificationError:
            return False
        subcert = item.get("certificate")
        if q < 2 or e <= 0:
            return False
        if subcert is None or not verify_pratt_certificate(subcert):
            return False
        try:
            if _to_int(subcert.get("n"), name="subcertificate.n") != q:
                return False
        except VerificationError:
            return False
        factor_product *= q ** e
        prime_divisors.append(q)

    if factor_product != p - 1:
        return False
    if pow(a, p - 1, p) != 1:
        return False
    for q in set(prime_divisors):
        if math.gcd(pow(a, (p - 1) // q, p) - 1, p) != 1:
            return False
    return True


def _require_exact_prime(
    n: int,
    *,
    certificate: Mapping[str, Any] | None,
    allow_probable_prime: bool,
    label: str,
) -> None:
    """Raise VerificationError unless primality of n is established exactly.

    Exact modes:
    - n <= 2**64-1: deterministic Miller-Rabin
    - certificate is provided: recursive Pratt verification

    Optional practical mode:
    - allow_probable_prime=True permits SymPy's primality test for larger n.
      This is useful operationally, but not proof-producing.
    """
    if certificate is not None:
        if not verify_pratt_certificate(certificate):
            raise VerificationError(f"invalid Pratt certificate for {label}")
        if _to_int(certificate.get("n"), name=f"{label}_certificate.n") != n:
            raise VerificationError(f"Pratt certificate for {label} proves a different integer")
        return
    if n <= UINT64_MAX:
        if not _miller_rabin_u64(n):
            raise VerificationError(f"{label} is not prime")
        return
    if allow_probable_prime:
        if not sp.isprime(n):
            raise VerificationError(f"{label} is not prime")
        return
    raise VerificationError(
        f"exact primality of {label} > 2^64 requires a certificate; either add one or "
        "enable allow_probable_prime=True for a practical (not proof-producing) fallback"
    )


def _extract_compositeness_data(candidate: Any) -> tuple[int, int | None, list[int] | None]:
    """Return ``(n, divisor, factorization_list)`` from any supported composite witness format."""
    if isinstance(candidate, Mapping):
        n = _to_int(candidate.get("n"), name="n")
        divisor = _mapping_get_first(candidate, "divisor", "factor", "proper_divisor")
        divisor_i = None if divisor is None else _to_int(divisor, name="divisor")
        if "factorization" in candidate:
            factors = candidate["factorization"]
        elif "factors" in candidate:
            factors = candidate["factors"]
        else:
            factors = None
        if factors is not None:
            if isinstance(factors, Mapping):
                expanded: list[int] = []
                for p_raw, e_raw in factors.items():
                    p = _to_int(p_raw, name="factorization prime")
                    e = _to_int(e_raw, name="factorization exponent")
                    if e <= 0:
                        raise VerificationError("factorization exponents must be positive")
                    expanded.extend([p] * e)
                factors_list = expanded
            else:
                factors_list = [
                    _to_int(x, name="factor")
                    for x in _as_nonstring_list(
                        factors, name="factorization/factors"
                    )
                ]
        else:
            factors_list = None
        return n, divisor_i, factors_list
    values = _as_nonstring_list(candidate, name="composite witness")
    if len(values) != 2:
        raise VerificationError("composite witness tuple must have the form (n, divisor)")
    return _to_int(values[0], name="n"), _to_int(values[1], name="divisor"), None


def _prove_composite(n: int, divisor: int | None, factors: list[int] | None) -> None:
    if n <= 1:
        raise VerificationError("n must be > 1")
    if divisor is not None:
        if not (1 < divisor < n):
            raise VerificationError("divisor must satisfy 1 < divisor < n")
        if n % divisor != 0:
            raise VerificationError("claimed divisor does not divide n")
        return
    if factors is not None:
        if not factors:
            raise VerificationError("factorization cannot be empty")
        prod = 1
        for f in factors:
            if not (1 < f < n):
                raise VerificationError("every factor must satisfy 1 < f < n")
            prod *= f
        if prod != n:
            raise VerificationError("factorization does not multiply to n")
        return
    if n <= UINT64_MAX and not _miller_rabin_u64(n):
        return
    raise VerificationError(
        "compositeness was not proved exactly; provide a divisor or factorization for full exactness"
    )


def _jacobi(a: int, n: int) -> int:
    """Compute the Jacobi symbol ``(a|n)`` exactly for integer ``a`` and positive odd ``n``."""
    if n <= 0 or n % 2 == 0:
        raise VerificationError("Jacobi symbol requires positive odd denominator")
    a %= n
    result = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    return result if n == 1 else 0


def is_fermat_prp(n: int, base: int) -> bool:
    if n <= 1 or math.gcd(n, base) != 1:
        return False
    return pow(base, n - 1, n) == 1


def is_strong_prp(n: int, base: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    if math.gcd(n, base) != 1:
        return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    x = pow(base, d, n)
    if x in (1, n - 1):
        return True
    for _ in range(s - 1):
        x = (x * x) % n
        if x == n - 1:
            return True
    return False


@dataclass(frozen=True)
class LucasParams:
    D: int
    P: int
    Q: int


def method_a_star_params(n: int) -> LucasParams | None:
    """Return Selfridge Method A* parameters for the standard BPSW/BFW test.

    Sequence of discriminants: 5, -7, 9, -11, 13, -15, ...
    Special case: when ``D == 5``, use ``(P, Q) = (5, 5)`` instead of ``(1, -1)``.

    Returns ``None`` when Method A* itself detects compositeness before reaching ``Jacobi(D, n) = -1``.
    In particular, odd perfect squares are rejected here rather than raising.
    """
    if n % 2 == 0 or n <= 1:
        raise VerificationError("Method A* expects an odd integer n > 1")
    if _is_square(n):
        return None
    k = 0
    while True:
        abs_d = 2 * k + 5
        D = abs_d if k % 2 == 0 else -abs_d
        j = _jacobi(D, n)
        if j == -1:
            if D == 5:
                return LucasParams(D=5, P=5, Q=5)
            return LucasParams(D=D, P=1, Q=(1 - D) // 4)
        if j == 0:
            g = math.gcd(abs(D), n)
            if g not in (1, n):
                return None
        k += 1
        if k > 10_000:  # pragma: no cover - pathological guard
            raise VerificationError("failed to find Method A* parameters")


def _quadratic_x_power_mod(n: int, P: int, Q: int, exponent: int) -> tuple[int, int]:
    """Return coefficients ``(a, b)`` of ``x**exponent == a*x+b`` modulo
    ``(n, x**2-P*x+Q)``.
    """

    def multiply(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
        a, b = left
        c, d = right
        return (
            (a * c * P + a * d + b * c) % n,
            (b * d - a * c * Q) % n,
        )

    result = (0, 1)
    base = (1, 0)
    power = exponent
    while power:
        if power & 1:
            result = multiply(result, base)
        base = multiply(base, base)
        power >>= 1
    return result


def is_quadratic_frobenius_prp(n: int, P: int, Q: int, D: int | None = None) -> bool:
    """Check the degree-two Frobenius condition when ``Jacobi(D, n) == -1``.

    In this case the complete factorization, Frobenius, and Jacobi steps are
    equivalent to ``x**n == P-x`` modulo ``(n, x**2-P*x+Q)``, provided
    ``gcd(n, 2*Q*D) == 1``.  Merely combining a Fermat and Lucas probable-prime
    test is weaker and is not sufficient.
    """
    if n <= 1 or n % 2 == 0:
        return False
    discriminant = P * P - 4 * Q
    if D is not None and D != discriminant:
        raise VerificationError("quadratic discriminant does not match P^2 - 4Q")
    D = discriminant
    if math.gcd(n, 2 * Q * D) != 1 or _jacobi(D, n) != -1:
        return False
    return _quadratic_x_power_mod(n, P, Q, n) == ((-1) % n, P % n)


def is_strong_lucas_prp(n: int, P: int, Q: int, D: int | None = None) -> bool:
    if n % 2 == 0 or n <= 1:
        return False
    if D is None:
        D = P * P - 4 * Q
    if math.gcd(n, 2 * Q * D) != 1:
        return False
    eps = _jacobi(D, n)
    delta = n - eps
    s = 0
    d = delta
    while d % 2 == 0:
        d //= 2
        s += 1
    U, V, Qk = _lucas_sequence(n, P, Q, d)
    if U % n == 0 or V % n == 0:
        return True
    for _ in range(1, s):
        V = (V * V - 2 * Qk) % n
        Qk = (Qk * Qk) % n
        if V % n == 0:
            return True
    return False


def is_standard_bpsw_pseudoprime(n: int) -> bool:
    if n <= 1 or n % 2 == 0 or _is_square(n):
        return False
    params = method_a_star_params(n)
    if params is None:
        return False
    return is_strong_prp(n, 2) and is_strong_lucas_prp(n, params.P, params.Q, params.D)


def is_bfw_enhanced_pseudoprime(n: int) -> bool:
    if n <= 1 or n % 2 == 0 or _is_square(n):
        return False
    params = method_a_star_params(n)
    if params is None:
        return False
    if not is_strong_prp(n, 2):
        return False
    if not is_strong_lucas_prp(n, params.P, params.Q, params.D):
        return False
    _, Vn1, _ = _lucas_sequence(n, params.P, params.Q, n + 1)
    if Vn1 % n != (2 * params.Q) % n:
        return False
    return pow(params.Q, (n + 1) // 2, n) == (params.Q * _jacobi(params.Q, n)) % n


def fib_mod(k: int, modulus: int) -> int:
    if k < 0:
        raise VerificationError("Fibonacci index must be nonnegative")
    if modulus <= 0:
        raise VerificationError("modulus must be positive")

    def rec(n: int) -> tuple[int, int]:
        if n == 0:
            return (0, 1)
        a, b = rec(n >> 1)
        c = (a * ((2 * b - a) % modulus)) % modulus
        d = (a * a + b * b) % modulus
        if n & 1:
            return (d, (c + d) % modulus)
        return (c, d)

    return rec(k)[0]


def _build_monomial(vars_: Sequence[sp.Symbol], monomial_key: Any) -> sp.Expr:
    if isinstance(monomial_key, str):
        expr = _to_exact_sympy(
            monomial_key,
            name="monomial key",
            extra_locals={v.name: v for v in vars_},
            allow_symbols=True,
        )
        return expr
    if isinstance(monomial_key, Mapping):
        if "exponents" not in monomial_key:
            raise VerificationError("monomial dict keys must contain an 'exponents' field")
        exps = _as_nonstring_list(
            monomial_key["exponents"], name="monomial exponents"
        )
        if len(exps) != len(vars_):
            raise VerificationError("monomial exponent tuple has the wrong dimension")
        expr = sp.Integer(1)
        for var, e_raw in zip(vars_, exps):
            e = _to_int(e_raw, name="monomial exponent")
            if e < 0:
                raise VerificationError("monomial exponents must be nonnegative")
            expr *= var ** e
        return expr
    exps = _as_nonstring_list(monomial_key, name="monomial exponent tuple")
    if len(exps) != len(vars_):
        raise VerificationError("monomial exponent tuple has the wrong dimension")
    expr = sp.Integer(1)
    for var, e_raw in zip(vars_, exps):
        e = _to_int(e_raw, name="monomial exponent")
        if e < 0:
            raise VerificationError("monomial exponents must be nonnegative")
        expr *= var ** e
    return expr


def _parse_polynomial_dict(poly: Mapping[Any, Any], vars_: Sequence[sp.Symbol]) -> sp.Expr:
    expr = sp.Integer(0)
    for mono_key, coeff_raw in poly.items():
        coeff = _to_exact_sympy(coeff_raw)
        expr += coeff * _build_monomial(vars_, mono_key)
    return sp.expand(expr)


def _format_symbol_list(symbols: Iterable[sp.Symbol]) -> str:
    return ", ".join(sorted(str(s) for s in symbols))


def _require_exact_algebraic_number(expr: sp.Expr, *, label: str) -> None:
    """Reject numeric expressions for which exact equality is not decidable here.

    Jacobian witnesses are allowed to use rational, Gaussian-rational, and algebraic
    coefficients/coordinates.  Restricting the certificate language to algebraic
    numbers lets every equality below be decided in an exact number field rather than
    by numerical evaluation or a heuristic symbolic comparison.
    """
    if expr.free_symbols or expr.is_algebraic is not True:
        raise VerificationError(f"{label} must be an exact algebraic number")
    try:
        to_number_field(expr)
    except Exception as exc:  # SymPy exposes several domain-specific exact-conversion errors
        raise VerificationError(f"{label} is not supported as an exact algebraic number") from exc


def _exact_algebraic_is_zero(expr: sp.Expr, *, label: str) -> bool:
    """Decide equality to zero exactly for an algebraic numeric expression."""
    expr = sp.cancel(sp.expand(expr))
    _require_exact_algebraic_number(expr, label=label)
    try:
        algebraic = to_number_field(expr)
    except Exception as exc:  # pragma: no cover - guarded above
        raise VerificationError(f"could not decide exact equality for {label}") from exc
    return all(coeff == 0 for coeff in algebraic.coeffs())


def _ensure_polynomial(expr: sp.Expr, vars_: Sequence[sp.Symbol], *, label: str) -> sp.Expr:
    expr = sp.expand(expr)
    extra_symbols = expr.free_symbols - set(vars_)
    if extra_symbols:
        raise VerificationError(
            f"{label} contains symbols outside the ambient variables: {_format_symbol_list(extra_symbols)}"
        )
    try:
        poly_obj = sp.Poly(expr, *vars_)
    except sp.PolynomialError as exc:
        raise VerificationError(f"{label} must be a polynomial in the ambient variables") from exc
    for index, coeff in enumerate(poly_obj.coeffs(), start=1):
        _require_exact_algebraic_number(coeff, label=f"{label} coefficient {index}")
    return expr


def _parse_polynomial_payload(payload: Any, vars_: Sequence[sp.Symbol]) -> sp.Expr:
    if isinstance(payload, Mapping):
        expr = _parse_polynomial_dict(payload, vars_)
    else:
        expr = _to_exact_sympy(
            payload,
            name="coordinate polynomial",
            extra_locals={v.name: v for v in vars_},
            allow_symbols=True,
        )
    return _ensure_polynomial(expr, vars_, label="coordinate polynomial")


def _parse_complex_point(point: Any, vars_: Sequence[sp.Symbol]) -> list[sp.Expr]:
    coordinates = _as_nonstring_list(point, name="point coordinates")
    if len(coordinates) != len(vars_):
        raise VerificationError("point has the wrong dimension")
    parsed = [_to_exact_sympy(v) for v in coordinates]
    for index, coordinate in enumerate(parsed, start=1):
        _require_exact_algebraic_number(coordinate, label=f"point coordinate {index}")
    return parsed


@dataclass(frozen=True)
class ParsedJacobianWitness:
    vars: tuple[sp.Symbol, ...]
    polynomials: tuple[sp.Expr, ...]
    x: tuple[sp.Expr, ...]
    y: tuple[sp.Expr, ...]
    form: str


def parse_complex_jacobian_witness(candidate: Mapping[str, Any], dimension: int | None = None) -> ParsedJacobianWitness:
    if not isinstance(candidate, Mapping):
        raise VerificationError("Complex Jacobian witnesses must be dictionaries")

    if candidate.get("form") is None:
        if "matrix" in candidate or "A" in candidate:
            form = "druzkowski"
        elif "H" in candidate:
            form = "bcw"
        else:
            form = "general"
    else:
        form = str(candidate.get("form")).lower()
    if form not in {"general", "bcw", "druzkowski"}:
        raise VerificationError(
            "unknown Complex Jacobian form; use 'general', 'bcw', or 'druzkowski'"
        )

    d_raw = dimension if dimension is not None else candidate.get("dimension", candidate.get("n"))
    if d_raw is None:
        d: int | None = None
    else:
        d = _to_int(d_raw, name="dimension")
        if d < 2:
            raise VerificationError("dimension must be at least 2")

    if form == "druzkowski":
        A_payload = _mapping_get_first(candidate, "matrix", "A")
        A = _as_nonstring_list(A_payload, name="Druzkowski matrix A")
        if not A:
            raise VerificationError("Druzkowski witnesses require a square matrix A")
        A_rows = [
            _as_nonstring_list(row, name=f"Druzkowski matrix row {i}")
            for i, row in enumerate(A)
        ]
        if any(len(row) != len(A_rows) for row in A_rows):
            raise VerificationError("Druzkowski matrix A must be square")
        if d is not None and d != len(A_rows):
            raise VerificationError("dimension does not match the size of the Druzkowski matrix")
        d = len(A_rows)
        vars_ = sp.symbols("x1:%d" % (d + 1))
        A_sym = sp.Matrix(
            [[_to_exact_sympy(v) for v in row] for row in A_rows]
        )
        Xvec = sp.Matrix(vars_)
        linear = A_sym * Xvec
        polys = [_ensure_polynomial(sp.expand(vars_[i] + linear[i] ** 3), vars_, label=f"coordinate polynomial {i + 1}") for i in range(d)]
    else:
        poly_payload = candidate.get("polynomials")
        if poly_payload is None:
            poly_payload = candidate.get("coefficients", candidate.get("coeffs"))
        if poly_payload is None and form == "bcw":
            poly_payload = candidate.get("H")
        if poly_payload is None:
            raise VerificationError("Complex Jacobian witnesses need 'polynomials' or an equivalent form-specific payload")
        poly_payload = _as_nonstring_list(
            poly_payload, name="polynomial payload"
        )
        if d is None:
            d = len(poly_payload)
        if len(poly_payload) != d:
            raise VerificationError("number of coordinate polynomials must equal the dimension")
        vars_ = sp.symbols("x1:%d" % (d + 1))
        if form == "bcw":
            H = [_parse_polynomial_payload(p, vars_) for p in poly_payload]
            polys = [_ensure_polynomial(sp.expand(vars_[i] + H[i]), vars_, label=f"coordinate polynomial {i + 1}") for i in range(d)]
        else:
            polys = [_ensure_polynomial(_parse_polynomial_payload(p, vars_), vars_, label=f"coordinate polynomial {i + 1}") for i, p in enumerate(poly_payload)]

    points_payload = candidate.get("points")
    if points_payload is not None:
        points = _as_nonstring_list(points_payload, name="points")
        if len(points) != 2:
            raise VerificationError("'points' must be a length-2 sequence [x, y]")
        points_x, points_y = points
    else:
        points_x = points_y = None

    x_raw = candidate.get("x", candidate.get("x1", candidate.get("point1", points_x)))
    y_raw = candidate.get("y", candidate.get("x2", candidate.get("point2", points_y)))
    if x_raw is None or y_raw is None:
        raise VerificationError("Complex Jacobian witnesses require two exact witness points x and y")
    x = _parse_complex_point(x_raw, vars_)
    y = _parse_complex_point(y_raw, vars_)
    return ParsedJacobianWitness(
        vars=tuple(vars_), polynomials=tuple(polys), x=tuple(x), y=tuple(y), form=form
    )


def _expand_euler_lhs(lhs_spec: Any) -> list[int]:
    if isinstance(lhs_spec, Mapping):
        lhs_vals: list[int] = []
        for val_raw, mult_raw in lhs_spec.items():
            val = _to_int(val_raw, name="lhs value")
            mult = _to_int(mult_raw, name="lhs multiplicity")
            if mult <= 0:
                raise VerificationError("lhs multiplicities must be positive")
            lhs_vals.extend([val] * mult)
        return lhs_vals
    return [
        _to_int(v, name="a_i")
        for v in _as_nonstring_list(
            lhs_spec, name="Euler left-hand side"
        )
    ]


def verify_euler_sum_of_powers(candidate: Any, *, k: int) -> ValidationResult:
    """Verify a witness for the fixed-k Euler/Lander-Parkin-Selfridge special case.

    Accepted formats:
    - sequence/list/tuple of length k: [a1, ..., a_{k-1}, b]
    - ([a1, ..., a_{k-1}], b)
    - {"a": [...], "b": b}
    - {"lhs": [...], "b": b}
    - {"lhs": {... multiplicity dict ...}, "b": b}
    """
    if k < 3:
        raise VerificationError("k must be at least 3")
    if isinstance(candidate, Mapping):
        if "a" in candidate:
            lhs_vals = _expand_euler_lhs(candidate["a"])
        elif "lhs" in candidate:
            lhs_vals = _expand_euler_lhs(candidate["lhs"])
        else:
            raise VerificationError("Euler witness dict must contain either 'a' or 'lhs'")
        b = _to_int(candidate.get("b"), name="b")
        values = lhs_vals + [b]
    else:
        values_raw = _as_nonstring_list(candidate, name="Euler witness")
        if (
            len(values_raw) == 2
            and isinstance(values_raw[0], Iterable)
            and not isinstance(values_raw[0], (str, bytes, bytearray))
        ):
            lhs_vals = _expand_euler_lhs(values_raw[0])
            b = _to_int(values_raw[1], name="b")
            values = lhs_vals + [b]
        else:
            values = [_to_int(v, name="value") for v in values_raw]
    if len(values) != k:
        raise VerificationError(f"expected exactly {k} integers: k-1 values on the left and one on the right")
    if any(v <= 0 for v in values):
        raise VerificationError("all Euler witness integers must be positive")
    *lhs_vals, b = values
    lhs_sum = sum(pow(a, k) for a in lhs_vals)
    rhs = pow(b, k)
    if lhs_sum == rhs:
        return ValidationResult(True, f"Euler sum of powers (k={k})", "valid counterexample witness", {"a": lhs_vals, "b": b})
    return ValidationResult(False, f"Euler sum of powers (k={k})", "the supplied tuple does not satisfy the Diophantine equation")


def verify_bpsw_standard(candidate: Any) -> ValidationResult:
    n, divisor, factors = _extract_compositeness_data(candidate)
    _prove_composite(n, divisor, factors)
    ok = is_standard_bpsw_pseudoprime(n)
    return ValidationResult(
        ok,
        "Baillie-PSW standard",
        "valid counterexample witness" if ok else "n does not pass the frozen standard BPSW test",
        {"n": n},
    )


def verify_weak_selfridge_fibonacci(candidate: Any) -> ValidationResult:
    n, divisor, factors = _extract_compositeness_data(candidate)
    _prove_composite(n, divisor, factors)
    ok = (
        n % 5 in (2, 3)
        and is_fermat_prp(n, 2)
        and fib_mod(n + 1, n) == 0
    )
    return ValidationResult(
        ok,
        "Weak Selfridge/Fibonacci",
        "valid counterexample witness" if ok else "n does not satisfy the base-2 Fermat + Fibonacci conditions",
        {"n": n},
    )


def verify_grantham_challenge(candidate: Any) -> ValidationResult:
    """Verify a counterexample to Grantham's quadratic Frobenius challenge.

    The fixed polynomial is ``x^2 + 5x + 5``, represented as ``x^2-Px+Q``
    with ``(P, Q, D) = (-5, 5, 5)``. The witness is a compositeness proof
    together with ``n``.
    """
    n, divisor, factors = _extract_compositeness_data(candidate)
    _prove_composite(n, divisor, factors)
    P, Q, D = -5, 5, 5
    ok = is_quadratic_frobenius_prp(n, P, Q, D)
    return ValidationResult(
        ok,
        "Grantham challenge",
        "valid counterexample witness" if ok else "n does not pass the quadratic Frobenius conditions",
        {"n": n},
    )


def verify_bfw_enhanced(candidate: Any) -> ValidationResult:
    n, divisor, factors = _extract_compositeness_data(candidate)
    _prove_composite(n, divisor, factors)
    ok = is_bfw_enhanced_pseudoprime(n)
    return ValidationResult(
        ok,
        "Baillie-Fiori-Wagstaff enhanced",
        "valid counterexample witness" if ok else "n does not pass the frozen BFW enhanced test",
        {"n": n},
    )


def verify_wall_sun_sun(
    candidate: Any,
    *,
    allow_probable_prime: bool = False,
) -> ValidationResult:
    if isinstance(candidate, Mapping):
        p = _to_int(candidate.get("p", candidate.get("n")), name="p")
        cert = _mapping_get_first(
            candidate, "certificate", "primality_certificate"
        )
    else:
        p = _to_int(candidate, name="p")
        cert = None
    if p in (2, 5):
        return ValidationResult(False, "Wall-Sun-Sun", "2 and 5 are excluded from the standard Wall-Sun-Sun definition")
    _require_exact_prime(p, certificate=cert, allow_probable_prime=allow_probable_prime, label="p")
    eps = _jacobi(5, p)
    modulus = p * p
    ok = fib_mod(p - eps, modulus) == 0
    return ValidationResult(
        ok,
        "Wall-Sun-Sun",
        "valid counterexample witness" if ok else "p is prime but not Wall-Sun-Sun",
        {"p": p},
    )


def verify_complex_jacobian(candidate: Mapping[str, Any], *, dimension: int | None = 2) -> ValidationResult:
    """Verify a complex Jacobian counterexample, defaulting to the open 2D case.

    The benchmark calls this function without keyword arguments, so benchmark
    submissions are fixed at dimension 2.  An explicit dimension remains available for
    checking already-known higher-dimensional examples and for library compatibility.
    Any nonzero constant determinant is accepted: scaling one target coordinate by its
    reciprocal gives determinant 1 and preserves a collision between witness points.
    """
    requested_dimension = 2 if dimension is None else _to_int(dimension, name="dimension")
    candidate_dimension = candidate.get("dimension", candidate.get("n")) if isinstance(candidate, Mapping) else None
    if candidate_dimension is not None and _to_int(candidate_dimension, name="dimension") != requested_dimension:
        raise VerificationError("candidate dimension does not match the requested dimension")

    witness = parse_complex_jacobian_witness(candidate, dimension=requested_dimension)
    vars_ = witness.vars
    polys = witness.polynomials
    conjecture_name = f"Complex Jacobian (dimension {len(vars_)})"

    if len(polys) != len(vars_):
        raise VerificationError("the map must have the same number of coordinates as the ambient dimension")

    if witness.form == "bcw":
        # Check the advertised shape F = X + H with each H_i homogeneous cubic.
        for i, poly in enumerate(polys):
            H_i = sp.expand(poly - vars_[i])
            if H_i == 0:
                continue
            poly_obj = sp.Poly(H_i, *vars_)
            if any(sum(mon) != 3 for mon, _ in poly_obj.terms()):
                raise VerificationError("BCW form requires each H_i to be homogeneous of degree 3")
    if witness.form == "druzkowski":
        # No additional shape check is necessary: the parser already built F_i = x_i + (A_i x)^3.
        pass

    J = sp.Matrix(polys).jacobian(vars_)
    detJ = sp.expand(J.det(method="berkowitz"))
    det_poly = sp.Poly(detJ, *vars_)
    if det_poly.is_zero or det_poly.total_degree() != 0:
        return ValidationResult(False, conjecture_name, "the Jacobian determinant is not a nonzero constant")
    det_constant = det_poly.coeff_monomial((0,) * len(vars_))
    if _exact_algebraic_is_zero(det_constant, label="Jacobian determinant"):
        return ValidationResult(False, conjecture_name, "the Jacobian determinant is not a nonzero constant")

    if all(
        _exact_algebraic_is_zero(a - b, label=f"point-coordinate difference {index}")
        for index, (a, b) in enumerate(zip(witness.x, witness.y), start=1)
    ):
        return ValidationResult(False, conjecture_name, "the two witness points are identical")

    subs_x = dict(zip(vars_, witness.x))
    subs_y = dict(zip(vars_, witness.y))
    Fx = [sp.expand(poly.subs(subs_x)) for poly in polys]
    Fy = [sp.expand(poly.subs(subs_y)) for poly in polys]
    if all(
        _exact_algebraic_is_zero(a - b, label=f"image-coordinate difference {index}")
        for index, (a, b) in enumerate(zip(Fx, Fy), start=1)
    ):
        return ValidationResult(
            True,
            conjecture_name,
            "valid counterexample witness",
            {"dimension": len(vars_), "form": witness.form, "jacobian_determinant": str(det_constant)},
        )
    return ValidationResult(False, conjecture_name, "the supplied points do not collide under the map")
