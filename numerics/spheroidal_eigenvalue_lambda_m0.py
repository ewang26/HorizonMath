from mpmath import mp


def _x2_coeffs_legendre(l):
    """Coefficients for x^2 P_l = a P_{l+2} + b P_l + g P_{l-2}."""
    l = mp.mpf(l)
    a = (l + 1) * (l + 2) / ((2 * l + 1) * (2 * l + 3))
    g = l * (l - 1) / ((2 * l + 1) * (2 * l - 1)) if l >= 2 else mp.mpf("0")
    b = (2 * l * l + 2 * l - 1) / ((2 * l - 1) * (2 * l + 3))
    if l == 0:
        b = mp.mpf(1) / 3
    return a, b, g


def compute(n, c, N=160, dps=220):
    """
    Compute lambda_n(c) for the m=0 angular prolate spheroidal eigenvalue problem:

        -d/dx((1-x^2)y') + c^2 x^2 y = lambda y,  y bounded at +/-1

    Uses a Legendre-expansion tridiagonal truncation of size N.
    """
    mp.dps = dps
    c = mp.mpf(c)

    parity = n % 2
    k_target = n // 2

    C = mp.matrix(N)
    for k in range(N):
        l = parity + 2 * k
        a, b, g = _x2_coeffs_legendre(l)

        C[k, k] = l * (l + 1) + (c**2) * b
        if k + 1 < N:
            C[k + 1, k] = (c**2) * a
        if k - 1 >= 0:
            C[k - 1, k] = (c**2) * g

    evals = mp.eig(C, left=False, right=False)
    evals = sorted([mp.re(z) for z in evals])
    return evals[k_target]


if __name__ == "__main__":
    tests = [
        (0, "0.5"),
        (1, "0.5"),
        (2, "0.5"),
        (0, "1.0"),
        (1, "1.0"),
        (2, "1.0"),
        (0, "2.0"),
        (1, "2.0"),
        (2, "2.0"),
        (3, "2.0"),
    ]
    for n, c in tests:
        val = compute(n, c)
        print(f"n={n} c={c}: {mp.nstr(val, 80)}")
