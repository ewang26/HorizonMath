from mpmath import mp

mp.dps = 110


def compute():
    """
    Resultant Res_x(T_30, P_20) of the Chebyshev polynomial T_30(x) and
    the Legendre polynomial P_20(x).

    No general closed-form formula is known for Res(T_n, P_m) when T_n is
    Chebyshev (first kind) and P_m is Legendre. This is in contrast to
    within-family resultants like Res(T_n, T_m) and Res(T_n, U_m), which
    have known closed forms (Gishe-Ismail, Canadian Math Bulletin 2008).

    Uses the standard resultant product formula:
      Res(P, Q) = lc(P)^{deg Q} * prod_{P(alpha)=0} Q(alpha)
    applied as:
      Res(T_n, P_m) = lc(T_n)^m * prod_{k=1}^{n} P_m(x_k)
    where x_k = cos((2k-1)*pi/(2n)) are the roots of T_n,
    and lc(T_n) = 2^{n-1} for n >= 1.

    Verification: this formula can be cross-checked against sympy's exact
    resultant(chebyshevt_poly(n, x), legendre_poly(m, x), x) for small or
    large (n, m). For the small case Res(T_3, P_2) = -25/8, both methods agree.
    """
    n = 30
    m = 20

    # Leading coefficient of T_n: 2^{n-1}
    lc_Tn = mp.power(2, n - 1)

    # Roots of T_n: x_k = cos((2k-1)*pi/(2n)) for k = 1, ..., n
    # Res = lc(T_n)^m * prod_{k=1}^{n} P_m(x_k)
    prod = mp.mpf(1)
    two_n = 2 * n
    for k in range(1, n + 1):
        # x_k = cos((2k-1)*pi/(2n))
        x_k = mp.cospi(mp.mpf(2 * k - 1) / two_n)
        prod *= mp.legendre(m, x_k)

    return lc_Tn ** m * prod


if __name__ == "__main__":
    print(str(compute()))
