from mpmath import mp

mp.dps = 110


def _poly_mul(a, b, deg):
    res = [mp.mpf("0")] * (deg + 1)
    la = min(len(a), deg + 1)
    lb = len(b)
    for i in range(la):
        ai = a[i]
        if not ai:
            continue
        jmax = min(lb - 1, deg - i)
        for j in range(jmax + 1):
            res[i + j] += ai * b[j]
    return res


def _poly_pow(a, power, deg):
    # binary exponentiation with truncation
    res = [mp.mpf("0")] * (deg + 1)
    res[0] = mp.mpf("1")
    base = (a[: deg + 1]) + [mp.mpf("0")] * max(0, deg + 1 - len(a))
    n = power
    while n > 0:
        if n & 1:
            res = _poly_mul(res, base, deg)
        n >>= 1
        if n:
            base = _poly_mul(base, base, deg)
    return res


def _poly_eval(c, z):
    s = mp.mpf("0")
    for coeff in reversed(c):
        s = s * z + coeff
    return s


def compute():
    # B6(1) = E[sqrt(X1^2+...+X6^2)] for Xi~Unif[0,1]
    # Using: sqrt(x) = (1/(2*sqrt(pi))) * ∫_0^∞ (1 - e^{-t x}) t^{-3/2} dt
    # and E[e^{-t sum Xi^2}] = (∫_0^1 e^{-t x^2} dx)^6
    # leads to 1D integral:
    # B6(1) = (1/sqrt(pi)) * ∫_0^∞ (1 - (sqrt(pi)*erf(u)/(2u))^6)/u^2 du
    # Map u in [0,∞) to t in [0,1): u = tan(pi*t/2)

    sqrtpi = mp.sqrt(mp.pi)

    # Series for g(u) = (1 - (sqrt(pi)*erf(u)/(2u))^6) / u^2 near u=0
    # Let z=u^2. f(z)=sqrt(pi)*erf(u)/(2u)=sum_{k>=0} (-1)^k z^k/(k!(2k+1)).
    # Then g(z)=(1-f(z)^6)/z = - (coeffs of f^6 excluding constant term).
    deg_g = 140
    deg_p = deg_g + 1  # need f^6 up to z^(deg_g+1)
    deg_f = (deg_p + 5) // 6 + 10  # safe margin

    fcoeff = [((-1) ** k) / (mp.factorial(k) * (2 * k + 1)) for k in range(deg_f + 1)]
    p = _poly_pow(fcoeff, 6, deg_p)  # p(z)=f(z)^6, truncated

    # g(z) = (1 - p(z))/z = -(p1 + p2 z + ...)
    gcoeff = [-p[i + 1] for i in range(deg_p)]  # length deg_g+1

    small_u_thresh = mp.mpf("0.2")

    def one_minus_L(u):
        # om(u) = 1 - (sqrt(pi)*erf(u)/(2u))^6
        f = sqrtpi * mp.erf(u) / (2 * u)
        # stable for f near 1:
        return -mp.expm1(6 * mp.log(f))

    def integrand_t(t):
        # u = tan(pi*t/2), I = ∫_0^1 g(u) du/dt dt
        # with g(u) = om(u)/u^2 and du/dt = (pi/2) * (1+u^2)
        # => integrand = (pi/2) * (om + om/u^2) = (pi/2) * (om + g)
        if t == 0:
            return mp.pi  # limit
        if t == 1:
            return mp.pi / 2  # limit

        theta = (mp.pi / 2) * t
        u = mp.tan(theta)

        if u == 0:
            return mp.pi

        au = abs(u)
        if au < small_u_thresh:
            z = u * u
            g = _poly_eval(gcoeff, z)  # g = om/u^2
            om = g * z
        else:
            om = one_minus_L(u)
            g = om / (u * u)

        return (mp.pi / 2) * (om + g)

    # Integrate on [0,1] with some manual splitting
    I = mp.quad(integrand_t, [mp.mpf("0"), mp.mpf("0.5"), mp.mpf("0.9"), mp.mpf("0.99"), mp.mpf("1")])
    return I / sqrtpi


if __name__ == "__main__":
    print(str(compute()))