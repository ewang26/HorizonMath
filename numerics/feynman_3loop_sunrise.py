from mpmath import mp

mp.dps = 110


def compute():
    """
    3-loop sunrise (banana) integral at threshold s = 16m^2.

    B(c) = int_0^inf r * I_0(c*r) * K_0(r)^4 dr

    This is the position-space Bessel representation of the L=3 loop banana
    Feynman integral with 4 equal-mass propagators. The parameter c = sqrt(s)/m,
    so threshold s = (4m)^2 = 16m^2 corresponds to c = 4.

    No closed form is known at threshold. (By contrast, the on-shell value
    B(1) at s = m^2 has a known closed form proved by Zhou (2018):
    B(1) = Gamma(1/15)*Gamma(2/15)*Gamma(4/15)*Gamma(8/15) / (240*sqrt(5)).
    This known special case can be used to validate the integrand formula by
    setting c=1 and checking against the closed form.)

    At threshold c=4, the exponential factors in I_0 and K_0 cancel exactly,
    so the integrand decays as r^{-3/2} (power law, not exponential).

    Strategy:
    - [0, R]: numerical integration using mpmath Bessel functions
    - [R, inf]: analytical integral of asymptotic expansion
      C * r^{-3/2} * sum_n s_n * r^{-n}

    Asymptotic tail accuracy at R=100: ~exp(-200) ~ 10^{-87}.
    Working at 70 dps, combined accuracy is ~50 digits.
    This is a computationally intensive integral; higher precision would
    require significantly more time due to the power-law tail decay.
    """
    c = mp.mpf(4)
    R = mp.mpf(100)

    # Working precision balances accuracy vs speed.
    # At threshold, Bessel evaluations for r in [30,100] are expensive.
    wdps = 70

    def integrand(t):
        if t == 0:
            return mp.zero
        if t < mp.mpf('1e-15'):
            L = -mp.log(t / 2) - mp.euler
            return t * (mp.one + (c * c * t * t) / 4) * (L ** 4)
        return t * mp.besseli(0, c * t) * mp.besselk(0, t) ** 4

    pts = [mp.mpf(0)]
    for x in [0.5, 1, 2, 4, 8, 16, 30, 50, 75]:
        pts.append(mp.mpf(x))
    pts.append(R)

    with mp.workdps(wdps):
        main = mp.quad(integrand, pts)

        # Asymptotic tail from R to infinity.
        # r * I_0(4r) * K_0(r)^4 ~ C * r^{-3/2} * sum_n s_n * r^{-n}
        # C = pi^{3/2} / (8*sqrt(2))
        #
        # Bessel asymptotic coefficients: a_k = [(2k-1)!!]^2 / (k! * 8^k)
        # I_0(z) ~ e^z/sqrt(2*pi*z) * sum_k a_k/z^k        (positive)
        # K_0(z) ~ sqrt(pi/(2z)) * e^{-z} * sum_k (-1)^k * a_k/z^k
        N = 60
        a = [mp.mpf(0)] * N
        a[0] = mp.one
        for k in range(1, N):
            dbl_fac = mp.one
            for j in range(1, k + 1):
                dbl_fac *= (2 * j - 1)
            a[k] = dbl_fac ** 2 / (mp.fac(k) * mp.power(8, k))

        p_I = [a[k] / mp.power(4, k) for k in range(N)]
        p_K = [(-1) ** k * a[k] for k in range(N)]

        def poly_mul(aa, bb, n):
            result = [mp.zero] * n
            for i in range(min(n, len(aa))):
                for j in range(min(n - i, len(bb))):
                    result[i + j] += aa[i] * bb[j]
            return result

        pk2 = poly_mul(p_K, p_K, N)
        pk4 = poly_mul(pk2, pk2, N)
        s = poly_mul(p_I, pk4, N)

        C = mp.power(mp.pi, mp.mpf('1.5')) / (8 * mp.sqrt(2))

        tail = mp.zero
        for n in range(N):
            tail += s[n] * 2 / ((2 * n + 1) * mp.power(R, (2 * n + 1) / mp.mpf(2)))
        tail *= C

        val = main + tail

    return +val


if __name__ == "__main__":
    print(str(compute()))
