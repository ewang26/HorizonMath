from mpmath import mp

mp.dps = 110


def _conv_trunc(a, b, n):
    res = [mp.mpf("0")] * n
    na = min(len(a), n)
    nb = min(len(b), n)
    for i in range(na):
        ai = a[i]
        if not ai:
            continue
        m = min(nb, n - i)
        for j in range(m):
            res[i + j] += ai * b[j]
    return res


def _tail_asymptotic(X0, N=300):
    # Asymptotic series coefficients for K0(x):
    # K0(x) ~ sqrt(pi/(2x)) * exp(-x) * sum_{k>=0} c_k / x^k,  x -> +inf
    # with recurrence (nu=0, mu=0): c_0=1,
    # c_k = c_{k-1} * (-(2k-1)^2) / (8k)
    c = [mp.mpf("0")] * N
    c[0] = mp.mpf("1")
    for k in range(1, N):
        c[k] = c[k - 1] * (-(2 * k - 1) ** 2) / (mp.mpf(8) * k)

    # I0(5x) asymptotic has series sum_{k>=0} (-1)^k c_k / (5x)^k
    p = [mp.mpf("0")] * N
    inv5 = mp.mpf(1) / 5
    inv5pow = mp.mpf(1)
    for k in range(N):
        pk = c[k] * inv5pow
        if k & 1:
            pk = -pk
        p[k] = pk
        inv5pow *= inv5

    # q = (sum c_k/x^k)^5 truncated
    q = [mp.mpf("0")] * N
    q[0] = mp.mpf("1")
    for _ in range(5):
        q = _conv_trunc(q, c, N)

    # r = p*q truncated
    r = _conv_trunc(p, q, N)

    # Prefactor for x*I0(5x)*K0(x)^5 after exponential cancellation:
    # x*I0(5x)*K0(x)^5 ~ c0 * sum_{k>=0} r_k / x^{2+k}
    c0 = mp.pi**2 / mp.sqrt(320)

    invX = mp.mpf(1) / X0
    invXpow = invX  # X0^-(k+1)
    s = mp.mpf("0")
    for k in range(N):
        s += r[k] * invXpow / (k + 1)
        invXpow *= invX

    return c0 * s


def compute():
    with mp.workdps(350):
        X0 = mp.mpf(200)

        def integrand(x):
            k0 = mp.besselk(0, x)
            return x * mp.besseli(0, 5 * x) * (k0**5)

        main = mp.quad(
            integrand,
            [mp.mpf("0"), mp.mpf("0.5"), 1, 2, 5, 10, 20, 40, 80, 120, 160, X0],
        )
        tail = _tail_asymptotic(X0, N=300)
        res = main + tail

    return +res


if __name__ == "__main__":
    print(str(compute()))