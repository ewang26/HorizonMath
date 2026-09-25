from mpmath import mp

mp.dps = 110

def _i0e_asymp(t, tol=None, max_terms=2000):
    # Asymptotic expansion for I0(t)*exp(-t) for large t:
    # I0(t) ~ exp(t)/sqrt(2*pi*t) * sum_{k>=0} a_k / t^k
    # where a_0=1 and a_k = a_{k-1} * (2k-1)^2 / (8k)
    if tol is None:
        tol = mp.mpf('1e-125')
    twopi = 2 * mp.pi
    invt = 1 / t
    s = mp.mpf(1)
    term = mp.mpf(1)
    k = 0
    while True:
        k += 1
        term *= ((2*k - 1)**2) * invt / (8*k)
        s += term
        if abs(term) < tol:
            break
        if k >= max_terms:
            break
    return s / mp.sqrt(twopi * t)

def compute():
    Tasym = mp.mpf(150)

    def i0e(t):
        if t < Tasym:
            return mp.besseli(0, t) * mp.exp(-t)
        return _i0e_asymp(t)

    def f(t):
        v = i0e(t)
        return v**5

    with mp.extradps(30):
        res = mp.quad(f, [0, 1, 10, 50, Tasym]) + mp.quad(f, [Tasym, mp.inf])
    return +res

if __name__ == "__main__":
    print(str(compute()))