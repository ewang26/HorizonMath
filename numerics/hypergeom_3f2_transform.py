from mpmath import mp

mp.dps = 110

def hyper3f2_half_series(z, tol=None, max_terms=200000):
    if tol is None:
        tol = mp.eps
    s = mp.mpf(1)
    term = mp.mpf(1)
    for n in range(1, max_terms + 1):
        term *= ((n - mp.mpf('0.5'))**3) * z / (n**3)
        s_new = s + term
        if abs(term) <= tol * abs(s_new):
            return s_new
        s = s_new
    raise RuntimeError("Series did not converge within max_terms")

def compute():
    # Non-trivial algebraic argument
    z = mp.sqrt(2) - 1

    with mp.workdps(140):
        # Clausen identity: 3F2(1/2,1/2,1/2;1,1;z) = [2F1(1/4,1/4;1;z)]^2
        f2 = mp.hyper([mp.mpf(1)/4, mp.mpf(1)/4], [mp.mpf(1)], z)
        val_clausen = f2 * f2

        # Independent computation by direct series for 3F2
        val_series = hyper3f2_half_series(z, tol=mp.mpf('1e-130'))

        # Return the more stable average if they agree closely
        if abs(val_clausen - val_series) <= mp.mpf('1e-120') * max(1, abs(val_clausen), abs(val_series)):
            return mp.mpf((val_clausen + val_series) / 2)
        return mp.mpf(val_clausen)

if __name__ == "__main__":
    print(str(compute()))