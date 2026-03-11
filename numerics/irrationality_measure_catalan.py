from mpmath import mp

mp.dps = 110

def compute():
    # Note: The irrationality measure μ(G) (and even the irrationality of G) is an open problem.
    # What we can compute to high precision is Catalan's constant itself:
    #   G = ∫_0^1 atan(t)/t dt = Im(Li_2(i))
    def f(t):
        return mp.mpf(1) if t == 0 else mp.atan(t) / t

    G_int = mp.quad(f, [0, 1])

    # Cross-check via polylog identity (not used for output, just sanity):
    G_poly = mp.im(mp.polylog(2, 1j))
    if abs(G_int - G_poly) > mp.mpf('1e-100'):
        raise ValueError("Cross-check failed: integral and polylog values disagree at required precision.")

    return G_int

if __name__ == "__main__":
    print(str(compute()))