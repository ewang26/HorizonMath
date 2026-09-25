from mpmath import mp

mp.dps = 110


def _lll_reduce(B, delta=mp.mpf("0.75")):
    # Basic floating-point LLL for small integer lattices
    B = [list(map(int, v)) for v in B]
    n = len(B)
    m = len(B[0])

    def gso(B):
        bstar = [[mp.mpf("0") for _ in range(m)] for _ in range(n)]
        mu = [[mp.mpf("0") for _ in range(n)] for _ in range(n)]
        Bnorm = [mp.mpf("0") for _ in range(n)]

        for i in range(n):
            vi = [mp.mpf(x) for x in B[i]]
            wi = vi[:]
            for j in range(i):
                denom = Bnorm[j]
                if denom != 0:
                    num = mp.fdot(vi, bstar[j])
                    mu[i][j] = num / denom
                    for k in range(m):
                        wi[k] -= mu[i][j] * bstar[j][k]
                else:
                    mu[i][j] = mp.mpf("0")
            bstar[i] = wi
            Bnorm[i] = mp.fdot(wi, wi)
        return bstar, mu, Bnorm

    k = 1
    while k < n:
        bstar, mu, Bnorm = gso(B)

        # size reduction
        for j in range(k - 1, -1, -1):
            r = int(mp.nint(mu[k][j]))
            if r:
                for i in range(m):
                    B[k][i] -= r * B[j][i]

        bstar, mu, Bnorm = gso(B)

        lhs = Bnorm[k]
        rhs = (delta - mu[k][k - 1] ** 2) * Bnorm[k - 1]
        if lhs >= rhs:
            k += 1
        else:
            B[k], B[k - 1] = B[k - 1], B[k]
            k = max(k - 1, 1)

    return B


def _integer_relation_lll(xs, powers=(60, 70, 80, 90)):
    # Try to find a small integer relation among xs via LLL on a scaled lattice.
    # Returns a list of integer coefficients [a0,a1,...] such that sum(ai*xi) ~ 0.
    import math

    n = len(xs)

    for p in powers:
        Q = 10 ** p
        qmp = mp.mpf(Q)

        B = []
        for i in range(n):
            v = [0] * (n + 1)
            v[i] = 1
            v[n] = int(mp.nint(qmp * xs[i]))
            B.append(v)

        Bred = _lll_reduce(B)

        best = None
        for v in Bred:
            coeff = v[:n]
            res = mp.fdot([mp.mpf(c) for c in coeff], xs)
            score = abs(res)
            if best is None or score < best[0]:
                best = (score, coeff)

        if best is None:
            continue

        score, coeff = best
        if score < mp.mpf("1e-80"):
            g = 0
            for c in coeff:
                g = math.gcd(g, abs(int(c)))
            if g > 1:
                coeff = [int(c // g) for c in coeff]

            for c in coeff:
                if c != 0:
                    if c < 0:
                        coeff = [-int(x) for x in coeff]
                    break
            return coeff

    return None


def compute():
    # Interpreting C_5 as the Ising-class Bessel moment:
    #   C_5 = ∫_0^∞ t * K0(t)^5 dt
    def f(t):
        if t == 0:
            return mp.mpf("0")
        k = mp.besselk(0, t)
        return t * k**5

    # Piecewise integration helps the adaptive routine
    intervals = [mp.mpf("0"), mp.mpf("1"), mp.mpf("2"), mp.mpf("4"), mp.mpf("8"), mp.inf]
    C5 = mp.mpf("0")
    for a, b in zip(intervals[:-1], intervals[1:]):
        C5 += mp.quad(f, [a, b])

    # Attempt an MZV weight-5 decomposition in the standard basis {zeta(5), zeta(2)zeta(3)}
    z5 = mp.zeta(5)
    z23 = mp.zeta(2) * mp.zeta(3)

    rel = _integer_relation_lll([C5, z5, z23])
    if rel is not None:
        a0, a1, a2 = rel
        if a0 != 0:
            combo = (-mp.mpf(a1) / a0) * z5 + (-mp.mpf(a2) / a0) * z23
            if abs(combo - C5) < mp.mpf("1e-90"):
                return combo

    return C5


if __name__ == "__main__":
    print(str(compute()))