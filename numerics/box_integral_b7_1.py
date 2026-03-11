from mpmath import mp

mp.dps = 110

def compute():
    n = 7
    K = 40  # series terms for small-t evaluation of L(t)

    with mp.extradps(50):
        # Precompute moments E[D^(2k)] for D = X-Y with X,Y ~ U(-1,1)
        # E[D^(2k)] = 2^(2k+1) / ((2k+1)(2k+2)), k>=1; and moment_0 = 1
        moments = [mp.mpf(0)] * (K + 1)
        facts = [mp.mpf(0)] * (K + 1)
        moments[0] = mp.mpf(1)
        facts[0] = mp.mpf(1)
        for k in range(1, K + 1):
            moments[k] = mp.power(2, 2*k + 1) / ((2*k + 1) * (2*k + 2))
            facts[k] = facts[k - 1] * k

        def L_series(t):
            s = mp.mpf(1)
            p = -t
            for k in range(1, K + 1):
                s += p * moments[k] / facts[k]
                p *= -t
            return s

        def L(t):
            if t == 0:
                return mp.mpf(1)
            # Use series where the closed form has cancellation (t -> 0)
            if t < mp.mpf("0.02"):
                return L_series(t)
            rt = mp.sqrt(t)
            term1 = mp.sqrt(mp.pi) * mp.erf(2 * rt) / (2 * rt)
            term2 = -mp.expm1(-4 * t) / (4 * t)  # (1 - exp(-4t)) / (4t)
            return term1 - term2

        def integrand(u):
            if u == 0:
                # limit u->0 of (1 - L(t)^n)/u^2 with t=(u/(1-u))^2:
                # 1 - L(t)^n ~ n*E[D^2]*t, E[D^2]=2/3, and t~u^2
                return mp.mpf(14) / 3
            if u == 1:
                return mp.mpf(1)

            a = u / (1 - u)
            t = a * a
            Lt = L(t)

            if abs(Lt - 1) < mp.mpf("0.1"):
                logLt = mp.log1p(Lt - 1)
            else:
                logLt = mp.log(Lt)

            one_minus_phi = -mp.expm1(n * logLt)  # 1 - Lt^n, stable for Lt~1
            return one_minus_phi / (u * u)

        # E[||D||] = 1/sqrt(pi) * ∫_0^1 (1 - E[e^{-t||D||^2}]) / u^2 du
        val = mp.quad(integrand, [0, mp.mpf("0.5"), mp.mpf("0.9"), mp.mpf("0.99"), mp.mpf("0.999"), 1])
        return +(val / mp.sqrt(mp.pi))

if __name__ == "__main__":
    print(str(compute()))