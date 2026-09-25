from mpmath import mp

mp.dps = 110

def compute():
    f = lambda t: mp.besselk(0, t) ** 5

    # Integral on [0,1], with t = x^2 to avoid 0*inf issues and smooth endpoint
    def g1(x):
        if x == 0:
            return mp.zero
        t = x * x
        return 2 * x * f(t)

    I1 = mp.quad(g1, [0, mp.mpf('0.25'), mp.mpf('0.5'), mp.mpf('0.75'), 1])

    # Integral on [1,∞), with t = 1 + u/(1-u), u in [0,1)
    def g2(u):
        if u == 1:
            return mp.zero
        omu = 1 - u
        t = 1 + u / omu
        return f(t) / (omu * omu)

    I2 = mp.quad(g2, [0, mp.mpf('0.5'), mp.mpf('0.9'), mp.mpf('0.99'), mp.mpf('0.999'), 1])

    return I1 + I2

if __name__ == "__main__":
    print(str(compute()))