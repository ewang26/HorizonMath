from mpmath import mp, mpf, odefun


TEST_POINTS = [
    ("0.5", "1.0", "1.0", "1.0", "1.0"),
    ("2.0", "0.8", "1.2", "0.7", "0.9"),
    ("5.0", "0.4", "0.6", "2.0", "0.5"),
]


def compute(t, k_bim, p0, epsilon_l, c0):
    mp.dps = 60

    t = mpf(t)
    k_bim = mpf(k_bim)
    p0 = mpf(p0)
    epsilon_l = mpf(epsilon_l)
    c0 = mpf(c0)
    ten = mpf(10)

    def rhs(_, C):
        return -k_bim * p0 * (1 - ten ** (-epsilon_l * C)) * C

    f = odefun(rhs, mpf(0), c0, tol=mpf(10) ** (-40))
    return f(t)


if __name__ == "__main__":
    for args in TEST_POINTS:
        print(args, mp.nstr(compute(*args), 25))
