from mpmath import mp


mp.dps = 120


TEST_POINTS = [("1", "0.25"), ("2", "-0.5"), ("0.75", "1.25")]


def compute(lam, nu):
    lam = mp.mpf(lam)
    nu = mp.mpf(nu)
    f = lambda x: (x * x - 1) ** mp.mpf("1.5") / (mp.exp(lam * x - nu) + 1)
    return mp.quad(f, [1, mp.mpf("1.25"), mp.mpf("1.5"), 2, 3, 5, 8, 12, 20, mp.inf])


if __name__ == "__main__":
    for lam, nu in TEST_POINTS:
        print(lam, nu, mp.nstr(compute(lam, nu), 100))
