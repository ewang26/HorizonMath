from mpmath import mp


mp.dps = 120


TEST_POINTS = [("1", "0.2"), ("2", "0.4"), ("-1", "0.6")]


def compute(tau, kappa):
    tau = mp.mpf(tau)
    kappa = mp.mpf(kappa)
    f = lambda x: x**2 * mp.exp(-(x**6) + tau * (x**4) - kappa * tau**2 * x**2)
    return 2 * mp.quad(f, [0, mp.mpf("0.5"), 1, mp.mpf("1.5"), 2, 3, mp.inf])


if __name__ == "__main__":
    for tau, kappa in TEST_POINTS:
        print(tau, kappa, mp.nstr(compute(tau, kappa), 100))
