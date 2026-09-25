from mpmath import mp, mpf, pi, asin, exp, findroot


TEST_POINTS = [
    ("0.25", "1.0", "2.0"),
    ("0.5", "0.5", "1.5"),
    ("0.8", "2.0", "0.75"),
]


def compute(p, theta, alpha):
    mp.dps = 60

    p = mpf(p)
    theta = mpf(theta)
    alpha = mpf(alpha)

    target = (2 * asin(p) / pi) ** (1 / alpha)

    def residual(x):
        return 1 - exp(-theta * x) - theta * x * exp(-2 * theta * x) / (1 + theta) - target

    lo = mpf(0)
    hi = 1 / theta
    while residual(hi) < 0:
        hi *= 2
    return findroot(residual, (lo, hi), solver="anderson")


if __name__ == "__main__":
    for args in TEST_POINTS:
        print(args, mp.nstr(compute(*args), 25))
