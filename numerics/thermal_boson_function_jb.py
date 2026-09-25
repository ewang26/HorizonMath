from mpmath import mp


mp.dps = 120


TEST_POINTS = ["0.5", "2", "8"]


def compute(y):
    y = mp.mpf(y)
    f = lambda x: x**2 * mp.log(1 - mp.exp(-mp.sqrt(x * x + y)))
    return mp.quad(f, [0, mp.mpf("0.5"), 1, 2, 4, 8, 16, mp.inf])


if __name__ == "__main__":
    for y in TEST_POINTS:
        print(y, mp.nstr(compute(y), 100))
