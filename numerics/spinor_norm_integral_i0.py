from mpmath import mp


mp.dps = 120


def compute():
    f = lambda z: (
        (1 + mp.sqrt(z))
        / (1 + z) ** 3
        * (2 * mp.ellipe(z) - (1 - z) * mp.ellipk(z))
    )
    return 4 * mp.sqrt(2) / mp.pi * mp.quad(f, [0, mp.mpf("0.25"), mp.mpf("0.5"), mp.mpf("0.75"), 1])


if __name__ == "__main__":
    print(mp.nstr(compute(), 100))
