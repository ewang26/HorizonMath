from mpmath import mp, li, findroot

mp.dps = 110


def compute():
    # The Ramanujan-Soldner constant μ is the unique positive zero of li(x).
    # Use Newton's method (findroot) starting near 2.
    with mp.extradps(20):
        mu = findroot(li, 2)
    return mu


if __name__ == "__main__":
    print(str(compute()))
