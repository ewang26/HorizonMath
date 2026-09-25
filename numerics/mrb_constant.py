from mpmath import mp

mp.dps = 110


def compute():
    # MRB constant: M = sum_{n=1}^{infty} (-1)^n (n^{1/n} - 1)
    # OEIS A037077: 0.18785964246206712024851793405427...
    # Using mpmath's nsum with built-in convergence acceleration.
    with mp.extradps(30):
        return mp.nsum(
            lambda n: (-1) ** n * (mp.power(n, mp.one / n) - 1), [1, mp.inf]
        )


if __name__ == "__main__":
    print(str(compute()))
