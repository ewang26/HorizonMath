from mpmath import mp

mp.dps = 110


def compute():
    f = lambda x: mp.airyai(x) ** 3

    # Use extra precision for reliable 100+ digit output
    with mp.extradps(80):
        # Split the range to help the adaptive integrator
        T = mp.mpf(35)
        val = mp.quad(f, [0, 1, 4, 10, 20, T])

        # Tail beyond T is astronomically small; estimate with asymptotic bound
        # Ai(x)^3 ~ (1/(8*pi^(3/2))) * x^(-3/4) * exp(-2*x^(3/2))
        C = mp.mpf(1) / (8 * mp.pi ** (mp.mpf(3) / 2))
        tail = mp.quad(lambda x: C * mp.exp(-2 * x ** (mp.mpf(3) / 2)) * x ** (mp.mpf(-3) / 4), [T, mp.inf])

        return val + tail


if __name__ == "__main__":
    print(str(compute()))
