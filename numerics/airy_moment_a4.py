from mpmath import mp

mp.dps = 110


def compute():
    f = lambda x: mp.airyai(x) ** 4

    # Use extra precision for reliable 100+ digit output
    with mp.extradps(80):
        # Split the range to help the adaptive integrator
        T = mp.mpf(35)
        val = mp.quad(f, [0, 1, 4, 10, 20, T])

        # Tail beyond T is astronomically small; estimate with asymptotic bound
        # Ai(x)^4 ~ (1/(16*pi^2)) * x^{-1} * exp(-(8/3)*x^(3/2))
        # Add a conservative asymptotic tail integral approximation (negligible at this T)
        C = mp.mpf(1) / (16 * mp.pi**2)
        tail = mp.quad(lambda x: C * mp.e**(-(mp.mpf(8) / 3) * x**(mp.mpf(3) / 2)) / x, [T, mp.inf])

        return val + tail


if __name__ == "__main__":
    print(str(compute()))