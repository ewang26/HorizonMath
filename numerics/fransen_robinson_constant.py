from mpmath import mp

mp.dps = 110


def compute():
    # Fransén-Robinson constant: F = integral from 0 to infinity of 1/Gamma(x) dx
    # OEIS A058655: 2.8077702420285193652215011865577729...
    # 1/Gamma(x) is entire and decays super-exponentially for large x.
    with mp.extradps(30):
        f = lambda x: mp.one / mp.gamma(x)
        # Breakpoints help the adaptive integrator handle the peak near x ~ 1-2
        val = mp.quad(f, [0, 1, 2, 5, 10, 20, mp.inf])
        return val


if __name__ == "__main__":
    print(str(compute()))
