from mpmath import mp

mp.dps = 110

def compute():
    n = 5
    pref = (2 ** n) * mp.factorial(n)

    with mp.extradps(40):
        # Use t = x^2 substitution for [0,1] to smooth log singularity
        def f_sub(x):
            if x == 0:
                return mp.zero
            t = x * x
            k = mp.besselk(0, t)
            return 2 * x**3 * (k ** n)  # Jacobian: dt = 2x dx, so t*dt = 2x^3 dx

        def f(t):
            if t == 0:
                return mp.zero
            k = mp.besselk(0, t)
            return t * (k ** n)

        # [0, 1] via substitution
        I1 = mp.quad(f_sub, [0, 1])

        # [1, infinity] directly
        I2 = mp.quad(f, [1, 5, 15, 40, mp.inf])

        C5 = pref * (I1 + I2)

    return +C5

if __name__ == "__main__":
    print(str(compute()))