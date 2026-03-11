from mpmath import mp

mp.dps = 110

def compute():
    n = 6
    factor = (2**n) * mp.factorial(n)

    with mp.workdps(160):
        # Use t = x^2 substitution for [0,1] to smooth log singularity
        def f_sub(x):
            if x == 0:
                return mp.zero
            t = x * x
            return 2 * x**3 * mp.besselk(0, t)**n

        def f(t):
            return t * mp.besselk(0, t)**n

        I1 = mp.quad(f_sub, [0, 1])
        I2 = mp.quad(f, [1, 5, 15, 40, mp.inf])

        C6 = factor * (I1 + I2)
        return +C6

if __name__ == "__main__":
    print(str(compute()))