from mpmath import mp

mp.dps = 110

def compute():
    # C_5 = ∫_0^∞ t * K0(t)^5 dt  (Bessel moment c_{5,1}).
    # Same integral as bessel_moment_c5_1.py but originally truncated at t=8.
    # Now integrates to infinity for full precision.

    with mp.workdps(mp.dps + 40):
        def f(t):
            """The integrand t * K_0(t)^5"""
            if t == 0:
                return mp.zero
            k0 = mp.besselk(0, t)
            return t * k0**5

        # For t in [0, 1]: substitute t = x^2, dt = 2x dx
        # Integral becomes: ∫_0^1 2 * x^3 * K_0(x^2)^5 dx
        def f_small(x):
            if x == 0:
                return mp.zero
            t = x * x
            k0 = mp.besselk(0, t)
            return 2 * x**3 * k0**5

        # Integrate [0,1] with substitution (handles log singularity)
        I1 = mp.quad(f_small, [mp.mpf(0), mp.mpf('0.5'), mp.mpf(1)])

        # Integrate [1, infinity] directly
        # K_0(t)^5 decays as exp(-5t), negligible beyond t~25
        I2 = mp.quad(f, [mp.mpf(1), mp.mpf(3), mp.mpf(8), mp.mpf(20), mp.inf])

        result = I1 + I2

    return +result  # Round to current precision

if __name__ == "__main__":
    print(mp.nstr(compute(), 110, strip_zeros=False))
