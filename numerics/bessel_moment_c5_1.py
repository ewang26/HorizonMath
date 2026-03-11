"""
Numerical computation for: Bessel Moment c_{5,1}

The Bessel function moments are defined by:
    c_{n,k} = integral_0^infinity t^k * K_0(t)^n dt

This computes c_{5,1} = integral_0^infinity t * K_0(t)^5 dt

where K_0 is the modified Bessel function of the second kind.

Behavior:
  - At t=0: K_0(t) ~ -ln(t/2) - gamma, so integrand has log^5 singularity
  - At t=infinity: K_0(t) ~ sqrt(pi/(2t)) * exp(-t), decays super-exponentially

Reference:
  Bailey, Borwein, Broadhurst, Glasser (2008), "Elliptic integral evaluations
  of Bessel moments and applications", https://arxiv.org/abs/0801.0891
"""
from mpmath import mp

mp.dps = 110


def compute():
    """
    Compute c_{5,1} = integral_0^infinity t * K_0(t)^5 dt

    Uses variable substitutions to handle endpoint behavior:
    - Near t=0: use t = x^2 substitution to smooth the log singularity
    - At infinity: K_0 decays as exp(-t), so integral converges rapidly
    """
    with mp.workdps(mp.dps + 40):
        def f(t):
            """The integrand t * K_0(t)^5"""
            if t == 0:
                return mp.zero
            k0 = mp.besselk(0, t)
            return t * k0**5

        # For t in [0, 1]: substitute t = x^2, dt = 2x dx
        # Integral becomes: integral_0^1 2 * x^3 * K_0(x^2)^5 dx
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
