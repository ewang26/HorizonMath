"""
Numerical computation for: Fourth Moment of the Complete Elliptic Integral K(k)

Computes the integral:
    M_4 = integral_0^1 K(k^2)^4 dk

where K is the complete elliptic integral of the first kind with parameter m = k^2.

This uses the same approach as elliptic_k_moment_3.py with the substitution
k = 1 - exp(-t) to handle the singularity at k=1.
"""
from mpmath import mp

mp.dps = 110


def compute():
    with mp.workdps(250):
        def integrand_t(t):
            # k = 1 - exp(-t), computed accurately for small t
            k = -mp.expm1(-t)
            w = 1 - k  # exp(-t) = dk/dt
            K = mp.ellipk(k * k)  # parameter m = k^2
            return (K**4) * w

        T = mp.mpf(300)
        breaks = [mp.mpf(0), 1, 2, 4, 8, 16, 32, 64, 128, 256, T]

        total = mp.mpf('0')
        # sum small tail contributions first
        for a, b in reversed(list(zip(breaks[:-1], breaks[1:]))):
            total += mp.quad(integrand_t, [a, b])

        return +total  # round to current mp.dps on exit


if __name__ == "__main__":
    print(str(compute()))
