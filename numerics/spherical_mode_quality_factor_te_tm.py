"""High-precision evaluator for the equal-power TE+TM spherical-mode Q family.

The source defines

    Q_n^{TE+TM}(ka) = int_{ka}^inf [ (1 + n(n+1)/rho^2) |rho h_n^{(2)}(rho)|^2
                                     + |d/drho (rho h_n^{(2)}(rho))|^2 ]
                                   * |Gamma(rho)|^2 / (1 + |Gamma(rho)|^2) drho

with Gamma determined by the local wave impedance and characteristic
impedance. This script keeps the full family from the paper by writing the
cutoff behavior explicitly:

    w_n(rho) = 1/2                                       for 0 < rho <= sqrt(n(n+1))
             = |Gamma_n(rho)|^2 / (1 + |Gamma_n(rho)|^2)  for rho > sqrt(n(n+1))

The first branch is exactly the paper's statement that Gamma = 1 below
cutoff. Above cutoff we use the paper's TM impedance convention

    eta_n(rho) = i * d[rho h_n^(2)(rho)]/drho / (rho h_n^(2)(rho)).
"""

from mpmath import mp


TEST_POINTS = [
    ("1", "1.0"),
    ("2", "1.5"),
    ("2", "2.44"),
    ("2", "2.46"),
    ("2", "2.5"),
    ("3", "4.0"),
    ("5", "8.0"),
    ("3", "5.0"),
    ("5", "10.0"),
]


def h2(n, rho):
    return mp.sqrt(mp.pi / (2 * rho)) * mp.hankel2(n + mp.mpf("0.5"), rho)


def drho_rho_h2(n, rho):
    return rho * h2(n - 1, rho) - n * h2(n, rho)


def gamma_tm(n, rho):
    eta = 1j * drho_rho_h2(n, rho) / (rho * h2(n, rho))
    z0 = mp.sqrt(1 - n * (n + 1) / rho**2)
    return (eta - z0) / (eta + z0)


def weight(n, rho):
    cutoff = mp.sqrt(n * (n + 1))
    if rho <= cutoff:
        return mp.mpf("0.5")
    gamma = gamma_tm(n, rho)
    return abs(gamma) ** 2 / (1 + abs(gamma) ** 2)


def integrand(rho, n):
    h = h2(n, rho)
    deriv = drho_rho_h2(n, rho)
    amplitude = (1 + n * (n + 1) / rho**2) * abs(rho * h) ** 2 + abs(deriv) ** 2
    return amplitude * weight(n, rho)


def compute(n, x):
    n = int(mp.mpf(n))
    x = mp.mpf(x)
    if n < 1:
        raise ValueError("n must be a positive integer")
    if x <= 0:
        raise ValueError("x must be positive")

    intervals = [x]
    cutoff = mp.sqrt(n * (n + 1))
    if x < cutoff:
        intervals.append(cutoff)
    for bound in (5, 10, 20, 40, 80, 160):
        bound = mp.mpf(bound)
        if bound > intervals[-1]:
            intervals.append(bound)
    intervals.append(mp.inf)
    return mp.quad(lambda rho: integrand(rho, n), intervals)


if __name__ == "__main__":
    mp.dps = 100
    for n, x in TEST_POINTS:
        print(n, x, mp.nstr(compute(n, x), 40))
