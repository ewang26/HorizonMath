"""High-precision evaluator for the non-resonant TM/TE spherical-mode Q family.

The source defines

    Q_n^TM(ka) = Q_n^TE(ka)
               = int_{ka}^inf [ n(n+1) |h_n^(2)(rho)|^2
                                + |d[rho h_n^(2)(rho)]/drho|^2 ] v_n(rho) drho

with

    v_n(rho) = 1                                      for 0 < rho <= sqrt(n(n+1))
             = 2 |Gamma_n(rho)|^2 / (1 + |Gamma_n(rho)|^2)  above cutoff.

The first branch is exactly the paper's statement that Gamma_n = 1 below
cutoff. Above cutoff we use the paper's TM impedance convention

    eta_n(rho) = i * d[rho h_n^(2)(rho)]/drho / (rho h_n^(2)(rho)).
"""

from mpmath import mp


TEST_POINTS = [
    ("1", "0.5"),
    ("2", "3.0"),
    ("4", "8.0"),
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
        return mp.mpf("1")
    gamma = gamma_tm(n, rho)
    return 2 * abs(gamma) ** 2 / (1 + abs(gamma) ** 2)


def integrand(rho, n):
    return (
        n * (n + 1) * abs(h2(n, rho)) ** 2 + abs(drho_rho_h2(n, rho)) ** 2
    ) * weight(n, rho)


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
