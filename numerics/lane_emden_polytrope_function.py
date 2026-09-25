"""Reference numerics for the Lane-Emden polytrope function benchmark.

This script computes high-precision values of the regular Lane-Emden solution

    (1/x^2) d/dx (x^2 theta'(x)) + theta(x)^n = 0,
    theta(0) = 1, theta'(0) = 0,

in the finite-polytrope regime 0 < n < 5, n != 1.

Method:
- seed the regular branch from its power series at a small epsilon;
- integrate with fixed-step RK4 at two step sizes;
- use Richardson extrapolation to suppress the O(h^4) global error.
"""

from mpmath import mp


TEST_POINTS = [
    ("1.5", "0.75"),
    ("2.0", "1.0"),
    ("2.5", "2.0"),
    ("3.0", "1.5"),
    ("4.0", "2.5"),
]


def _init_series(n, x):
    # theta(x) = 1 + a x^2 + b x^4 + c x^6 + d x^8 + e x^10 + ...
    a = -mp.mpf(1) / 6
    b = n / 120
    c = -n * (8 * n - 5) / 15120
    d = n * (122 * n**2 - 183 * n + 70) / 3265920
    e = -n * (5032 * n**3 - 12642 * n**2 + 10805 * n - 3150) / 1796256000

    theta = 1 + a * x**2 + b * x**4 + c * x**6 + d * x**8 + e * x**10
    dtheta = 2 * a * x + 4 * b * x**3 + 6 * c * x**5 + 8 * d * x**7 + 10 * e * x**9
    return theta, dtheta


def _rk4_theta(n, x, h):
    if x == 0:
        return mp.mpf(1)

    eps = mp.mpf("1e-6")
    theta, dtheta = _init_series(n, eps)
    t = eps

    def rhs(t_local, theta_local, dtheta_local):
        return dtheta_local, -(2 / t_local) * dtheta_local - theta_local**n

    while t < x:
        step = min(h, x - t)

        k1_theta, k1_dtheta = rhs(t, theta, dtheta)
        k2_theta, k2_dtheta = rhs(
            t + step / 2,
            theta + step * k1_theta / 2,
            dtheta + step * k1_dtheta / 2,
        )
        k3_theta, k3_dtheta = rhs(
            t + step / 2,
            theta + step * k2_theta / 2,
            dtheta + step * k2_dtheta / 2,
        )
        k4_theta, k4_dtheta = rhs(
            t + step,
            theta + step * k3_theta,
            dtheta + step * k3_dtheta,
        )

        theta += step * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta) / 6
        dtheta += step * (k1_dtheta + 2 * k2_dtheta + 2 * k3_dtheta + k4_dtheta) / 6
        t += step

    return theta


def compute(n, x):
    mp.dps = 100

    n = mp.mpf(n)
    x = mp.mpf(x)

    if not (mp.mpf(0) < n < mp.mpf(5)) or n == 1:
        raise ValueError("expected 0 < n < 5 with n != 1")
    if x < 0:
        raise ValueError("expected x >= 0")

    coarse = _rk4_theta(n, x, mp.mpf("0.00025"))
    fine = _rk4_theta(n, x, mp.mpf("0.000125"))
    richardson_1 = fine + (fine - coarse) / 15

    coarser = _rk4_theta(n, x, mp.mpf("0.0005"))
    richardson_0 = coarse + (coarse - coarser) / 15

    return richardson_1 + (richardson_1 - richardson_0) / 15


if __name__ == "__main__":
    for args in TEST_POINTS:
        print(args, mp.nstr(compute(*args), 18))
