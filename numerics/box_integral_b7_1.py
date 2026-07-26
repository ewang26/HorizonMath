from mpmath import mp


mp.dps = 110


def _poly_mul(a, b, degree):
    result = [mp.mpf("0")] * (degree + 1)
    for i, ai in enumerate(a[: degree + 1]):
        if not ai:
            continue
        for j in range(min(len(b) - 1, degree - i) + 1):
            result[i + j] += ai * b[j]
    return result


def _poly_pow(a, power, degree):
    result = [mp.mpf("0")] * (degree + 1)
    result[0] = mp.mpf("1")
    base = a[: degree + 1] + [mp.mpf("0")] * max(
        0, degree + 1 - len(a)
    )
    while power:
        if power & 1:
            result = _poly_mul(result, base, degree)
        power >>= 1
        if power:
            base = _poly_mul(base, base, degree)
    return result


def _poly_eval(coefficients, value):
    result = mp.mpf("0")
    for coefficient in reversed(coefficients):
        result = result * value + coefficient
    return result


def compute():
    """Compute B_7(1), the mean distance from a unit 7-cube vertex.

    For independent X_i uniformly distributed on [0, 1],

        B_7(1) = E[sqrt(X_1^2 + ... + X_7^2)].

    The Laplace representation of sqrt(x), followed by t = u^2, gives

        B_7(1) = 1/sqrt(pi) * integral_0^infinity
                 (1 - b(u)^7) / u^2 du,

    where b(u) = integral_0^1 exp(-u^2*x^2) dx
               = sqrt(pi)*erf(u)/(2*u).
    """

    # The endpoint transformation is smooth, but high-order quadrature loses
    # more guard digits than a typical finite-interval integral. Keep a large
    # precision margin so all 110 reported digits are stable.
    with mp.workdps(mp.dps + 130):
        sqrtpi = mp.sqrt(mp.pi)
        dimension = 7

        # Near u=0, evaluate g(z)=(1-b(sqrt(z))^7)/z as a power
        # series. This avoids cancellation in 1-b(u)^7.
        degree_g = 170
        degree_power = degree_g + 1
        # A coefficient of b(z)^dimension at degree k can contain the
        # degree-k coefficient of any one factor, so b itself must be known
        # through the full target degree.
        degree_b = degree_power
        b_coefficients = [
            (-1) ** k / (mp.factorial(k) * (2 * k + 1))
            for k in range(degree_b + 1)
        ]
        b_power = _poly_pow(
            b_coefficients,
            dimension,
            degree_power,
        )
        g_coefficients = [
            -b_power[k + 1] for k in range(degree_power)
        ]

        def transformed_integrand(t):
            # Map u in [0, infinity) to t in [0, 1] by
            # u = tan(pi*t/2).
            if t == 0:
                return dimension * mp.pi / 6
            if t == 1:
                return mp.pi / 2

            u = mp.tan(mp.pi * t / 2)
            if abs(u) < mp.mpf("0.2"):
                z = u * u
                g = _poly_eval(g_coefficients, z)
                one_minus_power = g * z
            else:
                b_value = sqrtpi * mp.erf(u) / (2 * u)
                one_minus_power = -mp.expm1(
                    dimension * mp.log(b_value)
                )
                g = one_minus_power / (u * u)

            # du/dt = (pi/2)*(1+u^2).
            return mp.pi / 2 * (one_minus_power + g)

        integral = mp.quad(
            transformed_integrand,
            [
                mp.mpf("0"),
                mp.mpf("0.25"),
                mp.mpf("0.5"),
                mp.mpf("0.75"),
                mp.mpf("0.9"),
                mp.mpf("0.99"),
                mp.mpf("1"),
            ],
        )
        return +(integral / sqrtpi)


if __name__ == "__main__":
    print(str(compute()))
