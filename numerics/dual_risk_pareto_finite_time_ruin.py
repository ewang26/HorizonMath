from mpmath import mp


mp.dps = 100

U0 = mp.mpf(10)
T_HORIZON = mp.mpf(20)
LAMBDA = mp.mpf(11) / mp.mpf(10)


def expint_e2(z):
    return mp.exp(-z) - z * mp.e1(z)


def expint_e3(z):
    return (mp.exp(-z) - z * expint_e2(z)) / 2


def laplace_transform_fX(theta):
    # L{f_X}(theta) = 2 exp(theta) E_3(theta) for f_X(x) = 2/(1+x)^3.
    return 2 * mp.exp(theta) * expint_e3(theta)


def laplace_transform_fX_derivative(theta):
    return 2 * mp.exp(theta) * (expint_e3(theta) - expint_e2(theta))


def levy_exponent(theta):
    return theta + LAMBDA * (laplace_transform_fX(theta) - 1)


def levy_exponent_derivative(theta):
    return 1 + LAMBDA * laplace_transform_fX_derivative(theta)


def first_passage_root(q):
    # Positive/root branch solving levy_exponent(theta) = q.
    theta = q + LAMBDA
    if abs(q) < 2:
        if abs(q) < mp.mpf("1e-30"):
            theta = mp.mpf("0.03")
        else:
            theta = mp.mpf("0.18") + q / mp.mpf("0.7")
    for _ in range(24):
        step = (levy_exponent(theta) - q) / levy_exponent_derivative(theta)
        theta -= step
        if abs(step) < mp.eps * max(1, abs(theta)) * 100:
            break
    return theta


def compute():
    mp.dps = 100

    def finite_time_ruin_laplace(q):
        return mp.exp(-U0 * first_passage_root(q)) / q

    return mp.invertlaplace(finite_time_ruin_laplace, T_HORIZON, method="dehoog")


if __name__ == "__main__":
    print(mp.nstr(compute(), 70))
