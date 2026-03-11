from mpmath import mp

mp.dps = 110


def sunset_2d(m1, m2, m3, s):
    m1 = mp.mpf(m1)
    m2 = mp.mpf(m2)
    m3 = mp.mpf(m3)
    s = mp.mpf(s)

    m1sq = m1 * m1
    m2sq = m2 * m2
    m3sq = m3 * m3

    def F(x1, x2, x3):
        U = x1 * x2 + x2 * x3 + x3 * x1
        A = m1sq * x1 + m2sq * x2 + m3sq * x3
        return A * U - s * x1 * x2 * x3

    def integrand(u, v):
        # Map unit square (u,v) -> simplex via:
        # x1 = u*(1-v), x2 = u*v, x3 = 1-u, Jacobian = u
        x1 = u * (1 - v)
        x2 = u * v
        x3 = 1 - u
        return u / F(x1, x2, x3)

    with mp.extradps(40):
        # Use native 2D quadrature (faster than nested 1D quad)
        val = mp.quad(integrand, [0, 1], [0, 1])

        # Standard D=2 normalization from Feynman parameters:
        # I = 1/(4*pi)^(L*D/2) * integral, with L=2, D=2 -> 1/(4*pi)^2
        val *= 1 / (4 * mp.pi) ** 2

    return mp.re(val)


def compute():
    # Representative "generic masses" and a nontrivial kinematic point below threshold:
    # m1=1, m2=2, m3=3, threshold s_th=(1+2+3)^2=36, choose s=30
    return sunset_2d(1, 2, 3, 30)


if __name__ == "__main__":
    print(str(compute()))