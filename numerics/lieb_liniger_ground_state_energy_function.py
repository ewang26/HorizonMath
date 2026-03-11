from mpmath import mp


def lieb_liniger_e(gamma, n_nodes=160, dps=140):
    mp.dps = dps
    gamma = mp.mpf(gamma)

    # Gauss–Legendre nodes/weights on [-1,1]
    X, W = mp.gauss_quadrature(n_nodes, "legendre")
    D = [[(X[j] - X[i])**2 for j in range(n_nodes)] for i in range(n_nodes)]
    two_pi = 2 * mp.pi
    rhs = mp.mpf(1) / two_pi

    def gamma_and_e_from_alpha(alpha):
        alpha = mp.mpf(alpha)
        alpha2 = alpha * alpha
        coef = (mp.mpf(1) / two_pi) * (2 * alpha)

        A = mp.matrix(n_nodes)
        b = mp.matrix(n_nodes, 1)
        for i in range(n_nodes):
            b[i] = rhs

        for i in range(n_nodes):
            for j in range(n_nodes):
                val = mp.mpf(1) if i == j else mp.mpf(0)
                val -= coef * W[j] / (alpha2 + D[i][j])
                A[i, j] = val

        g = mp.lu_solve(A, b)

        I0 = mp.mpf(0)
        I2 = mp.mpf(0)
        for i in range(n_nodes):
            I0 += W[i] * g[i]
            I2 += W[i] * g[i] * (X[i] ** 2)
        gam = alpha / I0
        e = I2 / (I0 ** 3)
        return gam, e

    # Secant inversion for alpha(gamma)
    # Two decent initial guesses: weak-coupling and strong-coupling heuristics
    a0 = mp.sqrt(gamma) / 2
    a1 = gamma / mp.pi + mp.mpf("0.2")

    f0 = gamma_and_e_from_alpha(a0)[0] - gamma
    f1 = gamma_and_e_from_alpha(a1)[0] - gamma

    for _ in range(8):
        a2 = a1 - f1 * (a1 - a0) / (f1 - f0)
        a0, f0, a1, f1 = a1, f1, a2, gamma_and_e_from_alpha(a2)[0] - gamma

    gam, e = gamma_and_e_from_alpha(a1)
    return e


if __name__ == "__main__":
    for g in ["0.5", "1.0", "2.0", "5.0", "10.0"]:
        val = lieb_liniger_e(g, n_nodes=160, dps=140)
        print(g, mp.nstr(val, 90))
