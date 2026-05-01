from mpmath import mp, mpf


TEST_POINTS = [
    ("0.5", "0.6", "-0.45"),
    ("1.2", "0.4", "-0.35"),
    ("2.0", "0.7", "-0.6"),
]


def compute(lam, n, pi_q, dps=80, quad_degree=40):
    mp.dps = dps

    lam = mpf(lam)
    n = mpf(n)
    pi_q = mpf(pi_q)

    # Gauss-Legendre nodes and weights on [0, 1]
    nodes_raw, weights_raw = mp.gauss_quadrature(quad_degree, "legendre")
    eta_nodes = [(x + 1) / 2 for x in nodes_raw]
    eta_weights = [w / 2 for w in weights_raw]

    def shear_from_stress(tau):
        # Solve viscosity(q) * q = tau for q > 0, where
        # viscosity(q) = 1 / (1 + (lam * q)^(1 - n))
        if tau <= 0:
            return mpf(0)

        def residual(q):
            return q / (1 + (lam * q) ** (1 - n)) - tau

        # Upper bracket: viscosity -> 0 as q grows (since 1 - n > 0),
        # but viscosity * q grows without bound because q outpaces (lam*q)^(1-n)
        hi = mp.mpf(1) if tau < 1 else tau + 1
        while residual(hi) < 0:
            hi *= 2
        lo = mpf(0)
        return mp.findroot(residual, (lo, hi), solver="anderson")

    def system(log_tau0, log_tau1):
        tau0 = mp.exp(log_tau0)
        tau1 = mp.exp(log_tau1)
        # tau(eta) = tau0 + (tau1 - tau0) * eta
        shears = [
            shear_from_stress(tau0 + (tau1 - tau0) * e) for e in eta_nodes
        ]
        # ∫_0^1 u'(eta) d_eta = u(1) - u(0) = 0 - (-1) = 1
        endpoint_gap = sum(w * s for w, s in zip(eta_weights, shears)) - 1
        # u(eta) = -1 + ∫_0^eta u'(s) ds
        # ∫_0^1 u(eta) d_eta = -1 + ∫_0^1 (1 - eta) u'(eta) d_eta = pi_q
        flux = (
            -1
            + sum(w * (1 - e) * s for w, e, s in zip(eta_weights, eta_nodes, shears))
            - pi_q
        )
        return endpoint_gap, flux

    sol = mp.findroot(system, (mpf(0), mpf(0)), solver="mnewton")
    log_tau0, log_tau1 = sol[0], sol[1]
    tau0 = mp.exp(log_tau0)
    tau1 = mp.exp(log_tau1)
    return tau1 - tau0


if __name__ == "__main__":
    for args in TEST_POINTS:
        print(args, mp.nstr(compute(*args), 25))
