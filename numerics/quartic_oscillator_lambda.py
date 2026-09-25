from mpmath import mp


def compute(n, lam, N=150, dps=220):
    """
    Compute the n-th eigenvalue epsilon_n(lam) of

        H = -1/2 d^2/dx^2 + x^4/4 - lam*x^2/2

    using a parity-blocked truncated harmonic oscillator basis of size N.

    H = H_HO + x^4/4 - (1 + lam)/2 * x^2
    where H_HO |k> = (k + 1/2)|k>.

    Parity symmetry (H commutes with x -> -x) is exploited: the basis is
    {|parity + 2k> : k = 0, ..., N-1} where parity = n % 2.

    Matrix elements within the parity sector (l = parity + 2k):
      A_l = sqrt(l*(l-1)) / 2   (x^2 sub-diagonal, l >= 2; else 0)
      B_l = (2*l + 1) / 2       (x^2 diagonal)
      C_l = sqrt((l+1)*(l+2)) / 2  (x^2 super-diagonal)

    x^4 is derived analytically from (x^2)^2:
      diagonal k:     A_l^2 + B_l^2 + C_l^2
      off-diag k+1:   C_l * (B_l + B_{l+2})
      off-diag k+2:   C_l * C_{l+2}
    """
    mp.dps = dps
    lam = mp.mpf(lam)

    parity = n % 2
    k_target = n // 2

    H = mp.matrix(N)

    for k in range(N):
        l = mp.mpf(parity + 2 * k)

        # x^2 matrix element helpers for |l>
        A_l = mp.sqrt(l * (l - 1)) / 2 if l >= 2 else mp.mpf(0)
        B_l = (2 * l + 1) / 2
        C_l = mp.sqrt((l + 1) * (l + 2)) / 2

        # Diagonal
        # x^4 diagonal = A_l^2 + B_l^2 + C_l^2  (since C_{l-2}=A_l, A_{l+2}=C_l)
        H[k, k] = (l + mp.mpf("0.5")) + (A_l**2 + B_l**2 + C_l**2) / 4 - (1 + lam) / 2 * B_l

        # k+1 off-diagonal: x^4 contribution = C_l*(B_l + B_{l+2}), x^2 = C_l
        if k + 1 < N:
            B_lp2 = (2 * (l + 2) + 1) / 2
            val = C_l * (B_l + B_lp2) / 4 - (1 + lam) / 2 * C_l
            H[k, k + 1] = val
            H[k + 1, k] = val

        # k+2 off-diagonal: x^4 only = C_l * C_{l+2}
        if k + 2 < N:
            l_p2 = l + 2
            C_lp2 = mp.sqrt((l_p2 + 1) * (l_p2 + 2)) / 2
            val = C_l * C_lp2 / 4
            H[k, k + 2] = val
            H[k + 2, k] = val

    evals, _ = mp.eigsy(H)
    evals = sorted(evals[i] for i in range(len(evals)))
    return evals[k_target]


if __name__ == "__main__":
    tests = [
        (0, "0"),
        (1, "0"),
        (2, "0"),
        (0, "1"),
        (1, "1"),
        (2, "1"),
        (0, "2"),
        (1, "2"),
        (2, "2"),
        (3, "2"),
        (0, "3"),
        (1, "3"),
    ]
    for n, lam in tests:
        val = compute(n, lam)
        print(f"n={n} lam={lam}: {mp.nstr(val, 45)}")
