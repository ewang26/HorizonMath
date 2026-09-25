from mpmath import mp, mpf, matrix, power, eye, det, nstr

# All ground-truth numerical values used in the script are from this paper: 
# https://www.ams.org/journals/btran/2022-09-35/S2330-0000-2022-00109-6/S2330-0000-2022-00109-6.pdf

def _build_matrix(N, s, M):
    """
    M×M monomial-basis truncation of the Ruelle transfer operator L_N^(s).

    [L_N^(s) x^j](x) = sum_{n=1}^{N} (n+x)^{-(2s+j)}

    Expanding (n+x)^{-alpha} = sum_{i>=0} (-1)^i * (alpha)_i/i! * n^{-(alpha+i)} * x^i:

        A[i,j] = (-1)^i * (2s+j)_i / i! * sigma_{j+i}(s)

    where sigma_k(s) = sum_{n=1}^{N} n^{-(2s+k)}.

    d(N) is the zero of det(I - A_M(s)), the Fredholm determinant approximation.
    """
    sigma = []
    for k in range(2 * M):
        alpha = 2 * s + k
        sigma.append(sum(power(mpf(n), -alpha) for n in range(1, N + 1)))

    A = matrix(M, M)
    for j in range(M):
        alpha_j = 2 * s + j
        poch = mpf(1)
        fact = mpf(1)
        for i in range(M):
            if i > 0:
                poch *= (alpha_j + i - 1)
                fact *= i
            coeff = poch / fact
            if i % 2 == 1:
                coeff = -coeff
            A[i, j] = coeff * sigma[j + i]
    return A


def compute(N, M=70, dps=25):
    """
    Compute d(N) = Hausdorff dimension of
        E_N = {x in [0,1] : all continued-fraction partial quotients of x are <= N}

    Method: bisect on the sign of det(I - A_M(s)), the Fredholm determinant
    approximation. At s < d(N) the sign is -1; at s > d(N) it is +1.
    Accuracy is ~(M/3) significant digits for N=2; M=70 gives ~24 digits.

    Parameters
    ----------
    N : int, N >= 2
    M : matrix truncation size (default 70 for ~24 digits)
    dps : working decimal precision (should exceed M/3)

    Returns
    -------
    mpf : d(N) to approximately min(dps, M/3) significant digits
    """
    mp.dps = max(dps, M // 2) + 20

    s0_map = {2: "0.531", 3: "0.731", 4: "0.819", 5: "0.870"}
    s0 = mpf(s0_map.get(N, str(round(1.0 - 6.0 / (3.14159265 ** 2 * N), 3))))
    s_lo = s0 - mpf("0.1")
    s_hi = s0 + mpf("0.1")

    sign_lo = 1 if det(eye(M) - _build_matrix(N, s_lo, M)) > 0 else -1

    tol = mpf(10) ** (-(dps + 5))
    while s_hi - s_lo > tol:
        s_mid = (s_lo + s_hi) / 2
        d = det(eye(M) - _build_matrix(N, s_mid, M))
        if (1 if d > 0 else -1) == sign_lo:
            s_lo = s_mid
        else:
            s_hi = s_mid

    return (s_lo + s_hi) / 2


if __name__ == "__main__":
    for N in [2, 3, 4, 5]:
        val = compute(N, M=70, dps=25)
        print(f"N={N}: {nstr(val, 25)}")
