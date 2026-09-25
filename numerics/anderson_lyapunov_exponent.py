import numpy as np
from numpy.polynomial.hermite_e import hermegauss

# Ground-truth values computed via Nyström discretization of the Fredholm stationarity equation
# for the Riccati map of the 1D Anderson model transfer matrix.
#
# Model: (Hψ)_n = -ψ_{n+1} - ψ_{n-1} + v_n ψ_n, v_n ~ N(0, σ²) i.i.d.
# Transfer matrix at E=0: T_n = [[-v_n, -1], [1, 0]] ∈ SL(2,ℝ)
# Lyapunov exponent: γ(σ) = lim_{n→∞} (1/n) E[log ‖T_n ... T_1‖]
#
# Method: Furstenberg-Khasminskii formula in sinh-parameterization.
#   z = sinh(s) parametrizes the projective line RP¹ = ℝ.
#   Stationary density q(s) (in s-coordinate) satisfies the Fredholm equation:
#     q(s') = ∫ cosh(s') φ_σ(sinh(s') + csch(s)) q(s) ds
#   Lyapunov exponent:
#     γ(σ) = ∫ F(s) q(s) ds
#   where
#     F(s) = (1/2) E_v[log((v·sinh(s)+1)² + sinh²(s))] - log(cosh(s)),  v ~ N(0, σ²)
#
# Nyström (midpoint rule) with N points on [-L, L], column-normalized stochastic matrix,
# power iteration for stationary vector q, Gauss-Hermite for F(s).
#
# Precision: limited to ~12-15 significant digits at N=16000 (float64 limit).
# The discretization error is super-algebraically convergent (≈ exp(-c/h)) but
# the essential singularity of the kernel at s=0 (csch(s) → ∞) means N~32000
# would be needed for 20-digit accuracy, requiring mpmath and ~days of compute.

def compute(sigma, N=16000, L=20.0):
    """
    Compute γ(σ) = Lyapunov exponent of 1D Anderson model at E=0,
    with Gaussian disorder N(0, σ²).

    Parameters
    ----------
    sigma : float, σ > 0
    N     : int, number of discretization nodes (default 16000 for ~12-15 digits)
    L     : float, truncation of the sinh-parameterized domain (default 20.0)

    Returns
    -------
    float : γ(σ)
    """
    ds = 2 * L / N
    s = -L + (np.arange(N) + 0.5) * ds   # midpoint rule nodes
    z = np.sinh(s)                         # z_j = sinh(s_j)
    ch = np.cosh(s)                        # cosh(s_j)

    # Build kernel K[i,j] = cosh(s_i) * φ_σ(sinh(s_i) + csch(s_j)) * ds
    # The argument is sinh(s_i) + 1/sinh(s_j)
    inv_z = 1.0 / z                        # csch(s_j)
    v_mat = z[:, np.newaxis] + inv_z[np.newaxis, :]   # (N, N), argument of φ_σ
    K = (np.exp(-v_mat**2 / (2 * sigma**2))
         / (sigma * np.sqrt(2 * np.pi))
         * ch[:, np.newaxis]
         * ds)

    # Column-normalize to stochastic matrix
    K /= K.sum(axis=0, keepdims=True)

    # Power iteration for stationary distribution
    q = np.ones(N) / N
    for _ in range(10000):
        q_new = K @ q
        q_new /= q_new.sum()
        if np.max(np.abs(q_new - q)) < 1e-15:
            break
        q = q_new

    # Furstenberg-Khasminskii integrand F(s)
    M_gh = 200
    gh_nodes, gh_weights = hermegauss(M_gh)   # Gauss-Hermite for N(0,1)
    v_gh = sigma * gh_nodes                   # v ~ N(0, σ²)
    inner = np.array([
        np.sum(gh_weights * np.log((v_gh * z[j] + 1)**2 + z[j]**2))
        / np.sqrt(2 * np.pi)
        for j in range(N)
    ])
    F = 0.5 * inner - np.log(ch)

    return np.sum(q * F)   # γ = ∫ F(s) q(s) ds (q is already a probability vector)


if __name__ == "__main__":
    print("Computing Lyapunov exponent γ(σ) for 1D Anderson model at E=0")
    print("(N=8000 and N=16000 to estimate precision)\n")

    for sigma in [1.0, 1.5, 2.0]:
        g8 = compute(sigma, N=8000)
        g16 = compute(sigma, N=16000)
        print(f"σ = {sigma}:")
        print(f"  N=8000  : {g8:.18f}")
        print(f"  N=16000 : {g16:.18f}")
        print(f"  |diff|  : {abs(g16 - g8):.2e}  "
              f"(~{int(-np.log10(abs(g16-g8)))} reliable digits)")
        print()
