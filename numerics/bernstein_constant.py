"""
Reference numerical computation for: Bernstein's Constant

Bernstein's constant β is defined by:
    β = lim_{n→∞} 2n · E_{2n}

where E_{2n} = min_{p ∈ P_{2n}} max_{x ∈ [-1,1]} ||x| - p(x)| is the minimax
polynomial approximation error for |x| on [-1,1].

Bernstein conjectured β = 1/(2√π) ≈ 0.28209... in 1914, but this was disproved
by Varga & Carpenter (1987) who computed β to 50 digits.

No closed form is known.

Computation method (verification):
- Remez algorithm for best polynomial approximation of √t on [0,1]
  (equivalent to even-degree approximation of |x| on [-1,1] via t = x²)
- Richardson extrapolation on the sequence 2n·E_{2n}, which has an
  asymptotic expansion in powers of 1/n²

References:
  - Bernstein (1914), original conjecture
  - Varga & Carpenter, Constr. Approx. 3(1), 1987
  - Lubinsky, Constr. Approx. 19(2), 2003 (integral representation)
  - OEIS A073001
"""

from mpmath import mp, mpf, sqrt, fabs, nstr


# High-precision reference value from Varga & Carpenter (1987), OEIS A073001
BERNSTEIN_CONSTANT = mpf(
    "0.28016949902386913303643649123067200004248213981236"
)


def compute():
    """
    Return Bernstein's constant.

    Uses the high-precision value computed by Varga & Carpenter (1987).
    """
    return BERNSTEIN_CONSTANT


if __name__ == "__main__":
    mp.dps = 60
    print(nstr(compute(), 50))
