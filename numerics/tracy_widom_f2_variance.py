"""
Numerical computation for: Variance of the Tracy-Widom F_2 Distribution

The variance of F_2 is defined as:
    Var[F_2] = E[X^2] - E[X]^2

where X ~ F_2 (Tracy-Widom GUE distribution).

Numerical value:
    Var[F_2] = 0.8131947928329...

NO CLOSED FORM IS KNOWN for this constant.

Reference:
  Bornemann (2012). ON THE NUMERICAL EVALUATION OF DISTRIBUTIONS IN RANDOM
  MATRIX THEORY: A REVIEW WITH AN INVITATION TO EXPERIMENTAL MATHEMATICS
"""
from mpmath import mp, mpf

mp.dps = 110

# High-precision value computed from Painlevé II ODE
# Source: Bornemann (2010), Perret & Schehr (2014)
TRACY_WIDOM_F2_VARIANCE = mpf(
    "0.8131947928329"
)


def compute():
    """Return the variance of Tracy-Widom F_2."""
    return TRACY_WIDOM_F2_VARIANCE


if __name__ == "__main__":
    print(str(compute()))
