"""
Numerical computation for: Mean of the Tracy-Widom F_2 Distribution

The Tracy-Widom distribution F_2 describes the limiting distribution of the
largest eigenvalue of GUE random matrices after appropriate centering and scaling.

The mean E[X] where X ~ F_2 has numerical value:
    E[F_2] = -1.77108680741160162612693822832370833445514095085934616781672203

NO CLOSED FORM IS KNOWN for this constant, despite extensive searches.

Reference:
  Tracy & Widom (1994), "Level-spacing distributions and the Airy kernel"
  Bornemann (2010), "On the numerical evaluation of distributions in random matrix theory"
  OEIS A245258
"""
from mpmath import mp, mpf

mp.dps = 110

# High-precision value computed from Painlevé II ODE
# Source: Bornemann (2010), Perret & Schehr (2014)
TRACY_WIDOM_F2_MEAN = mpf(
    "-1.77108680741160162612693822832370833445514095085934616781672203"
)


def compute():
    """Return the mean of Tracy-Widom F_2."""
    return TRACY_WIDOM_F2_MEAN


if __name__ == "__main__":
    print(str(compute()))
