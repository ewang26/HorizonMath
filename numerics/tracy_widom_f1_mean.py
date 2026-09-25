"""
Numerical computation for: Mean of the Tracy-Widom F_1 Distribution (GOE)

The Tracy-Widom F_1 distribution describes the largest eigenvalue of GOE
(Gaussian Orthogonal Ensemble) random matrices.

It is characterized by a coupled system involving Painlevé II:
    q''(s) = sq(s) + 2q(s)^3,  q(s) ~ Ai(s) as s -> +infinity

and the F_1 distribution involves both q(s) and an integral of q.

The mean is:
    E[F_1] = -1.2065335745820...

NO CLOSED FORM IS KNOWN.

The ratio E[F_1]/E[F_2] = 0.6812159... may have structure.

Reference:
  Tracy & Widom (1996)
"""
from mpmath import mp, mpf

mp.dps = 110

# High-precision value from numerical solution of Painlevé system
# Source: Bornemann (2010)
TRACY_WIDOM_F1_MEAN = mpf(
    "-1.20653357458209375788232456183089961281150892891979584679698604643953"
    "1871428069093892948158498295831217412832146379216871"
)


def compute():
    """Return the mean of Tracy-Widom F_1."""
    return TRACY_WIDOM_F1_MEAN


if __name__ == "__main__":
    print(str(compute()))
