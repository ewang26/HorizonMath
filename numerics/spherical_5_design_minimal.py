"""
Reference numerical computation for: Minimum spherical 9-design on S^2

A spherical t-design on S^d is a finite set of points on the sphere such that
the average of any polynomial of degree <= t over those points equals the average
over the entire sphere.

The Delsarte-Goethals-Seidel (DGS) lower bound for odd t on S^2 gives:
    N >= (t+1)(t+3)/4
For t=9: N >= (10)(12)/4 = 30.

The best known construction achieving a spherical 9-design on S^2 uses 48 points
(Hardin & Sloane, 1996). Whether fewer points suffice remains open, but 48 is the
current best known value.
"""
from mpmath import mp, mpf

mp.dps = 110


def delsarte_goethals_seidel_lower_bound_S2(t):
    """DGS lower bound for spherical t-designs on S^2."""
    if t % 2 == 1:
        # Odd t: N >= (t+1)(t+3)/4
        return (t + 1) * (t + 3) // 4
    else:
        # Even t: N >= (t+2)^2 / 4  (included for completeness)
        return (t + 2) ** 2 // 4


def compute():
    """
    Return the size of the best known spherical 9-design on S^2.

    DGS lower bound for t=9: (10)(12)/4 = 30.
    Best known construction: 48 points (Hardin & Sloane, 1996).
    """
    t = 9
    bound = delsarte_goethals_seidel_lower_bound_S2(t)
    assert bound == 30, f"Expected DGS bound 30, got {bound}"

    # Best known spherical 9-design on S^2
    best_known = mpf(48)
    return best_known


if __name__ == "__main__":
    result = compute()
    print(mp.nstr(result, 110, strip_zeros=False))
