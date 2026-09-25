"""
Reference numerical computation for: Feigenbaum Constant α

The Feigenbaum constant α governs the geometric scaling of the attractor in
period-doubling bifurcations. It is defined via the functional equation for
the universal function g(x) at the accumulation point of bifurcations:

    g(x) = -α · g(g(-x/α))

where g(0) = 1 and g'(0) = 0 (g has a quadratic maximum at 0).
The scaling factor α = 2.502907875095892822... is universal.
"""
from mpmath import mp, mpf

# Set precision to 110 decimal places
mp.dps = 110


def compute():
    """
    Return the Feigenbaum constant α.

    The constant can be computed via:
    1. The renormalization group fixed-point equation
    2. Measuring the scaling of superstable periodic orbits
    3. The width ratio of the attractor at successive period doublings

    For ground truth, we use the high-precision published value computed via
    renormalization group methods.

    The value has been computed to 1000+ digits by Briggs (1997) and others.
    """
    # Feigenbaum α computed to 100+ digits
    # Source: K. Briggs (1997), D. Broadhurst (1999)
    # Available here: https://oeis.org/A006891
    alpha = mpf(
        "2.50290787509589282228390287321821578638127137672714997733619205677923546317959020670329964974643383412959"
    )

    return alpha


if __name__ == "__main__":
    result = compute()
    print(str(result))
