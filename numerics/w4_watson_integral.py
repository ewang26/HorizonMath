"""
Numerical computation for: Closed Form for the 4-Dimensional Lattice Green's Function (W_4)

The Watson integral W_4 is computed using a 1D integral representation
with Modified Bessel functions of the first kind.
"""
from mpmath import mp, quad, exp, besseli, inf

# Set precision to 110 decimal places to ensure 100 accurate digits
mp.dps = 110

def compute():
    """Compute W_4 numerically."""
    def integrand(t):
        # The integrand reduces the 4D integral to a 1D form
        # using Modified Bessel functions of the first kind.
        return exp(-4*t) * besseli(0, t)**4

    # Perform the integration from 0 to infinity
    w4_val = quad(integrand, [0, inf])
    return w4_val

if __name__ == "__main__":
    result = compute()
    print(str(result))
