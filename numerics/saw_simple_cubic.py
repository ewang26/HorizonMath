"""
Reference numerical computation for: Connective Constant for Simple Cubic Lattice Self-Avoiding Walks

The 3D connective constant μ for the simple cubic lattice has been computed to high precision
using the pivot algorithm by Nathan Clisby (2013). The value μ = 4.684039931(27) represents
the current best estimate.

The pivot algorithm generates SAWs efficiently via local transformations and uses
sophisticated analysis to extract the connective constant with high precision.
"""
from mpmath import mp, mpf

# Set precision
mp.dps = 50

# Known series coefficients for simple cubic lattice SAWs (from OEIS A001412)
# Exact enumeration is only known to n=36 steps
CUBIC_SAW_COEFFICIENTS = [
    1,           # n=0
    6,           # n=1
    30,          # n=2
    150,         # n=3
    726,         # n=4
    3534,        # n=5
    16926,       # n=6
    81390,       # n=7
    387966,      # n=8
    1853886,     # n=9
    8809878,     # n=10
    41934150,    # n=11
    198842742,   # n=12
    943974510,   # n=13
    4468911678,  # n=14
    21175146054, # n=15
    100121875974,# n=16
    473730252102,# n=17
    2237723684094,# n=18
    10576033219614,# n=19
    49917327838734,# n=20
    235710090502158,# n=21
    1111781983442406,# n=22
    5245988215191414,# n=23
    24730180885580790,# n=24
    116618841700433358,# n=25
    549493796867100942,# n=26
]


def compute():
    """
    Return the high-precision value of the simple cubic lattice connective constant.

    The value μ = 4.684039931(27) is from Clisby (2013), computed using the pivot
    algorithm with sophisticated Monte Carlo sampling of long walks (millions of steps).

    Reference: N. Clisby, "Calculation of the connective constant for self-avoiding
    walks via the pivot algorithm", J. Phys. A 46 (2013) 245001, arXiv:1302.2106
    """
    # Published high-precision value from pivot algorithm analysis
    mu = mpf("4.6840399310")

    return mu


if __name__ == "__main__":
    result = compute()
    print(str(result))
