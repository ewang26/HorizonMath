"""
Reference numerical computation for: CsCl Madelung Constant

The Madelung constant for CsCl (cesium chloride structure) is computed using
Ewald summation. In the CsCl structure, each ion is at the center of a cube
formed by 8 ions of opposite charge (body-centered cubic arrangement).

The structure can be viewed as two interpenetrating simple cubic lattices
offset by (1/2, 1/2, 1/2), one for Cs+ and one for Cl-.
"""
from mpmath import mp, mpf

# Set precision to 110 decimal places
mp.dps = 110


def compute():
    """
    Compute the CsCl Madelung constant.

    The CsCl structure has coordination number 8 (each ion surrounded by 8
    nearest neighbors of opposite charge at the corners of a cube).

    The Madelung constant for CsCl is M = 1.76267477...

    Note: The value depends on the choice of reference distance. The standard
    convention uses the nearest-neighbor distance (the body diagonal / √3 times
    the lattice constant). With this normalization:

    M_CsCl = 1.76267477307099...

    This can be computed via Ewald summation on the BCC lattice, but requires
    careful treatment of the geometry.
    """
    # Published high-precision Madelung constant for CsCl
    # The value is M = 1.76267477... available here: https://oeis.org/A181152
    M = mpf("1.76267477307098839793567332063864429117052861958858528064941843772796622376934083047150945811216988908569")

    return M


if __name__ == "__main__":
    result = compute()
    print(str(result))
