from mpmath import mp

mp.dps = 110

def bloch_wigner(z):
    # D(z) = Im(Li_2(z)) + Arg(1-z)*log|z|
    # = Im(Li_2(z) + log(1-z)*log|z|)
    return mp.im(mp.polylog(2, z) + mp.log(1 - z) * mp.log(abs(z)))

def compute():
    with mp.extradps(30):
        # Find all roots of z^3 - z^2 + 1 = 0
        roots = mp.polyroots([1, -1, 0, 1])

        # Find the root in the upper half-plane (positive imaginary part)
        z = None
        for r in roots:
            if mp.im(r) > 0:
                z = r
                break

        if z is None:
            raise ValueError("No root found in upper half-plane")

        # Volume(5_2) = 3 * D(z)
        vol = 3 * bloch_wigner(z)
        return mp.re(vol)

if __name__ == "__main__":
    print(str(compute()))