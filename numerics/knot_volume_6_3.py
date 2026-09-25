from mpmath import mp

mp.dps = 110

def bloch_wigner_D(z):
    # Bloch-Wigner dilogarithm:
    # D(z) = Im(Li_2(z)) + arg(1-z)*log|z|
    return mp.im(mp.polylog(2, z)) + mp.arg(1 - z) * mp.log(abs(z))

def compute():
    z = (mp.mpf(3) + 1j * mp.sqrt(7)) / 4
    vol = mp.mpf(6) * bloch_wigner_D(z)
    return mp.re(vol)

if __name__ == "__main__":
    print(str(compute()))