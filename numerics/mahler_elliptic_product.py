import mpmath as mp

mp.mp.dps = 150


def abs_r1_minus_1(theta):
    """Auxiliary: |r1(theta)| - 1, used to locate kink points."""
    x = mp.exp(1j * theta)
    a = x + 1
    b = x**2 + 2*x + 2
    c = (x + 1)**2
    disc = b*b - 4*a*c
    sq = mp.sqrt(disc)
    r1 = (-b + sq) / (2*a)
    return abs(r1) - 1


def integrand(theta):
    """
    Jensen's formula applied to P(x,y) = (x+y+1)(x+1)(y+1) - xy
    viewed as a quadratic in y: a*y^2 + b*y + c with
      a = x+1,  b = x^2+2x+2,  c = (x+1)^2.

    Inner integral over y gives log|a| + log^+(|r1|) + log^+(|r2|).
    """
    x = mp.exp(1j * theta)
    a = x + 1
    b = x**2 + 2*x + 2
    c = (x + 1)**2

    if abs(a) < mp.mpf("1e-120"):
        # Degenerate case x = -1: P(-1,y) = y, average log|y| = 0
        r = -c / b
        return mp.log(abs(b)) + mp.log(max(1, abs(r)))

    disc = b*b - 4*a*c
    sq = mp.sqrt(disc)
    r1 = (-b + sq) / (2*a)
    r2 = (-b - sq) / (2*a)
    return mp.log(abs(a)) + mp.log(max(1, abs(r1))) + mp.log(max(1, abs(r2)))


def compute():
    with mp.workdps(mp.mp.dps + 40):
        # Locate the two theta values where |r1| = 1 (kink points).
        # These are symmetric: t2 = 2*pi - t1.
        t1 = mp.findroot(abs_r1_minus_1, mp.mpf("1.763"))
        t2 = 2 * mp.pi - t1

        # Integrate with breakpoints at all non-smooth points:
        #   theta=0 (disc=0), t1 (|r1|=1 kink), pi (log|x+1| singularity),
        #   t2 (|r1|=1 kink), 2*pi.
        val = mp.quad(
            lambda t: integrand(t), [0, t1, mp.pi, t2, 2 * mp.pi], maxdegree=14
        )
        return val / (2 * mp.pi)


if __name__ == "__main__":
    print(str(compute()))
