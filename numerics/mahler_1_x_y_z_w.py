from mpmath import mp

mp.dps = 110

_dilog = getattr(mp, "dilog", None)
if _dilog is None:
    def _dilog(z):
        return mp.polylog(2, z)


def F_truncated_avg_log(r):
    """
    F(r) = (1/2pi) int_0^{2pi} log^+(|r + e^{it}|) dt,  r >= 0
    Closed form:
      - for r >= 2: log(r)
      - for 0 <= r < 2:  -(1/pi) * Im( Li_2( -r * exp(i*acos(-r/2)) ) )
    """
    r = mp.mpf(r)
    if r <= 0:
        return mp.zero
    if r >= 2:
        return mp.log(r)
    phi = mp.acos(-r / 2)
    z = -r * mp.exp(1j * phi)
    return -mp.im(_dilog(z)) / mp.pi


def kink_condition(a, b):
    """
    Returns |1 + e^{ia} + e^{ib}|^2 - 4.
    The kink of F occurs at r = 2, i.e., where this equals 0.
    |1 + e^{ia} + e^{ib}|^2 = 3 + 2*(cos a + cos b + cos(a-b)).
    """
    return mp.mpf(3) + 2 * (mp.cos(a) + mp.cos(b) + mp.cos(a - b)) - 4


def find_kink_b_values(a):
    """
    For a given a, find all b in [0, 2*pi) where |1+e^{ia}+e^{ib}| = 2.
    This requires: cos(a) + cos(b) + cos(a-b) = 1/2.
    Let u = cos(a), s = sin(a).
    cos(b) + cos(a-b) = cos(b) + u*cos(b) + s*sin(b) = (1+u)*cos(b) + s*sin(b)
    So: u + (1+u)*cos(b) + s*sin(b) = 1/2
    i.e., (1+u)*cos(b) + s*sin(b) = 1/2 - u
    This is A*cos(b) + B*sin(b) = C with A=(1+u), B=s, C=(1/2-u).
    Solutions exist iff C^2 <= A^2 + B^2.
    """
    u = mp.cos(a)
    s = mp.sin(a)
    A = 1 + u
    B = s
    C = mp.mpf("0.5") - u
    R2 = A * A + B * B
    if C * C > R2:
        return []
    R = mp.sqrt(R2)
    # A*cos(b) + B*sin(b) = R*cos(b - phi) where phi = atan2(B, A)
    phi = mp.atan2(B, A)
    cos_val = C / R
    if abs(cos_val) > 1:
        return []
    delta = mp.acos(cos_val)
    b1 = phi + delta
    b2 = phi - delta
    # Normalize to [0, 2*pi)
    twopi = 2 * mp.pi
    b1 = b1 % twopi
    b2 = b2 % twopi
    # Return sorted unique values
    if abs(b1 - b2) < mp.mpf("1e-100"):
        return [b1]
    return sorted([b1, b2])


def inner_integrand(a, b):
    """F(|1 + e^{ia} + e^{ib}|) for given a, b."""
    ca = mp.cos(a)
    sa = mp.sin(a)
    cb = mp.cos(b)
    sb = mp.sin(b)
    r2 = (1 + ca + cb) ** 2 + (sa + sb) ** 2
    if r2 < 0:
        r2 = mp.zero
    r = mp.sqrt(r2)
    return F_truncated_avg_log(r)


def inner_integral(a):
    """
    Compute int_0^{2*pi} F(|1+e^{ia}+e^{ib}|) db
    with breakpoints at the kink locations (where |1+e^{ia}+e^{ib}| = 2).
    """
    twopi = 2 * mp.pi
    kinks = find_kink_b_values(a)
    breakpoints = [mp.zero] + kinks + [twopi]
    return mp.quad(lambda b: inner_integrand(a, b), breakpoints, maxdegree=14)


def compute():
    # m(1+x+y+z+w) = (1/(2pi)^2) int_0^{2pi} int_0^{2pi} F(|1+e^{ia}+e^{ib}|) db da
    # where F integrates out the two remaining variables z, w.
    #
    # F has a kink at r=2. We split the inner integral at the kink curve
    # and use mpmath's adaptive Gauss-Legendre quadrature for each smooth segment.
    with mp.workdps(mp.dps + 40):
        # The outer integrand (inner_integral) is itself smooth in a
        # (the kink locations vary smoothly with a), but has kinks at
        # a values where the number of kink b-values changes (tangency points).
        # These occur where the discriminant of the kink equation vanishes.
        # A^2 + B^2 = C^2 at cos(a)+cos(b)+cos(a-b)=1/2 tangency.
        # For simplicity, split the outer integral at a=0, pi, 2pi and
        # at the critical a values where kinks appear/disappear.

        # Find critical a values: (1+u)^2 + s^2 = (1/2-u)^2
        # 1 + 2u + u^2 + 1 - u^2 = 1/4 - u + u^2
        # 2 + 2u = 1/4 - u + u^2
        # u^2 - 3u - 7/4 = 0
        # u = (3 +/- sqrt(9+7))/2 = (3 +/- 4)/2
        # u = 7/2 (impossible for cos) or u = -1/2
        # So cos(a) = -1/2, i.e. a = 2*pi/3 or 4*pi/3
        a_crit1 = 2 * mp.pi / 3
        a_crit2 = 4 * mp.pi / 3

        val = mp.quad(
            inner_integral,
            [0, a_crit1, mp.pi, a_crit2, 2 * mp.pi],
            maxdegree=14,
        )
        return val / (4 * mp.pi ** 2)


if __name__ == "__main__":
    print(str(compute()))
