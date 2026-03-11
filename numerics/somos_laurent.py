from mpmath import mp
from fractions import Fraction

mp.dps = 110

def compute():
    # Somos-4: a(n)*a(n-4) = a(n-1)*a(n-3) + a(n-2)^2, with a0=a1=a2=a3=1
    N = 25
    a = [1, 1, 1, 1]
    for n in range(4, N + 1):
        num = a[n - 1] * a[n - 3] + a[n - 2] * a[n - 2]
        den = a[n - 4]
        if num % den != 0:
            raise ValueError("Non-integer term encountered (unexpected for Somos-4 with these initials).")
        a.append(num // den)

    # Define y_n = a_{n+1} a_{n-1} / a_n^2 (a QRT reduction)
    # Representative n=10
    n = 10
    y_n = Fraction(a[n + 1] * a[n - 1], a[n] * a[n])
    y_np1 = Fraction(a[n + 2] * a[n], a[n + 1] * a[n + 1])

    # Invariant for the reduced map y_{n+1} y_{n-1} = (1 + y_n)/y_n^2:
    # K = y_{n-1}y_n + 1/y_{n-1} + 1/y_n + 1/(y_{n-1}y_n)
    u = y_n
    v = y_np1
    K = u * v + Fraction(1, u) + Fraction(1, v) + Fraction(1, u * v)

    # Convert K to mp
    Kmp = mp.mpf(K.numerator) / mp.mpf(K.denominator)

    # Elliptic curve from invariant level set:
    # y^2 = (K*s - s^2 - 1)^2 - 4*s = s^4 - 2K s^3 + (K^2+2)s^2 - (2K+4)s + 1
    c4 = mp.mpf(1)
    c3 = -2 * Kmp
    c2 = Kmp**2 + 2
    c1 = -(2 * Kmp + 4)
    c0 = mp.mpf(1)

    with mp.workdps(250):
        roots = mp.polyroots([c4, c3, c2, c1, c0])
        roots = sorted(roots, key=lambda z: mp.re(z))
        e = [mp.re(r) for r in roots]

        # Cross-ratio modulus (0<lambda<1 for ordered real roots e1<e2<e3<e4)
        e1, e2, e3, e4 = e
        lam = (e2 - e1) * (e4 - e3) / ((e3 - e1) * (e4 - e2))

        # j-invariant for Legendre form
        j = 256 * (1 - lam + lam**2)**3 / (lam**2 * (1 - lam)**2)

    return +j  # round to current mp.dps

if __name__ == "__main__":
    print(str(compute()))