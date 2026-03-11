from mpmath import mp


mp.dps = 110


def compute():
    """
    Closed form for B_5(-2) from Borwein, Chan, Crandall (2010),
    "Higher-dimensional box integrals", Experimental Mathematics 19(4).

    B_5(-2) = (5/3) K_5 + (5/6) pi G - (5/12) pi^2 log(1+sqrt(2))
              - (5/6) pi Ti_2(3 - 2 sqrt(2)) + (10/3) C_{3,0}(-2, 2)

    where:
      K_5 = J(3) = int_[0,1]^2 log(3+x^2+y^2)/((1+x^2)(1+y^2)) dx dy
      G = Catalan's constant
      Ti_2(x) = int_0^x arctan(t)/t dt  (inverse tangent integral)
      C_{3,0}(-2, 2) = int_[0,1]^3 1/(2+x^2+y^2+z^2) dx dy dz
                      = int_[0,1]^2 arctan(1/sqrt(2+x^2+y^2))/sqrt(2+x^2+y^2) dx dy

    Derived via recurrence (1.11) with n=5, s=-2 and the known closed form
    for B_5(-4) from the same paper.
    """
    with mp.workdps(220):
        pi = mp.pi
        G = mp.catalan
        sqrt2 = mp.sqrt(2)

        # K_5 = J(3): 2D integral
        def j_integrand(x, y):
            return mp.log(3 + x**2 + y**2) / ((1 + x**2) * (1 + y**2))

        K5 = mp.quad(j_integrand, [0, 1], [0, 1])

        # Ti_2(x) = inverse tangent integral = int_0^x arctan(t)/t dt
        arg = 3 - 2 * sqrt2
        Ti2_val = mp.quad(lambda t: mp.atan(t) / t, [0, arg])

        # C_{3,0}(-2, 2): reduce 3D to 2D by integrating out z analytically
        # int_0^1 dz/(a+z^2) = arctan(1/sqrt(a))/sqrt(a)
        def c30_integrand(x, y):
            a = 2 + x**2 + y**2
            sa = mp.sqrt(a)
            return mp.atan(1 / sa) / sa

        C30 = mp.quad(c30_integrand, [0, 1], [0, 1])

        result = (
            mp.mpf(5) / 3 * K5
            + mp.mpf(5) / 6 * pi * G
            - mp.mpf(5) / 12 * pi**2 * mp.log(1 + sqrt2)
            - mp.mpf(5) / 6 * pi * Ti2_val
            + mp.mpf(10) / 3 * C30
        )

        return result


if __name__ == "__main__":
    print(str(compute()))
