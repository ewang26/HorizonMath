from mpmath import mp

mp.dps = 110

def compute():
    n = 7

    def f(t):
        k = mp.besselk(0, t)
        return t * (k ** n)

    # [0,1] with t = u^2 to smooth the logarithmic behavior of K0(t) at t=0
    def f0(u):
        t = u * u
        k = mp.besselk(0, t)
        return 2 * (u ** 3) * (k ** n)

    with mp.workdps(mp.dps + 50):
        I0 = mp.quad(f0, [0, 1])
        I1 = mp.quad(f, [1, 5, 15, 40, mp.inf])
        I = I0 + I1

        C7 = (2 ** n) * mp.factorial(n) * I

    return +C7

if __name__ == "__main__":
    print(str(compute()))