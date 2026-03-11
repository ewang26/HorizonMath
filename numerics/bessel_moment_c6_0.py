from mpmath import mp

mp.dps = 110

def compute():
    # c_{6,0} = ∫_0^∞ K0(t)^6 dt
    # Split into (0,1) and (1,∞), using substitutions to avoid the t=0 endpoint:
    # ∫_0^1 f(t) dt with t = e^{-x}  => ∫_0^∞ f(e^{-x}) e^{-x} dx
    # ∫_1^∞ f(t) dt with t = 1 + u => ∫_0^∞ f(1+u) du
    with mp.workdps(160):
        f_small = lambda x: mp.besselk(0, mp.e**(-x))**6 * mp.e**(-x)
        f_large = lambda u: mp.besselk(0, 1 + u)**6

        I_small = mp.quad(f_small, [0, 10, 30, mp.inf])
        I_large = mp.quad(f_large, [0, 2, 6, mp.inf])

        return +(I_small + I_large)

if __name__ == "__main__":
    print(str(compute()))