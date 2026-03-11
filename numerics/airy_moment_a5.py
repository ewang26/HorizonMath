from mpmath import mp

mp.dps = 110

def compute():
    f = lambda x: mp.airyai(x) ** 5

    def integrate_cuts(cuts):
        s = mp.mpf("0")
        for a, b in zip(cuts[:-1], cuts[1:]):
            s += mp.quad(f, [a, b])
        return s

    cuts_a = [mp.mpf("0"), mp.mpf("1"), mp.mpf("4"), mp.mpf("10"), mp.mpf("20")]
    cuts_b = [mp.mpf("0"), mp.mpf("0.5"), mp.mpf("2"), mp.mpf("6"), mp.mpf("12"), mp.mpf("20")]

    # Compute with guard digits for reliable 100+ digit output
    with mp.workdps(220):
        Ia = integrate_cuts(cuts_a)
        Ib = integrate_cuts(cuts_b)
        I = (Ia + Ib) / 2

    return I

if __name__ == "__main__":
    print(str(compute()))