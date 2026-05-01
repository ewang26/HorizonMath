from mpmath import mp

def compute_c3(dps=220):
    mp.dps = dps
    f = lambda x: mp.e**x - (x**2 + x + 1)
    # unique root in (1,2)
    lam = mp.findroot(f, 2)
    c3 = lam + 1 + 1/lam
    return c3

if __name__ == "__main__":
    mp.dps = 220
    val = compute_c3(220)
    # print enough significant digits to store
    print(mp.nstr(val, 210))