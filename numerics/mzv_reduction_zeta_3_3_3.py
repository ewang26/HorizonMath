from mpmath import mp

mp.dps = 110

def compute():
    z3 = mp.zeta(3)
    z6 = mp.pi**6 / mp.mpf(945)  # exact: zeta(6)
    z9 = mp.zeta(9)
    return z3**3 / mp.mpf(6) - (z3 * z6) / mp.mpf(2) + z9 / mp.mpf(3)

if __name__ == "__main__":
    print(str(compute()))