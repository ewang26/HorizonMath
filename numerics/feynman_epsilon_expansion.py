from mpmath import mp

mp.dps = 110

def compute():
    # ε^1 coefficient = 9*zeta(4) = pi^4/10
    return 9 * mp.zeta(4)

if __name__ == "__main__":
    print(str(compute()))