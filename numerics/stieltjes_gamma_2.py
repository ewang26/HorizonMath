from mpmath import mp

mp.dps = 110

def compute():
    # Stieltjes constant gamma_2 (coefficient in the Laurent expansion of zeta(s) at s=1)
    with mp.extradps(50):
        val = mp.stieltjes(2, 1)
    return +val

if __name__ == "__main__":
    print(str(compute()))