from mpmath import mp

mp.dps = 110

def compute():
    # Prefer direct Stieltjes-constant computation if available
    if hasattr(mp, "stieltjes"):
        return mp.stieltjes(1)

    # Fallback: extract gamma_1 from the Laurent expansion of zeta(s) at s=1
    # zeta(s) = 1/(s-1) + gamma_0 - gamma_1*(s-1) + ...
    # Let f(s) = zeta(s) - 1/(s-1), then f'(1) = -gamma_1
    f = lambda s: mp.zeta(s) - 1/(s - 1)
    return -mp.diff(f, 1)

if __name__ == "__main__":
    print(str(compute()))