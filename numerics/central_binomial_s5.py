from mpmath import mp

mp.dps = 110

def compute():
    # S_5 = sum_{n>=1} 1/(n^5 * binom(2n,n))
    # Use recurrence for a_n = 1/binom(2n,n): a_{n+1} = a_n * (n+1)/(4n+2)
    target = mp.eps * mp.mpf('1e-20')
    r_upper = mp.mpf('0.251')  # safely above the true term ratio (< 1/4)

    s = mp.mpf('0')
    a = mp.mpf('0.5')  # a_1 = 1/binom(2,1)
    n = 1

    while True:
        t = a / (n**5)
        s += t

        # remainder bound assuming geometric ratio <= r_upper:
        # R_n = sum_{k>=1} t_{n+k} <= t_n * r_upper/(1-r_upper)
        if t * r_upper / (1 - r_upper) < target:
            break

        a *= mp.mpf(n + 1) / mp.mpf(4 * n + 2)
        n += 1
        if n > 200000:
            raise RuntimeError("Convergence failure")

    return s

if __name__ == "__main__":
    print(mp.nstr(compute(), mp.dps))