from mpmath import mp

mp.dps = 110

def compute():
    k = 6
    # b_n = 1/binomial(2n,n), with recurrence:
    # b_1 = 1/2
    # b_n = b_{n-1} * n / (2*(2n-1))
    b = mp.mpf(1) / 2
    terms = [b]  # n=1 term: b_1 / 1^6

    # Truncation target far below 1e-100; tail is < (4/3)*last_term since ratio < 1/4
    tol = mp.power(10, -(mp.dps + 15))

    n = 1
    while True:
        n += 1
        b *= mp.mpf(n) / (2 * (2 * n - 1))
        term = b / (mp.mpf(n) ** k)
        terms.append(term)

        if term < tol and (mp.mpf(4) / 3) * term < tol:
            break
        if n > 100000:
            raise RuntimeError("Failed to converge fast enough")

    return mp.fsum(terms)

if __name__ == "__main__":
    print(str(compute()))