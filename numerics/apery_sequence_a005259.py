from mpmath import mp

mp.dps = 110

def apery_hyper(n):
    # A005259(n) = 4F3(-n, -n, n+1, n+1; 1, 1, 1; 1)
    return mp.hyper([ -n, -n, n + 1, n + 1 ], [1, 1, 1], 1)

def apery_recurrence(n):
    # (m+1)^3 a_{m+1} = (34 m^3 + 51 m^2 + 27 m + 5) a_m - m^3 a_{m-1}
    if n == 0:
        return 1
    if n == 1:
        return 5
    a_prev = 1
    a_cur = 5
    for m in range(1, n):
        num = (34*m**3 + 51*m**2 + 27*m + 5) * a_cur - (m**3) * a_prev
        den = (m + 1) ** 3
        a_next = num // den
        a_prev, a_cur = a_cur, a_next
    return a_cur

def compute():
    n = 10
    a_exact = apery_recurrence(n)  # exact integer
    a_hyp = apery_hyper(n)         # high-precision hypergeometric evaluation

    # sanity check: hypergeometric value should match the exact integer
    if abs(a_hyp - mp.mpf(a_exact)) > mp.mpf('1e-90'):
        raise ValueError("Consistency check failed")

    return a_exact

if __name__ == "__main__":
    print(str(compute()))