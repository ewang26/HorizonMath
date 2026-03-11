from mpmath import mp

mp.dps = 110

def poly_mul(a, b, maxdeg):
    na = len(a)
    nb = len(b)
    nres = min(maxdeg, na + nb - 2) + 1
    res = [mp.mpf('0') for _ in range(nres)]
    for i in range(na):
        ai = a[i]
        if not ai:
            continue
        jmax = min(nb - 1, maxdeg - i)
        for j in range(jmax + 1):
            res[i + j] += ai * b[j]
    return res

def tail_asymptotic(A, N=200):
    # I0(t)*exp(-t) ~ (2*pi*t)^(-1/2) * sum_{k>=0} a[k]/t^k
    a = [mp.mpf('0') for _ in range(N + 1)]
    a[0] = mp.mpf(1)
    for k in range(1, N + 1):
        a[k] = a[k - 1] * ((2 * k - 1) ** 2) / (8 * k)

    maxdeg = 6 * N
    p = a  # polynomial in u = 1/t

    # p^6 via exponentiation: p2 = p^2, p4 = p^4, p6 = p^6
    p2 = poly_mul(p, p, maxdeg)
    p4 = poly_mul(p2, p2, maxdeg)
    p6 = poly_mul(p4, p2, maxdeg)  # coefficients b_k in (sum a_k u^k)^6

    u = mp.mpf(1) / A
    pow_u = u ** (maxdeg + 2)  # u^(k+2) for k=maxdeg initially
    s = mp.mpf('0')
    for k in range(maxdeg, -1, -1):
        s += p6[k] * pow_u / (k + 2)
        pow_u /= u

    C = mp.mpf(1) / (2 * mp.pi) ** 3
    return C * s

def compute():
    with mp.workdps(250):
        A = mp.mpf(200)

        def f(t):
            i0e = mp.besseli(0, t) * mp.e**(-t)
            return i0e**6

        num = mp.quad(f, [0, 1, 5, 20, 60, 120, A])
        tail = tail_asymptotic(A, N=200)
        res = num + tail

    return +res

if __name__ == "__main__":
    print(str(compute()))