from mpmath import mp

mp.dps = 110

def compute():
    mp.dps = 150
    k = mp.mpf(5)
    a = [mp.mpf(4)/3, mp.mpf(5)/3, 1, 1]
    b = [2, 2, 2]
    z = mp.mpf(27) / k**3
    return mp.log(k) - (mp.mpf(2) / k**3) * mp.hyper(a, b, z)

if __name__ == "__main__":
    print(mp.nstr(compute(), 120))
