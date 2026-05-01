from mpmath import mp

mp.dps = 200

def compute():
    return mp.euler   # Euler–Mascheroni constant

if __name__ == "__main__":
    print(mp.nstr(compute(), 180))