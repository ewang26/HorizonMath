from mpmath import mp

mp.dps = 110

def compute():
    # Representative value (k = 10) for the Erdős random-graph diagonal Ramsey lower-bound scale:
    # r(k,k) ≳ (k / (e*sqrt(2))) * 2^(k/2)
    k = mp.mpf(10)
    result = (k * mp.power(2, k / 2)) / (mp.e * mp.sqrt(2))
    return result

if __name__ == "__main__":
    print(str(compute()))