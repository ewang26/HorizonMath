from mpmath import mp

mp.dps = 110


def compute():
    # Kasner's nested radical constant: sqrt(1 + sqrt(2 + sqrt(3 + ...)))
    # OEIS A072449: 1.7579327566180045326...
    # Iterate downward from N: result = sqrt(N), then for k = N-1 down to 1: result = sqrt(k + result)
    with mp.extradps(30):
        N = 500
        result = mp.sqrt(N)
        for k in range(N - 1, 0, -1):
            result = mp.sqrt(k + result)
        return result


if __name__ == "__main__":
    print(str(compute()))
