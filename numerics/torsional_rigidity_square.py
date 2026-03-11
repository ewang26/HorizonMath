from mpmath import mp

mp.dps = 110


def compute():
    # Dimensionless torsional rigidity ratio J/a^4 for a square cross-section.
    # Saint-Venant formula:
    #   J/a^4 = (16/3) * [1 - (192/pi^5) * sum_{n=0}^{infty} tanh((2n+1)*pi/2) / (2n+1)^5]
    # The series converges exponentially fast: tanh -> 1 exponentially,
    # combined with 1/(2n+1)^5 decay.
    # Reference: Timoshenko & Goodier, "Theory of Elasticity" (1951)
    # Value approx 2.2492322392...
    with mp.extradps(30):
        S = mp.mpf("0")
        for n in range(200):
            k = 2 * n + 1
            term = mp.tanh(k * mp.pi / 2) / mp.power(k, 5)
            S += term
            if abs(term) < mp.mpf("1e-150"):
                break

        result = (mp.mpf(16) / 3) * (1 - (mp.mpf(192) / mp.pi ** 5) * S)
        return result


if __name__ == "__main__":
    print(str(compute()))
