from mpmath import mp

mp.dps = 110


def bloch_wigner(z):
    # D(z) = Im(Li_2(z)) + Arg(1-z)*log|z|
    return mp.im(mp.polylog(2, z) + mp.log(1 - z) * mp.log(abs(z)))


def compute():
    # Hyperbolic volume of the 7_2 knot complement.
    # The 7_2 knot is a twist knot (two-bridge knot K(11,5)).
    #
    # Approach: Solve the gluing equations of the ideal triangulation obtained
    # from SnapPy (4 tetrahedra, triangulation code "evQkbccddtnrnj_BbDc").
    # Starting from SnapPy's 60-digit shape parameters, refine to 110+ digits
    # via Newton's method on the log-form gluing equations.
    #
    # Gluing equations from SnapPy (format: A_vec, B_vec, sign):
    #   Eq 0: ([1,2,0,0], [-1,0,1,0], -1)
    #   Eq 1: ([0,-1,1,-2], [-1,1,0,2], -1)
    #   Eq 2: ([0,-1,-1,1], [1,-1,0,0], -1)
    #   Eq 3: ([-1,0,0,1], [1,0,-1,-2], -1)
    #   Eq 4: ([0,-1,0,0], [0,0,-1,0], 1)     # meridian
    #
    # We use equations 0,1,2,4 (3 independent edge + 1 cusp completeness).

    with mp.extradps(30):
        # Starting shape parameters from SnapPy high_precision (60 digits)
        z = [
            mp.mpc(
                "0.979683927137063080360443583225912498526944739792254472909696",
                "0.590569559841547738085433207813503541833670692235462901341630",
            ),
            mp.mpc(
                "0.251322701057396787068916574052517527698543073419837511877978",
                "0.451314970729364036154899986170441362413612486336944204016703",
            ),
            mp.mpc(
                "0.05818137738476620957186092260681916651032819794670750704818",
                "1.69127914951419451109509131997221641885831120673024304031914",
            ),
            mp.mpc(
                "1.16369117147491476375354246222499900315270704909808869777148",
                "0.56418563226878988033974884693917445186365596844491528772036",
            ),
        ]

        # Gluing equation exponents (using equations 0,1,2,4)
        A = [
            [1, 2, 0, 0],
            [0, -1, 1, -2],
            [0, -1, -1, 1],
            [0, -1, 0, 0],
        ]
        B = [
            [-1, 0, 1, 0],
            [-1, 1, 0, 2],
            [1, -1, 0, 0],
            [0, 0, -1, 0],
        ]
        signs = [-1, -1, -1, 1]

        # Determine target values from approximate solution
        targets = []
        for i in range(4):
            val = sum(A[i][j] * mp.log(z[j]) + B[i][j] * mp.log(1 - z[j])
                      for j in range(4))
            # Round to nearest multiple of pi*i
            k = round(float(mp.im(val) / mp.pi))
            targets.append(mp.mpc(0, k * mp.pi))

        # Newton's method to refine shapes to full precision
        for iteration in range(10):
            # Evaluate residuals
            g = []
            for i in range(4):
                val = sum(A[i][j] * mp.log(z[j]) + B[i][j] * mp.log(1 - z[j])
                          for j in range(4))
                g.append(val - targets[i])

            # Check convergence
            max_err = max(abs(gi) for gi in g)
            if max_err < mp.mpf(10) ** (-(mp.dps + 20)):
                break

            # Compute Jacobian (4x4 complex matrix)
            J = mp.matrix(4, 4)
            for i in range(4):
                for j in range(4):
                    J[i, j] = A[i][j] / z[j] - B[i][j] / (1 - z[j])

            # Solve J * dz = -g
            g_vec = mp.matrix([g[0], g[1], g[2], g[3]])
            dz = mp.lu_solve(J, -g_vec)

            # Update shape parameters
            for j in range(4):
                z[j] += dz[j]

        # Compute volume as sum of Bloch-Wigner values
        vol = sum(bloch_wigner(zi) for zi in z)
        return mp.re(vol)


if __name__ == "__main__":
    print(str(compute()))
