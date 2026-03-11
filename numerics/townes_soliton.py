"""
Reference numerical computation for: Townes Soliton Critical Mass

Computes N_c = 2*pi * integral_0^inf Q(r)^2 * r dr, where Q(r) is the unique
positive radial solution of:

    Q''(r) + (1/r)*Q'(r) - Q(r) + Q(r)^3 = 0,  r > 0
    Q'(0) = 0,  Q(r) -> 0 as r -> infinity.

This is the radial reduction of Delta Q - Q + Q^3 = 0 in R^2 (the ground state
of the 2D focusing cubic NLS).  The critical mass N_c determines the sharp
constant in the 2D Gagliardo-Nirenberg inequality (Weinstein, 1983).

Method:
  1. Taylor series in r^2 around r=0 (80 terms, converges for |r| < ~1).
  2. High-order Taylor stepping for ODE integration (order 55, step 0.15).
  3. K_0 asymptotic matching at r = R_MATCH: the linearized equation has
     decaying K_0 and growing I_0 modes. Q(0) is found by requiring zero
     I_0 coefficient via the Wronskian criterion Q*K_0' - Q'*K_0 = 0.
  4. Bisection at small R_MATCH for initial estimate, then Newton's method
     with the variational equation at large R_MATCH for precision.
  5. Exact Taylor-based integration of Q^2*r, plus K_0 tail correction.
"""

from mpmath import mp

mp.dps = 110


def compute():
    with mp.extradps(50):
        # ==================================================================
        # Taylor series of Q around r = 0 in powers of r^2.
        #   Q(r) = c[0] + c[1]*r^2 + c[2]*r^4 + ...
        # Recurrence: c[k+1] = (c[k] - [Q^3]_k) / (2(k+1))^2
        # ==================================================================
        def r0_taylor(a, nterms):
            c = [mp.mpf(0)] * nterms
            c[0] = a
            for k in range(nterms - 1):
                q3k = mp.mpf(0)
                for j1 in range(k + 1):
                    s = mp.mpf(0)
                    for j2 in range(k - j1 + 1):
                        s += c[j2] * c[k - j1 - j2]
                    q3k += c[j1] * s
                c[k + 1] = (c[k] - q3k) / mp.mpf((2 * (k + 1)) ** 2)
            return c

        # Variational equation P'' + P'/r - P + 3*Q^2*P = 0, P(0)=1, P'(0)=0.
        def r0_taylor_var(c, nterms):
            d = [mp.mpf(0)] * nterms
            d[0] = mp.mpf(1)
            q2 = [mp.mpf(0)] * nterms
            for m in range(nterms):
                s = mp.mpf(0)
                for j in range(m + 1):
                    s += c[j] * c[m - j]
                q2[m] = s
            for k in range(nterms - 1):
                q2p = mp.mpf(0)
                for j in range(k + 1):
                    q2p += q2[j] * d[k - j]
                d[k + 1] = (d[k] - 3 * q2p) / mp.mpf((2 * (k + 1)) ** 2)
            return d

        def eval_poly(c, x):
            val = c[-1]
            for k in range(len(c) - 2, -1, -1):
                val = val * x + c[k]
            return val

        def eval_r0(c, r):
            return eval_poly(c, r ** 2)

        def eval_r0_deriv(c, r):
            dc = [2 * (k + 1) * c[k + 1] for k in range(len(c) - 1)]
            return r * eval_poly(dc, r ** 2)

        # ==================================================================
        # Taylor stepping for the ODE system (Q and P simultaneously).
        # Q'' = -Q'/r + Q - Q^3
        # P'' = -P'/r + P - 3*Q^2*P
        # Expands 1/r around r0 and computes all coefficients recursively.
        # ==================================================================
        def taylor_step_qp(r0, Q0, Qp0, P0, Pp0, order):
            a = [mp.mpf(0)] * (order + 1)
            e = [mp.mpf(0)] * (order + 1)
            a[0] = Q0; a[1] = Qp0
            e[0] = P0; e[1] = Pp0

            inv_r0 = mp.mpf(1) / r0
            b = [mp.mpf(0)] * (order + 1)
            bk = inv_r0
            for k in range(order + 1):
                b[k] = bk
                bk *= -inv_r0

            q2 = [mp.mpf(0)] * (order + 1)
            q3 = [mp.mpf(0)] * (order + 1)
            q2p = [mp.mpf(0)] * (order + 1)
            q2[0] = a[0] * a[0]
            q3[0] = a[0] * q2[0]
            q2p[0] = q2[0] * e[0]
            q2[1] = 2 * a[0] * a[1]
            q3[1] = a[0] * q2[1] + a[1] * q2[0]
            q2p[1] = q2[0] * e[1] + q2[1] * e[0]

            for n in range(order - 1):
                qp_r = mp.mpf(0)
                for j in range(n + 1):
                    qp_r += (j + 1) * a[j + 1] * b[n - j]
                a[n + 2] = (-qp_r + a[n] - q3[n]) / ((n + 2) * (n + 1))

                pp_r = mp.mpf(0)
                for j in range(n + 1):
                    pp_r += (j + 1) * e[j + 1] * b[n - j]
                e[n + 2] = (-pp_r + e[n] - 3 * q2p[n]) / ((n + 2) * (n + 1))

                m = n + 2
                s2 = mp.mpf(0)
                for j in range(m + 1):
                    s2 += a[j] * a[m - j]
                q2[m] = s2
                s3 = mp.mpf(0)
                for j in range(m + 1):
                    s3 += a[j] * q2[m - j]
                q3[m] = s3
                sp = mp.mpf(0)
                for j in range(m + 1):
                    sp += q2[j] * e[m - j]
                q2p[m] = sp

            return a, e

        # Q-only stepping (for the final integral computation)
        def taylor_step_q(r0, Q0, Qp0, order):
            a = [mp.mpf(0)] * (order + 1)
            a[0] = Q0; a[1] = Qp0

            inv_r0 = mp.mpf(1) / r0
            b = [mp.mpf(0)] * (order + 1)
            bk = inv_r0
            for k in range(order + 1):
                b[k] = bk
                bk *= -inv_r0

            q2 = [mp.mpf(0)] * (order + 1)
            q3 = [mp.mpf(0)] * (order + 1)
            q2[0] = a[0] * a[0]
            q3[0] = a[0] * q2[0]
            q2[1] = 2 * a[0] * a[1]
            q3[1] = a[0] * q2[1] + a[1] * q2[0]

            for n in range(order - 1):
                qp_r = mp.mpf(0)
                for j in range(n + 1):
                    qp_r += (j + 1) * a[j + 1] * b[n - j]
                a[n + 2] = (-qp_r + a[n] - q3[n]) / ((n + 2) * (n + 1))
                m = n + 2
                s2 = mp.mpf(0)
                for j in range(m + 1):
                    s2 += a[j] * a[m - j]
                q2[m] = s2
                s3 = mp.mpf(0)
                for j in range(m + 1):
                    s3 += a[j] * q2[m - j]
                q3[m] = s3

            return a

        # ==================================================================
        # Parameters
        # ==================================================================
        NTERMS_R0 = 80
        ORDER = 55
        R_START = mp.mpf('0.5')
        STEP = mp.mpf('0.15')

        # ==================================================================
        # Integrate Q (and optionally P) from 0 to R_max.
        # Returns (Q, Q', [P, P',] integral) at R_max.
        # ==================================================================
        def integrate_qp(a_val, R_max):
            c = r0_taylor(a_val, NTERMS_R0)
            d = r0_taylor_var(c, NTERMS_R0)
            Q_val = eval_r0(c, R_START)
            Qp_val = eval_r0_deriv(c, R_START)
            P_val = eval_r0(d, R_START)
            Pp_val = eval_r0_deriv(d, R_START)

            # Integral from 0 to R_START (exact from Taylor series)
            integral = mp.mpf(0)
            for j in range(NTERMS_R0):
                if abs(c[j]) < mp.power(10, -160):
                    break
                for k in range(NTERMS_R0):
                    if abs(c[k]) < mp.power(10, -160):
                        break
                    integral += (c[j] * c[k]
                                 * R_START ** (2 * (j + k) + 2)
                                 / (2 * (j + k) + 2))

            # Taylor stepping from R_START to R_max
            r = R_START
            while r < R_max - STEP / 2:
                h = min(STEP, R_max - r)
                ta, te = taylor_step_qp(r, Q_val, Qp_val, P_val, Pp_val, ORDER)

                # Integrate Q^2*(r+s) from s=0 to h
                q2_ta = [mp.mpf(0)] * (ORDER + 1)
                for n in range(ORDER + 1):
                    s = mp.mpf(0)
                    for j in range(n + 1):
                        s += ta[j] * ta[n - j]
                    q2_ta[n] = s
                h_pow = h
                for n in range(ORDER + 1):
                    term = r * q2_ta[n]
                    if n > 0:
                        term += q2_ta[n - 1]
                    integral += term * h_pow / (n + 1)
                    h_pow *= h

                Q_val = eval_poly(ta, h)
                P_val = eval_poly(te, h)
                Qp_val = mp.mpf(0)
                Pp_val = mp.mpf(0)
                h_pow = mp.mpf(1)
                for k in range(ORDER):
                    Qp_val += (k + 1) * ta[k + 1] * h_pow
                    Pp_val += (k + 1) * te[k + 1] * h_pow
                    h_pow *= h

                r += h

            return Q_val, Qp_val, P_val, Pp_val, integral

        def integrate_q(a_val, R_max):
            c = r0_taylor(a_val, NTERMS_R0)
            Q_val = eval_r0(c, R_START)
            Qp_val = eval_r0_deriv(c, R_START)

            integral = mp.mpf(0)
            for j in range(NTERMS_R0):
                if abs(c[j]) < mp.power(10, -160):
                    break
                for k in range(NTERMS_R0):
                    if abs(c[k]) < mp.power(10, -160):
                        break
                    integral += (c[j] * c[k]
                                 * R_START ** (2 * (j + k) + 2)
                                 / (2 * (j + k) + 2))

            r = R_START
            while r < R_max - STEP / 2:
                h = min(STEP, R_max - r)
                ta = taylor_step_q(r, Q_val, Qp_val, ORDER)

                q2_ta = [mp.mpf(0)] * (ORDER + 1)
                for n in range(ORDER + 1):
                    s = mp.mpf(0)
                    for j in range(n + 1):
                        s += ta[j] * ta[n - j]
                    q2_ta[n] = s
                h_pow = h
                for n in range(ORDER + 1):
                    term = r * q2_ta[n]
                    if n > 0:
                        term += q2_ta[n - 1]
                    integral += term * h_pow / (n + 1)
                    h_pow *= h

                Q_val = eval_poly(ta, h)
                Qp_val = mp.mpf(0)
                h_pow = mp.mpf(1)
                for k in range(ORDER):
                    Qp_val += (k + 1) * ta[k + 1] * h_pow
                    h_pow *= h

                r += h

            return Q_val, Qp_val, integral

        # ==================================================================
        # Wronskian shooting criterion at a given r_match.
        # F = Q*K_0' - Q'*K_0 = -B/r  where B is the I_0 coefficient.
        # a < a* => B > 0 => F < 0.   a > a* => B < 0 => F > 0.
        # ==================================================================
        def wronskian_at(r_match, Q_m, Qp_m):
            K0 = mp.besselk(0, r_match)
            K0p = -mp.besselk(1, r_match)
            return Q_m * K0p - Qp_m * K0

        # ------------------------------------------------------------------
        # Phase 1: Bisection at R_MATCH_BISECT (small, fast) for ~12 digits.
        # ------------------------------------------------------------------
        R_MATCH_BISECT = mp.mpf(8)

        a_lo = mp.mpf('2.2')
        a_hi = mp.mpf('2.3')

        for _ in range(55):
            a_mid = (a_lo + a_hi) / 2
            Q_m, Qp_m, _, _, _ = integrate_qp(a_mid, R_MATCH_BISECT)
            F = wronskian_at(R_MATCH_BISECT, Q_m, Qp_m)
            if F < 0:
                a_lo = a_mid
            else:
                a_hi = a_mid

        # ------------------------------------------------------------------
        # Phase 2: Newton at R_MATCH_NEWTON (larger) for ~35 digits.
        # ------------------------------------------------------------------
        R_MATCH_NEWTON = mp.mpf(25)
        K0_n = mp.besselk(0, R_MATCH_NEWTON)
        K0p_n = -mp.besselk(1, R_MATCH_NEWTON)

        a_val = (a_lo + a_hi) / 2
        for _ in range(15):
            Q_m, Qp_m, P_m, Pp_m, _ = integrate_qp(a_val, R_MATCH_NEWTON)
            F = Q_m * K0p_n - Qp_m * K0_n
            dF = P_m * K0p_n - Pp_m * K0_n
            if abs(dF) < mp.power(10, -160):
                break
            da = -F / dF
            a_val += da
            if abs(da) < mp.power(10, -130):
                break

        a_star = a_val

        # ------------------------------------------------------------------
        # Final integration to R_MATCH_NEWTON + K_0 tail correction.
        # ------------------------------------------------------------------
        Q_m, Qp_m, integral_main = integrate_q(a_star, R_MATCH_NEWTON)

        A_coeff = Q_m / K0_n

        # Tail: int_{R_MATCH}^inf A^2 * K_0(r)^2 * r dr
        tail = mp.quad(
            lambda r: A_coeff ** 2 * mp.besselk(0, r) ** 2 * r,
            [R_MATCH_NEWTON, mp.inf],
        )

        total_integral = integral_main + tail
        Nc = 2 * mp.pi * total_integral
        return Nc


if __name__ == "__main__":
    print(str(compute()))
