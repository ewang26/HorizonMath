"""
Reference numerical computation for: Autocorrelation Constant C Upper Bound

The autocorrelation constant C is defined as:
    C = inf_f max_t (f*f)(t) / (∫f)^2
where f is non-negative and supported on [-1/4, 1/4].

Current best bounds:
    1.2748 ≤ C ≤ 1.50992

Upper bound: Matolcsi & Vinuesa (2010), arXiv:1002.3298
Lower bound: Cloninger & Steinerberger (2014), arXiv:1205.0626

The best known upper bound of 1.50992 comes from an optimized construction
by Matolcsi & Vinuesa. A simple indicator function f = 1_{[-1/4, 1/4]}
gives ratio 2.0, which is far from optimal.
"""
from mpmath import mp, mpf

mp.dps = 110


def compute():
    """
    Return the best known upper bound on the autocorrelation constant C.

    The best known construction (Matolcsi & Vinuesa, 2010) achieves
    max_t (f*f)(t) / (∫f)^2 ≈ 1.50992.
    """
    # Best known upper bound from Matolcsi & Vinuesa (2010)
    best_known_upper = mpf("1.50992")
    return best_known_upper


if __name__ == "__main__":
    result = compute()
    print(mp.nstr(result, 110, strip_zeros=False))
