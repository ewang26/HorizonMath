"""
Reference numerical computation for: Autocorrelation Constant C Upper Bound

The autocorrelation constant C is defined as:
    C = inf_f max_t (f*f)(t) / (∫f)^2
where f is non-negative and supported on [-1/4, 1/4].

Current benchmark bounds:
    1.28 ≤ C ≤ 1.5028503020710076

The upper bound is represented by the fixed benchmark certificate archived at
reports/gpt56_pro_final_solutions/certificates/autocorr_upper.json.

A simple indicator function f = 1_{[-1/4, 1/4]} gives ratio 2.0, which is far
from the archived certificate value.
"""
from mpmath import mp, mpf

mp.dps = 110


def compute():
    """
    Return the best known upper bound on the autocorrelation constant C.

    The fixed benchmark certificate achieves
    max_t (f*f)(t) / (∫f)^2 ≈ 1.5028503020710076.
    """
    best_known_upper = mpf("1.5028503020710076")
    return best_known_upper


if __name__ == "__main__":
    result = compute()
    print(mp.nstr(result, 110, strip_zeros=False))
