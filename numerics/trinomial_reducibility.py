"""
Reference computation for: Trinomial Reducibility x^n + x^k + 1 over Q

This script determines whether the trinomial x^n + x^k + 1 is reducible
over the rational numbers for given (n, k) with 1 <= k < n.

Uses sympy for polynomial factorization.
"""

from sympy import symbols, factor, Poly, QQ
from sympy.polys.factortools import dup_factor_list
import json


def is_reducible_over_Q(n, k):
    """
    Determine if x^n + x^k + 1 is reducible over Q.

    Returns True if reducible, False if irreducible.
    """
    if k <= 0 or k >= n:
        raise ValueError(f"Invalid (n, k) = ({n}, {k}): need 0 < k < n")

    x = symbols('x')
    poly = x**n + x**k + 1

    # Factor over rationals
    factored = factor(poly, domain=QQ)

    # Check if it factors non-trivially
    # If the polynomial is irreducible, factor() returns the polynomial itself
    poly_obj = Poly(poly, x, domain=QQ)
    factors = poly_obj.factor_list()[1]  # Get list of (factor, multiplicity)

    # Reducible if more than one factor or if the single factor has degree < n
    if len(factors) > 1:
        return True
    if len(factors) == 1 and factors[0][0].degree() < n:
        return True

    return False


def compute_reducibility_table(max_n=50):
    """
    Compute reducibility for all (n, k) pairs with 1 <= k < n <= max_n.

    Returns a dictionary mapping (n, k) -> bool (True = reducible).
    """
    results = {}

    for n in range(2, max_n + 1):
        for k in range(1, n):
            is_red = is_reducible_over_Q(n, k)
            results[(n, k)] = is_red

    return results


def analyze_patterns(results):
    """Analyze patterns in the reducibility data."""
    reducible_pairs = [(n, k) for (n, k), v in results.items() if v]
    irreducible_pairs = [(n, k) for (n, k), v in results.items() if not v]

    print(f"Total pairs analyzed: {len(results)}")
    print(f"Reducible: {len(reducible_pairs)}")
    print(f"Irreducible: {len(irreducible_pairs)}")

    print("\nReducible cases (n, k):")
    for n, k in sorted(reducible_pairs)[:50]:
        print(f"  ({n}, {k})", end="")
        # Check if gcd(n, k) > 1
        from math import gcd
        g = gcd(n, k)
        if g > 1:
            print(f" [gcd={g}]", end="")
        print()

    # Check pattern: is it always reducible when gcd(n,k) > 1?
    gcd_reducible = [(n, k) for n, k in reducible_pairs if gcd(n, k) > 1]
    gcd_irreducible = [(n, k) for n, k in irreducible_pairs if gcd(n, k) > 1]

    print(f"\nWith gcd(n,k) > 1: {len(gcd_reducible)} reducible, {len(gcd_irreducible)} irreducible")

    if gcd_irreducible:
        print("Irreducible despite gcd > 1:")
        for n, k in gcd_irreducible[:10]:
            print(f"  ({n}, {k}), gcd = {gcd(n, k)}")


def compute():
    """
    Compute ground truth reducibility for all (n, k) with 1 <= k < n <= 200.

    Returns dictionary of results.
    """
    print("Computing trinomial reducibility for n <= 200...")
    print("This may take a while...")

    results = {}

    for n in range(2, 201):
        if n % 20 == 0:
            print(f"  Processing n = {n}...")
        for k in range(1, n):
            try:
                results[(n, k)] = is_reducible_over_Q(n, k)
            except Exception as e:
                print(f"Error at ({n}, {k}): {e}")
                results[(n, k)] = None

    return results


def verify_predicate(predicate_func, max_n=100):
    """
    Verify a proposed predicate against computed ground truth.

    predicate_func(n, k) should return True if reducible, False if irreducible.
    """
    correct = 0
    total = 0
    errors = []

    for n in range(2, max_n + 1):
        for k in range(1, n):
            total += 1
            predicted = predicate_func(n, k)
            actual = is_reducible_over_Q(n, k)

            if predicted == actual:
                correct += 1
            else:
                errors.append((n, k, predicted, actual))

    print(f"Accuracy: {correct}/{total} = {100*correct/total:.2f}%")

    if errors:
        print(f"\nFirst 20 errors:")
        for n, k, pred, actual in errors[:20]:
            print(f"  ({n}, {k}): predicted={pred}, actual={actual}")

    return correct, total, errors


# Precomputed sample results for quick verification
# Note: gcd(n,k) > 1 does NOT imply reducibility over Q
SAMPLE_RESULTS = {
    (4, 1): False,   # x^4 + x + 1 is irreducible
    (4, 2): True,    # x^4 + x^2 + 1 = (x^2 + x + 1)(x^2 - x + 1)
    (5, 1): True,    # x^5 + x + 1 = (x^2 + x + 1)(x^3 - x^2 + 1)
    (5, 2): False,
    (6, 1): False,
    (6, 2): False,   # irreducible despite gcd(6,2) = 2
    (6, 3): False,   # irreducible despite gcd(6,3) = 3
    (7, 1): False,
    (7, 2): True,    # x^7 + x^2 + 1 = (x^2 + x + 1)(x^5 - x^4 + x^2 - x + 1)
    (7, 3): False,
    (8, 1): True,    # x^8 + x + 1 = (x^2 + x + 1)(x^6 - x^5 + x^3 - x^2 + 1)
    (8, 2): False,   # irreducible despite gcd(8,2) = 2
    (8, 4): True,    # x^8 + x^4 + 1 = (x^2 - x + 1)(x^2 + x + 1)(x^4 - x^2 + 1)
    (9, 3): False,   # irreducible despite gcd(9,3) = 3
    (10, 1): False,
    (10, 2): True,   # x^10 + x^2 + 1 = (x^2 + x + 1)(x^8 - x^7 + x^5 - x^4 + x^3 - x + 1)
    (10, 5): True,   # x^10 + x^5 + 1 = (x^2 + x + 1)(x^8 - x^7 + x^6 - x^4 + x^2 - x + 1)
    (12, 4): False,  # irreducible despite gcd(12,4) = 4
    (12, 6): True,   # x^12 + x^6 + 1 factors
    (15, 5): False,  # irreducible despite gcd(15,5) = 5
}


if __name__ == "__main__":
    print("Sample reducibility results for x^n + x^k + 1:")
    print()

    for (n, k), expected in sorted(SAMPLE_RESULTS.items()):
        computed = is_reducible_over_Q(n, k)
        status = "✓" if computed == expected else "✗"
        red_str = "reducible" if computed else "irreducible"
        print(f"  ({n}, {k}): {red_str} {status}")

    print()
    print("Computing more extensive table...")
    results = compute_reducibility_table(max_n=30)
    analyze_patterns(results)
