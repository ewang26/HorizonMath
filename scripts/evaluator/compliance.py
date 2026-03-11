"""LLM-based compliance checker for solution methods.

Verifies that model solutions use genuine closed-form expressions rather than
forbidden numerical techniques (numerical integration, truncated series,
numerical root-finding, etc.). Uses Gemini 3 Flash with low thinking budget.
"""

import json
import os
import re
from dataclasses import dataclass

from google import genai
from google.genai import types


@dataclass
class ComplianceResult:
    """Result of a compliance check on a proposed solution."""
    compliant: bool
    reason: str


_COMPLIANCE_PROMPT = """\
You are a code reviewer checking whether a mathematical solution follows the rules.

The solution MUST be a genuine closed-form symbolic expression. The following techniques are FORBIDDEN:

1. **Numerical integration**: `mp.quad()`, `mp.quadgl()`, `mp.quadts()`, `mpmath.quad()`, `scipy.integrate`, or any numerical integration routine.
2. **Finite truncations of infinite series**: Loops summing many terms (e.g., `for k in range(10000)`) to approximate an infinite series. A small finite sum that IS the exact answer is fine (for example, over five terms), but a solution that involves many precomputed terms (hardcoded constants, long lists of coefficients, etc.) is unacceptable.
3. **Numerical root-finding**: `fsolve`, `brentq`, `newton`, `nsolve`, `findroot`, `scipy.optimize`, or any numerical solver.
4. **Restating the defining expression as a computational procedure**: Computing a resultant by evaluating one polynomial at the roots of another, or computing a sum/product by iterating over its terms. This INCLUDES calling `mp.hyper()` or other hypergeometric functions when the hypergeometric series is simply the defining sum rewritten in hypergeometric notation. Specifically: if the problem defines a quantity as an infinite series `Σ a(n)` where the ratio `a(n+1)/a(n)` is rational in `n`, then expressing that same series as `mp.hyper(...)` is just a notational restatement and is FORBIDDEN. However, if the problem defines a quantity as an integral, a limit, or another non-series expression, and the solution expresses it via a hypergeometric function, that IS a non-trivial closed form and IS acceptable — the hypergeometric series is not the defining expression in that case.
5. **Unevaluated infinite series/products/limits**: Using `mpmath.nsum`, `mpmath.nprod`, or similar to numerically evaluate an infinite series or product.
6. **Hardcoded numeric literals**: Returning a bare multi-digit decimal string as the answer without any symbolic derivation, e.g. `return mpf("1.20205690315959428539973816151144999076")` or `return mpf(1.2020569031595942)`. A valid solution must express the answer in terms of named mathematical operations or constants — not by encoding the digits of the answer directly.
7. **Circular / tautological identities**: Using special functions that internally encode or trivially compute the target constant. For example, using `mp.hyperu` or `mp.gammainc` to compute the Euler-Mascheroni constant γ is forbidden because these functions evaluate to expressions involving γ at the chosen arguments. The solution must provide a genuinely independent expression, not an identity that the constant satisfies by definition.
8. **Numerical parameter fitting / digit-matching constructions**: Expressions where arbitrary-looking numerical coefficients or high powers appear to have been tuned to match the target constant's known digits. One indicator of this is coefficients or exponents that are unusually specific, like `sqrt(30261)/26` or high powers used as successive correction terms like `- q**12/9 - (173/4)*q**18`. These patterns suggest the expression was found by searching over coefficient/exponent space rather than derived from mathematical reasoning. However, note that conjectures, ansatzes, and combinations of constants from different domains are acceptable when the coefficients are structurally simple (small integers, simple fractions like 1/2 or 1/π, well-known constants). This rule specifically targets cases where the coefficients or terms strongly appear to have been reverse-engineered to force a numerical match.

ALLOWED techniques include:
- Using known constants (pi, e, euler-gamma, Catalan's constant)
- Calling special functions (gamma, zeta, polylog, elliptic integrals, hypergeometric) at specific arguments — but only when the function represents a genuinely different mathematical quantity, not a restatement of the problem's defining expression
- Symbolic algebra to combine these into a closed-form expression
- Small exact finite sums that represent the actual mathematical answer (not truncations)
- Novel conjectures combining constants from different mathematical domains, as long as the coefficients are structurally simple and not arbitrarily tuned

{problem_context}Here is the code to review:

```python
{code}
```

Respond with ONLY a JSON object (no markdown fences) with two fields:
- "compliant": true if the solution follows the rules, false if it uses forbidden techniques
- "reason": a brief explanation (one sentence)
"""


DEFAULT_COMPLIANCE_ROUNDS = 3


def _single_compliance_check(prompt: str) -> ComplianceResult:
    """Run a single compliance check call against Gemini."""
    try:
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.LOW
                ),
            ),
        )
        text = response.text.strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            text = text.strip()

        result = json.loads(text)
        return ComplianceResult(
            compliant=bool(result.get("compliant", True)),
            reason=str(result.get("reason", "")),
        )
    except (json.JSONDecodeError, KeyError) as e:
        return ComplianceResult(
            compliant=True,
            reason=f"Compliance check parse error (defaulting to compliant): {e}",
        )
    except Exception as e:
        return ComplianceResult(
            compliant=True,
            reason=f"Compliance check API error (defaulting to compliant): {e}",
        )


def check_solution_compliance(
    code: str,
    problem_prompt: str = "",
    n: int = DEFAULT_COMPLIANCE_ROUNDS,
) -> ComplianceResult:
    """Check whether extracted solution code uses only allowed techniques.

    Runs the check n times and takes a majority vote to reduce LLM
    non-determinism. A solution is compliant only if a strict majority
    of rounds agree it is compliant.

    Args:
        code: The extracted proposed_solution() source code.
        problem_prompt: Optional problem prompt text. If provided, any
            problem-specific restrictions (e.g. forbidden functions) are
            included in the compliance check so the reviewer can enforce them.
        n: Number of compliance check rounds (default 3). Majority vote
            determines the final result.

    Returns:
        ComplianceResult with compliant flag and explanation.
    """
    if not os.environ.get("GOOGLE_API_KEY"):
        return ComplianceResult(
            compliant=True,
            reason="Compliance check skipped (GOOGLE_API_KEY is not set).",
        )

    problem_context = ""
    if problem_prompt:
        problem_context = (
            "The problem being solved is described below. Pay close attention to any "
            "problem-specific restrictions — these are additional rules that MUST be enforced "
            "on top of the general rules above.\n\n"
            f"**Problem description:**\n{problem_prompt}\n\n"
        )

    prompt = _COMPLIANCE_PROMPT.format(code=code, problem_context=problem_context)

    results = []
    for _ in range(n):
        results.append(_single_compliance_check(prompt))

    compliant_count = sum(1 for r in results if r.compliant)
    non_compliant_count = n - compliant_count
    majority_compliant = compliant_count > n / 2

    # Pick the reason from the majority side (first occurrence)
    majority_reason = next(
        r.reason for r in results if r.compliant == majority_compliant
    )
    vote_str = f" [{compliant_count}/{n} compliant]"

    return ComplianceResult(
        compliant=majority_compliant,
        reason=majority_reason + vote_str,
    )
