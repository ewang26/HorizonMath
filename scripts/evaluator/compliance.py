"""LLM-based compliance checker for solution methods.

Verifies that model solutions use genuine closed-form expressions rather than
forbidden numerical techniques (numerical integration, truncated series,
numerical root-finding, etc.). Uses Gemini 3.6 Flash with high thinking by
default, with GPT-5.6 Terra at high reasoning available as an alternative.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import openai
from google import genai
from google.genai import types


@dataclass
class ComplianceResult:
    """Result of a compliance check on a proposed solution."""
    compliant: Optional[bool]
    reason: str
    provider: str = ""
    model: str = ""

    @property
    def status(self) -> str:
        """Return an explicit status without conflating errors with rejection."""
        if self.compliant is True:
            return "compliant"
        if self.compliant is False:
            return "non_compliant"
        return "indeterminate"


_COMPLIANCE_PROMPT = """\
You are a code reviewer checking whether a mathematical solution follows the rules.

Apply these two principles before the numbered method rules:

- **Task fulfillment**: The response must provide the object and form requested by the problem, respect every explicit problem-specific restriction, and cover the full requested parameter domain. A numerically matching constant is non-compliant if the task instead asks for equations, a construction, a formula valid over a parameter domain, or a representation beyond a specifically excluded known form. Problem-specific restrictions override the general allowances below.
- **Judge the mathematical representation, not library internals**: The benchmark separately checks numerical accuracy. Decide whether the mathematical expression represented by the code is permissible. The fact that `mpmath` numerically evaluates an otherwise allowed named function is not by itself a violation.

The solution MUST be a genuine closed-form symbolic expression. The following techniques are FORBIDDEN:

1. **Numerical integration**: `mp.quad()`, `mp.quadgl()`, `mp.quadts()`, `mpmath.quad()`, `scipy.integrate`, or any numerical integration routine.
2. **Finite truncations of infinite series**: Loops summing many terms (e.g., `for k in range(10000)`) to approximate an infinite series, precision-dependent or accuracy-dependent stopping rules, and long lists of precomputed terms are forbidden. In contrast, a finite sum, product, loop, or matrix construction whose length is exactly determined by an input integer such as `n` is allowed when it constructs the exact mathematical answer; it is not a truncation merely because its size varies with `n`.
3. **Numerical root-finding**: `fsolve`, `brentq`, `newton`, `nsolve`, `findroot`, `scipy.optimize`, or any numerical solver.
4. **Restating the defining expression as a computational procedure**: Computing a resultant by evaluating one polynomial at the roots of another, or computing a defining sum/product by iterating over its terms, is forbidden. This includes calling `mp.hyper()` or another hypergeometric function when it merely rewrites the problem's defining series. Specifically, if the problem defines `Σ a(n)` and `a(n+1)/a(n)` is rational in `n`, expressing that same series as `mp.hyper(...)` is a notational restatement. Judge the transformation relative to the original definition: when the problem instead defines an integral, limit, differential equation, or other non-series object, a fixed hypergeometric or other named special-function value can be a genuine closed form even though that function has an internal series definition. Do not reject such a transformation merely because an equivalent series exists. An explicit problem restriction excluding series or hypergeometric repackagings still controls.
5. **Unevaluated infinite series/products/limits**: Using `mpmath.nsum`, `mpmath.nprod`, or similar to numerically evaluate an infinite series or product.
6. **Hardcoded or encoded target values**: Returning a bare multi-digit decimal string as the answer without a symbolic derivation is forbidden, e.g. `return mpf("1.20205690315959428539973816151144999076")`. The same prohibition applies to a rational with a power-of-ten denominator, an enormous unexplained integer or integer factorization, a high-degree algebraic number, a continued fraction, or another exact representation whose apparent purpose is to encode the known target digits. Exactness alone does not make such an encoding a symbolic derivation.
7. **Circular / tautological identities**: Using special functions that internally encode or trivially compute the target constant is forbidden. For example, using `mp.hyperu` or `mp.gammainc` to compute the Euler-Mascheroni constant γ is circular when the chosen value is defined or conventionally evaluated through an identity involving γ. Apply an independence test: reject a special-function value when the target is part of its defining local expansion, parameter derivative, normalization, or standard identity at the chosen arguments. The expression must be genuinely independent, not an identity that the constant satisfies by definition.
8. **Numerical parameter fitting / digit-matching constructions**: Expressions with arbitrary-looking coefficients, denominators, exponents, or successive tiny correction terms are forbidden when they appear reverse-engineered to match known target digits. Concrete warning signs include unusually specific values such as `sqrt(30261)/26`, unexplained large coefficients, and high-power corrections such as `- q**12/9 - (173/4)*q**18`. Structural conjectures remain allowed when supported by small integers, simple rational parameters, symmetries, known related constants, or a uniform formula. State the concrete feature that indicates fitting; do not reject merely because a conjecture is unproven or unfamiliar.

ALLOWED techniques include:
- Using known constants (pi, e, euler-gamma, Catalan's constant)
- Calling special functions (gamma, zeta, polylog, elliptic integrals, hypergeometric) at specific arguments when they represent a genuinely different mathematical quantity rather than the problem's defining expression
- Symbolic algebra to combine these into a closed-form expression
- Exact finite sums, products, loops, and matrix constructions that represent the actual mathematical answer rather than an approximation
- Named functions of exactly specified finite matrices, including matrix logarithms, square roots, exponentials, determinants, and traces, unless the matrix encodes fitted target digits, disguises a forbidden representation, or violates a problem-specific restriction
- Novel conjectures combining constants from different mathematical domains when the coefficients are structurally simple and not arbitrarily tuned

Before returning a verdict:
1. Identify the original mathematical definition and every explicit problem-specific restriction.
2. Check that the response answers the requested object and, for functions, covers the full requested input domain.
3. Apply Rules 1–8 to the mathematical representation, not merely to library implementation details.
4. Check for target-digit encoding, arbitrary fitted parameters, and circular special-function identities.
5. When rejecting, identify either the task-fulfillment failure or the most specific violated numbered rule and cite a concrete feature of the submitted code. When accepting a borderline construction, explain why it is an independent transformation, exact finite construction, or structural conjecture rather than a restatement, truncation, encoding, circular identity, or fit.

{problem_context}Here is the code to review:

```python
{code}
```

Respond with ONLY a JSON object (no markdown fences) with two fields:
- "compliant": true if the solution follows the rules, false if it uses forbidden techniques
- "reason": a brief explanation (one sentence)
"""


DEFAULT_COMPLIANCE_ROUNDS = 3
DEFAULT_COMPLIANCE_PROVIDER = "gemini"
COMPLIANCE_MODEL = "gemini-3.6-flash"
COMPLIANCE_THINKING_LEVEL = types.ThinkingLevel.HIGH
OPENAI_COMPLIANCE_MODEL = "gpt-5.6-terra"
OPENAI_COMPLIANCE_REASONING_EFFORT = "high"

_COMPLIANCE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "compliant": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["compliant", "reason"],
    "additionalProperties": False,
}


def _reviewer_config() -> tuple[str, str]:
    """Return the configured compliance provider and model."""
    provider = os.environ.get(
        "COMPLIANCE_PROVIDER", DEFAULT_COMPLIANCE_PROVIDER
    ).strip().lower()
    if provider == "google":
        provider = "gemini"
    if provider == "gemini":
        default_model = COMPLIANCE_MODEL
    elif provider == "openai":
        default_model = OPENAI_COMPLIANCE_MODEL
    else:
        raise ValueError(
            "COMPLIANCE_PROVIDER must be either 'gemini' or 'openai'"
        )
    model = os.environ.get("COMPLIANCE_MODEL", default_model).strip()
    if not model:
        raise ValueError("COMPLIANCE_MODEL must not be empty")
    return provider, model


def _parse_compliance_response(
    text: str,
    provider: str,
    model: str,
) -> ComplianceResult:
    """Parse and validate one reviewer's structured response."""
    if not isinstance(text, str):
        raise ValueError("Reviewer response text is missing or is not a string")
    text = text.strip()

    # Be defensive if a provider returns JSON inside markdown fences.
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Reviewer response must be a JSON object")
    if not isinstance(result.get("compliant"), bool):
        raise ValueError(
            "Reviewer response must contain a boolean 'compliant' field"
        )
    if not isinstance(result.get("reason"), str):
        raise ValueError(
            "Reviewer response must contain a string 'reason' field"
        )
    return ComplianceResult(
        compliant=result["compliant"],
        reason=result["reason"],
        provider=provider,
        model=model,
    )


def _single_compliance_check(
    prompt: str,
    provider: str,
    model: str,
) -> ComplianceResult:
    """Run a single compliance check against the configured reviewer."""
    try:
        if provider == "gemini":
            client = genai.Client()
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_level=COMPLIANCE_THINKING_LEVEL
                    ),
                    response_mime_type="application/json",
                    response_schema=_COMPLIANCE_RESPONSE_SCHEMA,
                ),
            )
            text = response.text
        else:
            reasoning_effort = os.environ.get(
                "COMPLIANCE_REASONING_EFFORT",
                OPENAI_COMPLIANCE_REASONING_EFFORT,
            ).strip().lower()
            client = openai.OpenAI(timeout=10 * 60)
            response = client.responses.create(
                model=model,
                input=prompt,
                reasoning={"effort": reasoning_effort},
                max_output_tokens=2000,
                store=False,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "compliance_result",
                        "strict": True,
                        "schema": _COMPLIANCE_RESPONSE_SCHEMA,
                    },
                },
            )
            text = response.output_text
        return _parse_compliance_response(text, provider, model)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        return ComplianceResult(
            compliant=None,
            reason=(
                "Compliance check indeterminate due to response parse/schema "
                f"error: {e}"
            ),
            provider=provider,
            model=model,
        )
    except Exception as e:
        return ComplianceResult(
            compliant=None,
            reason=f"Compliance check indeterminate due to API error: {e}",
            provider=provider,
            model=model,
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
        ComplianceResult with compliant=True for a strict compliant majority,
        False for a strict non-compliant majority, or None when the check is
        indeterminate (for example, due to missing credentials, API failures,
        parse failures, or no strict majority).
    """
    try:
        provider, model = _reviewer_config()
    except ValueError as e:
        return ComplianceResult(
            compliant=None,
            reason=f"Compliance check indeterminate due to configuration error: {e}",
        )

    api_key_name = "GOOGLE_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
    if not os.environ.get(api_key_name):
        return ComplianceResult(
            compliant=None,
            reason=f"Compliance check indeterminate ({api_key_name} is not set).",
            provider=provider,
            model=model,
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
        results.append(_single_compliance_check(prompt, provider, model))

    compliant_count = sum(1 for r in results if r.compliant is True)
    non_compliant_count = sum(1 for r in results if r.compliant is False)
    indeterminate_count = n - compliant_count - non_compliant_count

    if compliant_count > n / 2:
        outcome = True
        outcome_reason = next(r.reason for r in results if r.compliant is True)
    elif non_compliant_count > n / 2:
        outcome = False
        outcome_reason = next(r.reason for r in results if r.compliant is False)
    else:
        outcome = None
        indeterminate_reasons = [
            r.reason for r in results if r.compliant is None
        ]
        outcome_reason = (
            "Compliance check indeterminate because no strict majority was reached."
        )
        if indeterminate_reasons:
            outcome_reason += f" {indeterminate_reasons[0]}"

    vote_str = (
        f" [{compliant_count}/{n} compliant, "
        f"{non_compliant_count}/{n} non-compliant, "
        f"{indeterminate_count}/{n} indeterminate]"
    )

    return ComplianceResult(
        compliant=outcome,
        reason=outcome_reason + vote_str,
        provider=provider,
        model=model,
    )
