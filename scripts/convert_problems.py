"""Convert problems.tex to JSON format for LLM prompting."""

import json
import re
from pathlib import Path

PREAMBLE = "You are a research mathematician. Consider the following open problem in mathematics."

INSTRUCTIONS = """As this is an open research problem, the solution may not be known. However, your goal is novel mathematical discovery, so you must find a solution that exactly satisfies the problem task detailed above. It is possible there is no known existing solution, but you must find the desired solution regardless. Your solution must include only constants, elementary functions, gamma, zeta, polylog, elliptic integrals, and hypergeometric functions. Functions or expressions of greater complexity are explicitly not permissible.

Your solution must be a finite combination of:

- Rational and algebraic numbers

- Constants: $\\pi$, e, $\\gamma$ (Euler-Mascheroni), G (Catalan's constant)

- Elementary functions at algebraic arguments- Special functions: $\\Gamma$, $\\zeta$, polylogarithms, elliptic integrals, hypergeometric functions at algebraic or rational arguments

INADMISSIBLE expressions or tools:

- The defining integral/sum itself or equivalent reformulations

- Unevaluated infinite series, products, or limits

- Numerical approximations

REQUIRED OUTPUT FORMAT:

A Python function using mpmath that computes your expression with the following structure.

def proposed_solution():
    import mpmath
# Your implementation using only: constants, elementary functions,    # gamma, zeta, polylog, elliptic integrals, hypergeometric functions
    return result"""


def parse_problems(tex_content: str) -> list[dict]:
    """Parse LaTeX content and extract problems."""
    problems = []

    # Find all problem subsections
    pattern = r'\\subsection\*\{Problem:\s*(.+?)\}\s*\\textbf\{Definition:\}\s*(.+?)\\noindent\\textbf\{Task:\}\s*(.+?)(?=\\subsection\*|\\section\*|\\end\{document\})'

    matches = re.findall(pattern, tex_content, re.DOTALL)

    for title, definition, task in matches:
        # Clean up whitespace
        title = title.strip()
        definition = ' '.join(definition.split())
        task = ' '.join(task.split())

        # Build the full prompt
        prompt = f"{PREAMBLE}\n\n**{title}**\n\n**Definition:** {definition}\n\n**Task:** {task}\n\n{INSTRUCTIONS}"

        problems.append({
            "title": title,
            "definition": definition,
            "task": task,
            "prompt": prompt
        })

    return problems


def main():
    tex_path = Path(__file__).parent.parent / "problems.tex"
    output_path = Path(__file__).parent.parent / "data" / "problems.json"

    tex_content = tex_path.read_text()
    problems = parse_problems(tex_content)

    output_path.write_text(json.dumps(problems, indent=2, ensure_ascii=False))
    print(f"Converted {len(problems)} problems to {output_path}")


if __name__ == "__main__":
    main()
