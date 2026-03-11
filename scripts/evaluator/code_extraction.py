"""Code extraction module for extracting proposed_solution() from LLM outputs."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple


class ExtractionStatus(Enum):
    """Status of code extraction."""
    SUCCESS = "success"
    NO_FUNCTION_FOUND = "no_function_found"
    MALFORMED_CODE = "malformed_code"


@dataclass
class ExtractionResult:
    """Result of extracting code from LLM output."""
    status: ExtractionStatus
    code: Optional[str] = None
    error_message: Optional[str] = None

    def __bool__(self) -> bool:
        """Return True if extraction was successful."""
        return self.status == ExtractionStatus.SUCCESS


def _extract_code_blocks(llm_output: str) -> List[Tuple[str, str]]:
    """
    Extract all code blocks from markdown-formatted text.

    Returns a list of (language_tag, code_content) tuples.
    Handles various language tags: python, py, python3, Python, etc.
    """
    # Pattern to match code blocks with optional language specifier
    # Captures: language tag (may be empty) and code content
    pattern = r'```(\w*)\s*\n?(.*?)```'
    matches = re.findall(pattern, llm_output, re.DOTALL)
    return [(lang.lower(), code) for lang, code in matches]


def _is_python_code_block(language_tag: str) -> bool:
    """Check if a language tag indicates Python code."""
    python_tags = {'python', 'py', 'python3', 'py3', ''}
    return language_tag in python_tags


def _extract_function_with_imports(code: str) -> str:
    """
    Extract the proposed_solution function along with any necessary imports.

    If imports are at module level (outside the function), include them.
    """
    lines = code.split('\n')
    result_lines = []
    in_function = False
    function_indent = 0
    found_function = False

    for line in lines:
        stripped = line.lstrip()

        # Capture import statements at module level
        if not in_function and (stripped.startswith('import ') or stripped.startswith('from ')):
            result_lines.append(line)
            continue

        # Check for function definition (with optional decorators, type hints)
        # Matches: def proposed_solution(), def proposed_solution() -> type:
        if re.match(r'(@\w+.*|def proposed_solution\s*\([^)]*\)\s*(->.*)?:)', stripped):
            if stripped.startswith('@'):
                # Decorator - include it
                result_lines.append(line)
                continue
            # Function definition
            in_function = True
            found_function = True
            function_indent = len(line) - len(stripped)
            result_lines.append(line)
            continue

        # Inside function - capture all indented lines
        if in_function:
            # Empty lines are allowed inside function
            if not stripped:
                result_lines.append(line)
                continue

            current_indent = len(line) - len(stripped)
            # If we hit a line with same or less indentation (and not empty), function ends
            if current_indent <= function_indent and stripped:
                # Check if this is a continuation (like a new def or class)
                if stripped.startswith(('def ', 'class ', 'if __name__')):
                    break
                # Could be part of multiline expression - be conservative
                result_lines.append(line)
            else:
                result_lines.append(line)

    if not found_function:
        return code  # Return original if we couldn't parse it

    return '\n'.join(result_lines).strip()


def _normalize_indentation(code: str) -> str:
    """
    Normalize indentation in extracted code.

    Removes common leading whitespace from all lines while preserving
    relative indentation structure.
    """
    lines = code.split('\n')
    if not lines:
        return code

    # Find minimum non-zero indentation (for non-empty lines)
    min_indent = float('inf')
    for line in lines:
        stripped = line.lstrip()
        if stripped:  # Non-empty line
            indent = len(line) - len(stripped)
            if indent > 0:
                min_indent = min(min_indent, indent)

    # If no indentation found or first line has no indent, return as-is
    if min_indent == float('inf') or min_indent == 0:
        return code

    # Check if first non-empty line starts with 'def' or 'import' at column 0
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            if len(line) == len(stripped):  # No leading whitespace
                return code
            break

    # Remove common indentation
    result = []
    for line in lines:
        if line.strip():  # Non-empty
            if len(line) >= min_indent:
                result.append(line[min_indent:])
            else:
                result.append(line.lstrip())
        else:
            result.append('')

    return '\n'.join(result)


def extract_proposed_solution(llm_output: str) -> ExtractionResult:
    """
    Extract the proposed_solution() function from LLM output.

    Handles various formats:
    - Code blocks with ```python, ```py, ```python3, ``` ... ```
    - Multiple code blocks (prefers the last one containing proposed_solution)
    - Raw function definitions
    - Functions with decorators and type hints
    - Imports at module level (outside the function)
    - Indentation normalization

    Args:
        llm_output: Raw LLM output text

    Returns:
        ExtractionResult with status, extracted code, and any error message
    """
    if not llm_output or not llm_output.strip():
        return ExtractionResult(
            status=ExtractionStatus.NO_FUNCTION_FOUND,
            error_message="Empty LLM output"
        )

    # Strategy 1: Extract from markdown code blocks
    code_blocks = _extract_code_blocks(llm_output)

    # Filter to Python code blocks containing proposed_solution
    # Prefer the LAST matching block (LLMs often refine their answer)
    candidate_blocks = []
    for lang, code in code_blocks:
        if _is_python_code_block(lang) and 'def proposed_solution' in code:
            candidate_blocks.append(code)

    if candidate_blocks:
        # Take the last matching block (most likely to be the final answer)
        code = candidate_blocks[-1].strip()
        code = _normalize_indentation(code)

        if _validate_code(code):
            return ExtractionResult(
                status=ExtractionStatus.SUCCESS,
                code=code
            )
        else:
            # Try all candidates in reverse order
            for code in reversed(candidate_blocks[:-1]):
                code = code.strip()
                code = _normalize_indentation(code)
                if _validate_code(code):
                    return ExtractionResult(
                        status=ExtractionStatus.SUCCESS,
                        code=code
                    )

            # Return the last one with error
            return ExtractionResult(
                status=ExtractionStatus.MALFORMED_CODE,
                code=candidate_blocks[-1].strip(),
                error_message="Code block found but appears malformed (missing return statement or unbalanced delimiters)"
            )

    # Strategy 2: Find raw function definition with type hints support
    # Use a more careful extraction that identifies the function boundaries
    raw_code = _extract_raw_function(llm_output)
    if raw_code:
        code = _normalize_indentation(raw_code['function'])

        # Include imports if found
        if raw_code['imports']:
            code = raw_code['imports'] + '\n\n' + code

        if _validate_code(code):
            return ExtractionResult(
                status=ExtractionStatus.SUCCESS,
                code=code
            )
        else:
            return ExtractionResult(
                status=ExtractionStatus.MALFORMED_CODE,
                code=code,
                error_message="Function definition found but appears malformed"
            )

    return ExtractionResult(
        status=ExtractionStatus.NO_FUNCTION_FOUND,
        error_message="Could not find proposed_solution() function in LLM output"
    )


def _extract_leading_imports(text: str, function_start: int) -> str:
    """
    Extract import statements that appear before the function definition.

    Only includes imports that are at module level (not indented).
    """
    prefix = text[:function_start]
    lines = prefix.split('\n')

    imports = []
    for line in lines:
        stripped = line.strip()
        # Only capture module-level imports (no leading whitespace)
        if line == stripped and (stripped.startswith('import ') or stripped.startswith('from ')):
            imports.append(stripped)

    return '\n'.join(imports)


def _extract_raw_function(text: str) -> Optional[dict]:
    """
    Extract a raw function definition from text (not in a code block).

    Returns a dict with 'function' and 'imports' keys, or None if not found.
    """
    lines = text.split('\n')
    function_lines = []
    import_lines = []
    in_function = False
    function_indent = 0

    for i, line in enumerate(lines):
        stripped = line.lstrip()

        # Collect import statements at module level (before function)
        if not in_function:
            if stripped.startswith('import ') or stripped.startswith('from '):
                # Check if it's at column 0 or consistent indent
                if not line.startswith(' ') and not line.startswith('\t'):
                    import_lines.append(stripped)
                    continue

        # Look for function definition
        if not in_function:
            # Match: def proposed_solution() or def proposed_solution() -> type:
            func_match = re.match(r'(\s*)(def proposed_solution\s*\([^)]*\)\s*(?:->[^:]+)?:)', line)
            if func_match:
                function_indent = len(func_match.group(1))
                function_lines.append(line)
                in_function = True
                continue

        # Inside function - collect lines
        if in_function:
            # Empty line - include it
            if not stripped:
                function_lines.append(line)
                continue

            current_indent = len(line) - len(stripped)

            # Line at function indent or less means function ended
            if current_indent <= function_indent:
                # Check if it looks like code continuation or new statement
                if stripped.startswith(('def ', 'class ', 'if __name__', '#')):
                    # Definitely end of function
                    break
                # Check if it's just text (not Python code)
                if not _looks_like_python(stripped):
                    break
                # Otherwise might be at function level - ambiguous
                # If it starts with a letter and isn't a keyword, probably prose
                if stripped[0].isalpha() and not _is_python_keyword_start(stripped):
                    break

            # Include the line
            function_lines.append(line)

    if not function_lines:
        return None

    return {
        'function': '\n'.join(function_lines).rstrip(),
        'imports': '\n'.join(import_lines) if import_lines else ''
    }


def _looks_like_python(line: str) -> bool:
    """Check if a line looks like Python code rather than prose."""
    # Common Python patterns
    python_patterns = [
        r'^(if|elif|else|for|while|try|except|finally|with|return|yield|raise|assert|pass|break|continue|import|from|def|class|lambda)\b',
        r'^\w+\s*[=\+\-\*\/\%\&\|\^]\s*',  # Assignment or operation
        r'^[\[\{\(]',  # Starts with bracket
        r'^\s*#',  # Comment
        r'\(\s*\)|\[\s*\]|\{\s*\}',  # Empty brackets
        r'^\w+\(',  # Function call
        r'^\w+\.\w+',  # Attribute access
    ]
    for pattern in python_patterns:
        if re.match(pattern, line):
            return True
    return False


def _is_python_keyword_start(line: str) -> bool:
    """Check if line starts with a Python keyword."""
    keywords = {
        'if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally',
        'with', 'return', 'yield', 'raise', 'assert', 'pass', 'break',
        'continue', 'import', 'from', 'def', 'class', 'lambda', 'and',
        'or', 'not', 'in', 'is', 'True', 'False', 'None'
    }
    word = line.split()[0] if line.split() else ''
    return word in keywords


def _validate_code(code: str) -> bool:
    """
    Basic validation that extracted code is syntactically reasonable.

    This is a lightweight check - actual syntax errors will be caught
    during execution.
    """
    if not code:
        return False

    # Must have the function definition
    if 'def proposed_solution' not in code:
        return False

    # Should have a return statement (but check it's not commented out)
    # Look for 'return' that's not in a comment
    has_return = False
    for line in code.split('\n'):
        stripped = line.strip()
        # Skip comments
        if stripped.startswith('#'):
            continue
        # Check for return in the non-comment portion
        comment_idx = stripped.find('#')
        if comment_idx >= 0:
            stripped = stripped[:comment_idx]
        if 'return ' in stripped or stripped == 'return' or stripped.endswith('return'):
            has_return = True
            break

    if not has_return:
        return False

    # Basic balance check for common delimiters
    # Count only outside of string literals for more accuracy
    open_parens, close_parens = _count_delimiters(code, '(', ')')
    open_brackets, close_brackets = _count_delimiters(code, '[', ']')
    open_braces, close_braces = _count_delimiters(code, '{', '}')

    # Allow small imbalances (could be in strings we didn't perfectly parse)
    if abs(open_parens - close_parens) > 2:
        return False
    if abs(open_brackets - close_brackets) > 2:
        return False
    if abs(open_braces - close_braces) > 2:
        return False

    # Check for common syntax errors
    # Unterminated triple quotes
    triple_double = code.count('"""')
    triple_single = code.count("'''")
    if triple_double % 2 != 0 or triple_single % 2 != 0:
        return False

    return True


def _count_delimiters(code: str, open_char: str, close_char: str) -> Tuple[int, int]:
    """
    Count delimiter pairs, attempting to ignore those inside string literals.

    This is a simplified heuristic - not a full parser.
    """
    open_count = 0
    close_count = 0
    in_string = False
    string_char = None
    i = 0

    while i < len(code):
        char = code[i]

        # Check for string start/end
        if char in ('"', "'"):
            # Check for triple quotes
            if code[i:i+3] in ('"""', "'''"):
                if in_string and string_char == code[i:i+3]:
                    in_string = False
                    string_char = None
                elif not in_string:
                    in_string = True
                    string_char = code[i:i+3]
                i += 3
                continue
            # Single/double quotes
            if in_string and string_char == char:
                in_string = False
                string_char = None
            elif not in_string:
                in_string = True
                string_char = char
            i += 1
            continue

        # Check for escape in string
        if in_string and char == '\\':
            i += 2  # Skip escaped character
            continue

        # Count delimiters outside strings
        if not in_string:
            if char == open_char:
                open_count += 1
            elif char == close_char:
                close_count += 1

        i += 1

    return open_count, close_count
