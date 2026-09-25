"""Sandboxed execution module for running proposed_solution() code safely."""

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class ExecutionStatus(Enum):
    """Status of code execution."""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    RUNTIME_ERROR = "runtime_error"
    SYNTAX_ERROR = "syntax_error"


@dataclass
class ExecutionResult:
    """Result of executing code in sandbox."""
    status: ExecutionStatus
    output: Optional[str] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None

    def __bool__(self) -> bool:
        """Return True if execution was successful."""
        return self.status == ExecutionStatus.SUCCESS


# Default execution settings
DEFAULT_TIMEOUT = 300  # 5 minutes
DEFAULT_PRECISION_DPS = 110  # digits of precision for mpmath


def get_python_executable() -> str:
    """Get the Python executable, preferring venv if available."""
    script_dir = Path(__file__).parent.parent
    project_root = script_dir.parent
    venv_python = project_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def execute_sandboxed(
    code: str,
    timeout: int = DEFAULT_TIMEOUT,
    precision_dps: int = DEFAULT_PRECISION_DPS,
    return_json: bool = False,
    test_points: list[dict] | None = None,
) -> ExecutionResult:
    """
    Execute the proposed_solution() function in a subprocess sandbox.

    The code is run in a subprocess with:
    - A timeout for safety
    - High-precision mpmath settings
    - Isolated from the main process

    Args:
        code: Python code containing proposed_solution() function
        timeout: Execution timeout in seconds
        precision_dps: Decimal places of precision for mpmath
        return_json: If True, serialize result as JSON (for construction problems)
                    If False, convert to string (for numeric problems)
        test_points: If provided, evaluate proposed_solution at multiple points.
                    Each entry is {"args": [...], "expected": "..."}.
                    Output will be a JSON list of result strings.

    Returns:
        ExecutionResult with status, output, error message, and execution time
    """
    if test_points is not None:
        # Multi-point evaluation mode
        wrapper_code = f'''
import sys, json
sys.setrecursionlimit(10000)

from mpmath import mp
mp.dps = {precision_dps}

{code}

if __name__ == "__main__":
    try:
        test_points = {json.dumps(test_points)}
        results = []
        for tp in test_points:
            args = [mp.mpf(a) if isinstance(a, str) else a for a in tp["args"]]
            try:
                result = proposed_solution(*args)
                results.append(str(result))
            except Exception as e:
                print(f"EXECUTION_ERROR at test point {{tp}}: {{type(e).__name__}}: {{e}}", file=sys.stderr)
                results.append(None)
        print(json.dumps(results))
    except Exception as e:
        print(f"EXECUTION_ERROR: {{type(e).__name__}}: {{e}}", file=sys.stderr)
        sys.exit(1)
'''
    else:
        # Single-point evaluation mode (original behavior)
        if return_json:
            output_code = '''
        import json
        print(json.dumps(result))
'''
        else:
            output_code = '''
        print(str(result))
'''

        wrapper_code = f'''
import sys
sys.setrecursionlimit(10000)

from mpmath import mp
mp.dps = {precision_dps}  # High precision

{code}

if __name__ == "__main__":
    try:
        result = proposed_solution()
        {output_code}
    except Exception as e:
        print(f"EXECUTION_ERROR: {{type(e).__name__}}: {{e}}", file=sys.stderr)
        sys.exit(1)
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(wrapper_code)
        temp_path = f.name

    python_exe = get_python_executable()

    start_time = time.time()
    try:
        result = subprocess.run(
            [python_exe, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        execution_time_ms = int((time.time() - start_time) * 1000)

        if result.returncode != 0:
            error_msg = result.stderr.strip() or "Unknown execution error"

            # Detect syntax errors
            if "SyntaxError" in error_msg:
                return ExecutionResult(
                    status=ExecutionStatus.SYNTAX_ERROR,
                    error_message=error_msg,
                    execution_time_ms=execution_time_ms
                )

            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                error_message=error_msg,
                execution_time_ms=execution_time_ms
            )

        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            output=result.stdout.strip(),
            execution_time_ms=execution_time_ms
        )

    except subprocess.TimeoutExpired:
        execution_time_ms = int((time.time() - start_time) * 1000)
        return ExecutionResult(
            status=ExecutionStatus.TIMEOUT,
            error_message=f"Execution timed out after {timeout} seconds",
            execution_time_ms=execution_time_ms
        )
    except Exception as e:
        execution_time_ms = int((time.time() - start_time) * 1000)
        return ExecutionResult(
            status=ExecutionStatus.RUNTIME_ERROR,
            error_message=f"Execution failed: {e}",
            execution_time_ms=execution_time_ms
        )
    finally:
        Path(temp_path).unlink(missing_ok=True)
