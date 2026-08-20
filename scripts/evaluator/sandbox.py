"""Sandboxed execution module for running proposed_solution() code safely."""

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional


_TRANSPORT_TYPE_KEY = "__horizonmath_transport_type__"


def decode_json_result(value: Any) -> Any:
    """Decode the safe tagged values emitted by the construction subprocess."""
    if isinstance(value, list):
        return [decode_json_result(item) for item in value]
    if not isinstance(value, dict):
        return value

    transport_type = value.get(_TRANSPORT_TYPE_KEY)
    if transport_type == "tuple" and set(value) == {_TRANSPORT_TYPE_KEY, "items"}:
        items = value["items"]
        if not isinstance(items, list):
            raise ValueError("invalid tuple in sandbox JSON transport")
        return tuple(decode_json_result(item) for item in items)

    if transport_type == "mapping" and set(value) == {_TRANSPORT_TYPE_KEY, "items"}:
        items = value["items"]
        if not isinstance(items, list):
            raise ValueError("invalid mapping in sandbox JSON transport")
        decoded = {}
        for item in items:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("invalid mapping item in sandbox JSON transport")
            key = decode_json_result(item[0])
            try:
                if key in decoded:
                    raise ValueError("duplicate mapping key in sandbox JSON transport")
                decoded[key] = decode_json_result(item[1])
            except TypeError as exc:
                raise ValueError("unhashable mapping key in sandbox JSON transport") from exc
        return decoded

    return {key: decode_json_result(item) for key, item in value.items()}


def load_json_result(output: str) -> Any:
    """Parse and decode a JSON construction result from the sandbox."""
    return decode_json_result(json.loads(output))


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
        return_json: If True, serialize the result through the safe JSON witness
                    transport used by construction problems
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
            serializer_body = f'''
import json
from collections.abc import Iterable, Mapping
from fractions import Fraction

import networkx as nx
import sympy as sp

_TRANSPORT_TYPE_KEY = {_TRANSPORT_TYPE_KEY!r}


def _complex_to_exact_string(value):
    if not value.real.is_integer() or not value.imag.is_integer():
        raise TypeError("only complex values with integral components are supported")
    real = int(value.real)
    imag = int(value.imag)
    if imag == 0:
        return str(real)
    if real == 0:
        if imag == 1:
            return "I"
        if imag == -1:
            return "-I"
        return f"{{imag}}*I"
    sign = "+" if imag > 0 else "-"
    coefficient = "" if abs(imag) == 1 else str(abs(imag)) + "*"
    return f"{{real}}{{sign}}{{coefficient}}I"


def _to_json_transport(value, *, preserve_tuple=False):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("ascii", errors="strict")
    if isinstance(value, complex):
        return _complex_to_exact_string(value)
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, nx.Graph):
        nodes = list(value.nodes())
        try:
            nodes = sorted(nodes)
        except TypeError:
            pass
        edge_key = "arcs" if value.is_directed() else "edges"
        return {{
            "vertices": _to_json_transport(nodes, preserve_tuple=True),
            edge_key: _to_json_transport(list(value.edges()), preserve_tuple=True),
        }}
    if hasattr(value, "tolist") and not isinstance(value, Mapping):
        return _to_json_transport(value.tolist())
    if isinstance(value, sp.Basic) and isinstance(value, Iterable):
        return [_to_json_transport(item) for item in value]
    if isinstance(value, sp.Basic):
        return str(value)
    if isinstance(value, tuple):
        if not preserve_tuple:
            return [_to_json_transport(item) for item in value]
        return {{
            _TRANSPORT_TYPE_KEY: "tuple",
            "items": [
                _to_json_transport(item, preserve_tuple=True) for item in value
            ],
        }}
    if isinstance(value, Mapping):
        if _TRANSPORT_TYPE_KEY not in value and all(
            isinstance(key, str) for key in value
        ):
            return {{key: _to_json_transport(item) for key, item in value.items()}}
        return {{
            _TRANSPORT_TYPE_KEY: "mapping",
            "items": [
                [
                    _to_json_transport(key, preserve_tuple=True),
                    _to_json_transport(item),
                ]
                for key, item in value.items()
            ],
        }}
    if isinstance(value, Iterable):
        return [
            _to_json_transport(item, preserve_tuple=preserve_tuple) for item in value
        ]
    raise TypeError(f"unsupported construction result type: {{type(value).__name__}}")
'''
            serializer_code = (
                "def __horizonmath_internal_serialize_construction_result(value):\n"
                + "\n".join(
                    f"    {line}" if line else ""
                    for line in serializer_body.strip("\n").splitlines()
                )
                + "\n    return json.dumps(_to_json_transport(value))"
            )
            serializer_code = "\n".join(
                f"        {line}" if line else ""
                for line in serializer_code.splitlines()
            )
            output_code = '''
        print(__horizonmath_internal_serialize_construction_result(result))
'''
        else:
            serializer_code = ""
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
{serializer_code}
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
