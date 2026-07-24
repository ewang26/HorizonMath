"""Sandboxed execution module for running proposed_solution() code safely."""

import json
import math
import os
import platform
import resource
import shutil
import signal
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
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_ADDRESS_SPACE_BYTES = 4 * 1024 * 1024 * 1024
MAX_OPEN_FILES = 128
MAX_LINUX_PROCESSES = 64


def get_python_executable() -> str:
    """Get the Python executable, preferring venv if available."""
    script_dir = Path(__file__).parent.parent
    project_root = script_dir.parent
    venv_python = project_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _runtime_site_packages(python_exe: Path) -> Path:
    candidates = sorted((python_exe.parent.parent / "lib").glob("python*/site-packages"))
    if not candidates:
        raise RuntimeError(f"Cannot locate venv site-packages for {python_exe}")
    return candidates[-1].resolve()


def _sandbox_environment(
    temp_dir: Path,
    *,
    python_exe: Path | None = None,
) -> dict[str, str]:
    """Return a credential-free environment for untrusted answer execution."""

    environment = {
        "HOME": str(temp_dir),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": str(temp_dir),
    }
    if python_exe is not None:
        environment["PYTHONPATH"] = str(_runtime_site_packages(python_exe))
    return environment


def _escape_sandbox_profile_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def _macos_sandbox_command(
    runtime_python: Path,
    wrapper_path: Path,
    temp_dir: Path,
) -> tuple[list[str], Path]:
    """Build a fail-closed macOS sandbox-exec command."""

    sandbox_exec = shutil.which("sandbox-exec")
    if not sandbox_exec:
        raise RuntimeError(
            "sandbox-exec is required for trusted evaluation on macOS"
        )

    trusted_root = Path(__file__).resolve().parents[2]
    denied_paths = [
        Path.home().resolve(),
        trusted_root,
        Path("/private/tmp"),
    ]
    deny_reads = "\n".join(
        f'(deny file-read* (subpath "{_escape_sandbox_profile_path(path)}"))'
        for path in denied_paths
        if path.exists()
    )
    deny_writes = "\n".join(
        f'(deny file-write* (subpath "{_escape_sandbox_profile_path(path)}"))'
        for path in denied_paths
        if path.exists()
    )
    profile = f"""(version 1)
(allow default)
(deny network*)
(deny process-fork)
{deny_reads}
{deny_writes}
"""
    profile_path = temp_dir / "sandbox.sb"
    profile_path.write_text(profile, encoding="utf-8")
    return (
        [
            sandbox_exec,
            "-f",
            str(profile_path),
            str(runtime_python),
            "-I",
            "-B",
            str(wrapper_path),
        ],
        profile_path,
    )


def _clone_tree_macos(source: Path, destination: Path) -> None:
    """Clone a directory with copy-on-write semantics on APFS."""

    result = subprocess.run(
        ["/bin/cp", "-cR", str(source), str(destination)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        try:
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
        except Exception as exc:
            raise RuntimeError(
                f"Could not copy sandbox runtime {source}: "
                f"{(result.stderr or result.stdout).strip()[:200]}; {exc}"
            ) from exc


def _prepare_macos_runtime(python_exe: Path, temp_dir: Path) -> Path:
    """Clone a matching Python plus the packages explicitly used by prompts."""

    base_prefix = Path(
        subprocess.run(
            [str(python_exe), "-I", "-c", "import sys; print(sys.base_prefix)"],
            check=True,
            capture_output=True,
            text=True,
            env=_sandbox_environment(temp_dir),
        ).stdout.strip()
    ).resolve()
    runtime_dir = temp_dir / "runtime"
    _clone_tree_macos(base_prefix, runtime_dir)

    site_packages = _runtime_site_packages(python_exe)
    packages_dir = temp_dir / "packages"
    packages_dir.mkdir()
    for package_name in ("mpmath", "numpy"):
        source = site_packages / package_name
        if not source.is_dir():
            raise RuntimeError(
                f"{package_name} is missing from the evaluator venv"
            )
        _clone_tree_macos(source, packages_dir / package_name)

    version = subprocess.run(
        [str(python_exe), "-I", "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        check=True,
        capture_output=True,
        text=True,
        env=_sandbox_environment(temp_dir),
    ).stdout.strip()
    runtime_python = runtime_dir / "bin" / f"python{version}"
    if not runtime_python.exists():
        raise RuntimeError(f"Cloned Python executable is missing: {runtime_python}")
    return runtime_python


def _linux_sandbox_command(
    python_exe: Path,
    wrapper_path: Path,
    temp_dir: Path,
) -> tuple[list[str], None]:
    """Build a Linux bubblewrap command with no trusted-repo or network access."""

    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError(
            "bubblewrap (bwrap) is required for trusted evaluation on Linux"
        )
    base_prefix = Path(
        subprocess.run(
            [str(python_exe), "-I", "-c", "import sys; print(sys.base_prefix)"],
            check=True,
            capture_output=True,
            text=True,
            env=_sandbox_environment(temp_dir),
        ).stdout.strip()
    ).resolve()
    base_python = python_exe.resolve()
    site_packages = _runtime_site_packages(python_exe)
    runtime_python = Path("/runtime/bin") / base_python.name

    command = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--clearenv",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    for system_path in (Path("/usr"), Path("/lib"), Path("/lib64")):
        if system_path.exists():
            command.extend(["--ro-bind", str(system_path), str(system_path)])
    command.extend(
        [
            "--ro-bind",
            str(base_prefix),
            "/runtime",
            "--ro-bind",
            str(site_packages),
            "/packages",
            "--ro-bind",
            str(temp_dir),
            "/sandbox",
            "--chdir",
            "/sandbox",
            "--setenv",
            "HOME",
            "/tmp",
            "--setenv",
            "PATH",
            "/usr/bin:/bin",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--setenv",
            "PYTHONPATH",
            "/packages",
            str(runtime_python),
            "-s",
            "-B",
            f"/sandbox/{wrapper_path.name}",
        ]
    )
    return command, None


def _run_isolated_wrapper(wrapper_code: str, timeout: float) -> ExecutionResult:
    """Execute one wrapper under an OS-enforced filesystem/network boundary."""

    temp_parent = (
        "/private/var/tmp"
        if platform.system() == "Darwin" and Path("/private/var/tmp").is_dir()
        else None
    )
    with tempfile.TemporaryDirectory(
        prefix="horizonmath-eval-",
        dir=temp_parent,
    ) as temp:
        temp_dir = Path(temp).resolve()
        wrapper_path = temp_dir / "answer.py"
        wrapper_path.write_text(wrapper_code, encoding="utf-8")
        # Preserve the venv path rather than resolving its interpreter symlink;
        # the policy must allow both venv packages and the base interpreter.
        python_exe = Path(get_python_executable()).absolute()

        try:
            system = platform.system()
            if system == "Darwin":
                runtime_python = _prepare_macos_runtime(python_exe, temp_dir)
                command, _ = _macos_sandbox_command(
                    runtime_python, wrapper_path, temp_dir
                )
            elif system == "Linux":
                command, _ = _linux_sandbox_command(
                    python_exe, wrapper_path, temp_dir
                )
            else:
                raise RuntimeError(
                    f"No trusted answer sandbox is implemented for {system}"
                )
        except Exception as exc:
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                error_message=f"Secure sandbox unavailable: {exc}",
                execution_time_ms=0,
            )

        def set_resource_limits() -> None:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            resource.setrlimit(
                resource.RLIMIT_FSIZE,
                (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES),
            )
            resource.setrlimit(
                resource.RLIMIT_NOFILE,
                (MAX_OPEN_FILES, MAX_OPEN_FILES),
            )
            cpu_seconds = max(1, math.ceil(timeout) + 1)
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (cpu_seconds, cpu_seconds),
            )
            if system == "Linux":
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (MAX_ADDRESS_SPACE_BYTES, MAX_ADDRESS_SPACE_BYTES),
                )
                resource.setrlimit(
                    resource.RLIMIT_NPROC,
                    (MAX_LINUX_PROCESSES, MAX_LINUX_PROCESSES),
                )

        stdout_path = temp_dir / "stdout"
        stderr_path = temp_dir / "stderr"
        start_time = time.time()
        process: subprocess.Popen | None = None
        try:
            with stdout_path.open("w+b") as stdout_file, stderr_path.open(
                "w+b"
            ) as stderr_file:
                process = subprocess.Popen(
                    command,
                    cwd=temp_dir,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    env=_sandbox_environment(
                        temp_dir,
                        python_exe=python_exe if system == "Linux" else None,
                    ),
                    start_new_session=True,
                    preexec_fn=set_resource_limits,
                )
                process.wait(timeout=max(timeout, 0.001))
                stdout_size = os.fstat(stdout_file.fileno()).st_size
                stderr_size = os.fstat(stderr_file.fileno()).st_size
                stdout_file.seek(0)
                stderr_file.seek(0)
                stdout = stdout_file.read(MAX_OUTPUT_BYTES + 1).decode(
                    errors="replace"
                )
                stderr = stderr_file.read(MAX_OUTPUT_BYTES + 1).decode(
                    errors="replace"
                )
            execution_time_ms = int((time.time() - start_time) * 1000)
            output_limit_hit = (
                stdout_size > MAX_OUTPUT_BYTES
                or stderr_size > MAX_OUTPUT_BYTES
                or (
                    process.returncode != 0
                    and (
                        stdout_size >= MAX_OUTPUT_BYTES
                        or stderr_size >= MAX_OUTPUT_BYTES
                    )
                )
            )
            if output_limit_hit:
                return ExecutionResult(
                    status=ExecutionStatus.RUNTIME_ERROR,
                    error_message=(
                        f"Execution output exceeded {MAX_OUTPUT_BYTES} bytes"
                    ),
                    execution_time_ms=execution_time_ms,
                )
            if process.returncode != 0:
                error_msg = stderr.strip() or "Unknown execution error"
                status = (
                    ExecutionStatus.SYNTAX_ERROR
                    if "SyntaxError" in error_msg
                    else ExecutionStatus.RUNTIME_ERROR
                )
                return ExecutionResult(
                    status=status,
                    error_message=error_msg,
                    execution_time_ms=execution_time_ms,
                )
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                output=stdout.strip(),
                execution_time_ms=execution_time_ms,
            )
        except subprocess.TimeoutExpired:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            execution_time_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error_message=f"Execution timed out after {timeout:g} seconds",
                execution_time_ms=execution_time_ms,
            )
        except Exception as exc:
            execution_time_ms = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                error_message=f"Execution failed: {exc}",
                execution_time_ms=execution_time_ms,
            )
        finally:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def _build_wrapper(
    code: str,
    *,
    precision_dps: int,
    return_json: bool,
    call_args: list | None,
) -> str:
    """Build a wrapper containing no expected values or other scoring secrets."""

    if call_args is None:
        invocation = "result = proposed_solution()"
    else:
        invocation = (
            f"_args = {json.dumps(call_args)}\n"
            "_args = [mp.mpf(a) if isinstance(a, str) else a for a in _args]\n"
            "result = proposed_solution(*_args)"
        )
    output = "print(json.dumps(result))" if return_json else "print(str(result))"
    return f"""import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "packages"))
sys.setrecursionlimit(10000)

from mpmath import mp
mp.dps = {precision_dps}

{code}

if __name__ == "__main__":
    try:
        {invocation.replace(chr(10), chr(10) + "        ")}
        {output}
    except Exception as e:
        print(f"EXECUTION_ERROR: {{type(e).__name__}}: {{e}}", file=sys.stderr)
        sys.exit(1)
"""


def execute_sandboxed(
    code: str,
    timeout: int = DEFAULT_TIMEOUT,
    precision_dps: int = DEFAULT_PRECISION_DPS,
    return_json: bool = False,
    test_points: list[dict] | None = None,
) -> ExecutionResult:
    """
    Execute proposed_solution() in an OS-enforced sandbox.

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
    if test_points is None:
        wrapper = _build_wrapper(
            code,
            precision_dps=precision_dps,
            return_json=return_json,
            call_args=None,
        )
        return _run_isolated_wrapper(wrapper, timeout)

    # Run each hidden point in a fresh process. The candidate sees only the
    # arguments passed to that invocation and never receives expected values or
    # the other hidden points.
    deadline = time.monotonic() + timeout
    outputs: list[str | None] = []
    total_execution_ms = 0
    for point in test_points:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error_message=f"Execution timed out after {timeout:g} seconds",
                execution_time_ms=total_execution_ms,
            )
        wrapper = _build_wrapper(
            code,
            precision_dps=precision_dps,
            return_json=False,
            call_args=point["args"],
        )
        result = _run_isolated_wrapper(wrapper, remaining)
        total_execution_ms += result.execution_time_ms or 0
        if result.status != ExecutionStatus.SUCCESS:
            outputs.append(None)
        else:
            outputs.append(result.output)
    return ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        output=json.dumps(outputs),
        execution_time_ms=total_execution_ms,
    )
