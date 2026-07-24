"""Shared utilities for HorizonMath coding-agent benchmark runs.

The trusted benchmark repository owns this module.  It submits tasks to a
separate, verifier-free Codex cloud repository and converts the resulting
single-file diff back into the response format used by evaluate_responses.py.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit


BENCHMARK_REPOSITORY = "github.com/ewang26/horizonmath"
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TASK_URL = re.compile(r"https?://\S+/codex/tasks/([^/?#\s]+)")
_TASK_STATE = re.compile(r"\[(PENDING|READY|APPLIED|ERROR)\]")


class CloudTaskError(RuntimeError):
    """A Codex cloud command failed or returned an unexpected result."""


class IsolationError(ValueError):
    """The requested cloud setup does not preserve verifier isolation."""


@dataclass(frozen=True)
class CommandResult:
    """Captured subprocess result."""

    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class TaskSubmission:
    """Codex cloud task identifier and browser URL."""

    task_id: str
    url: str


def strip_ansi(text: str) -> str:
    """Remove terminal color sequences from CLI output."""

    return _ANSI_ESCAPE.sub("", text)


def parse_task_submission(output: str) -> TaskSubmission:
    """Parse the URL printed by ``codex cloud exec``."""

    clean = strip_ansi(output)
    matches = list(_TASK_URL.finditer(clean))
    if not matches:
        raise CloudTaskError(
            "codex cloud exec did not print a task URL; "
            f"received: {clean.strip()[:300]!r}"
        )
    match = matches[-1]
    return TaskSubmission(task_id=match.group(1), url=match.group(0))


def parse_task_state(output: str) -> str:
    """Parse the state printed by ``codex cloud status``."""

    match = _TASK_STATE.search(strip_ansi(output))
    if not match:
        raise CloudTaskError(
            "codex cloud status returned no recognizable task state; "
            f"received: {strip_ansi(output).strip()[:300]!r}"
        )
    return match.group(1)


def sanitize_repository_url(url: str) -> str:
    """Strip credentials, query parameters, and fragments before logging a URL."""

    raw = url.strip()
    if not raw:
        raise IsolationError("--agent-repo-url must not be empty")
    if "://" not in raw:
        # Handles git@github.com:owner/repo.git and github.com/owner/repo.
        without_user = raw.split("@", 1)[-1]
        if ":" in without_user and "/" not in without_user.split(":", 1)[0]:
            host, path = without_user.split(":", 1)
            return f"{host}/{path}".rstrip("/")
        return without_user.rstrip("/")
    parsed = urlsplit(raw)
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    return urlunsplit((parsed.scheme, host + port, parsed.path, "", "")).rstrip("/")


def repository_identity(url: str) -> str:
    """Return a normalized host/path identity for repository comparisons."""

    sanitized = sanitize_repository_url(url).lower()
    if "://" in sanitized:
        parsed = urlsplit(sanitized)
        identity = f"{parsed.hostname or ''}{parsed.path}"
    else:
        identity = sanitized
    return identity.removesuffix(".git").strip("/")


def validate_isolated_repository(agent_repo_url: str) -> str:
    """Reject the trusted benchmark repository as a cloud agent workspace."""

    sanitized = sanitize_repository_url(agent_repo_url)
    if repository_identity(sanitized) == BENCHMARK_REPOSITORY:
        raise IsolationError(
            "The Codex cloud environment cannot use ewang26/HorizonMath: that "
            "repository exposes validators and hidden ground truth. Point it to "
            "a separate verifier-free repository with independent git history."
        )
    return sanitized


def answer_path(run_id: str, problem_id: str) -> str:
    """Build a unique, safe answer path within the agent repository."""

    safe = re.compile(r"^[A-Za-z0-9_.-]+$")
    if not safe.fullmatch(run_id) or not safe.fullmatch(problem_id):
        raise ValueError("run_id and problem_id must contain only safe path characters")
    return f"answers/{run_id}/{problem_id}.md"


def build_agent_prompt(
    *,
    problem_id: str,
    system_message: str,
    problem_prompt: str,
    output_path: str,
) -> str:
    """Create a cloud-agent prompt without copying any hidden problem fields."""

    objective = (
        f"Solve HorizonMath problem {problem_id} and save the final proposed "
        f"solution to {output_path}."
    )
    return f"""You are being evaluated as a coding agent on one HorizonMath problem.
You have full use of the tools installed in this cloud container. Do all reasoning,
experimentation, and computation in the cloud.

Required first action: use Codex goal setting to create this exact goal:
{objective}
Keep that goal active while you work and mark it complete only after the required
answer file has been written.

The benchmark instructions and problem below are the same text supplied to the
single-shot model runners. Follow them exactly.

<benchmark_system_message>
{system_message}
</benchmark_system_message>

<problem>
{problem_prompt}
</problem>

Submission contract:
- Put the complete response that should be evaluated, including the required
  proposed_solution() function, in `{output_path}`.
- The trusted evaluator will consume that file verbatim. Do not put the answer only
  in your final chat message.
- Do not add, delete, or modify any other tracked file. Use /tmp for scratch work,
  and remove any accidental repository changes before finishing.
- You are intentionally not given validators, evaluator code, baselines beyond
  those stated in the problem, numeric ground truth, or hidden test points. Do not
  attempt to locate or reconstruct benchmark-private evaluation artifacts.
"""


def extract_new_file_from_diff(diff: str, expected_path: str) -> str:
    """Extract one newly-created answer file from a unified git diff.

    The cloud task is allowed to return exactly one tracked change.  Rejecting all
    other paths prevents an agent from smuggling unrelated repository state into
    the trusted evaluation phase.
    """

    expected = PurePosixPath(expected_path)
    if expected.is_absolute() or ".." in expected.parts:
        raise ValueError("expected_path must be a safe repository-relative path")

    sections = re.split(r"(?=^diff --git )", diff, flags=re.MULTILINE)
    sections = [section for section in sections if section.startswith("diff --git ")]
    if len(sections) != 1:
        raise CloudTaskError(
            f"Expected exactly one changed file, but cloud diff contains {len(sections)}"
        )

    section = sections[0]
    first_line = section.splitlines()[0]
    expected_header = f"diff --git a/{expected_path} b/{expected_path}"
    if first_line != expected_header:
        raise CloudTaskError(
            f"Cloud task changed an unexpected path: {first_line!r}; "
            f"expected only {expected_path!r}"
        )

    lines = section.splitlines()
    try:
        old_index = lines.index("--- /dev/null")
        new_index = lines.index(f"+++ b/{expected_path}")
    except ValueError as exc:
        raise CloudTaskError(
            f"{expected_path!r} must be a newly created file in the cloud diff"
        ) from exc
    if new_index != old_index + 1:
        raise CloudTaskError("Malformed unified diff header for answer file")

    content: list[str] = []
    in_hunk = False
    saw_hunk = False
    for line in lines[new_index + 1 :]:
        if line.startswith("@@ "):
            in_hunk = True
            saw_hunk = True
            continue
        if not in_hunk:
            continue
        if line == r"\ No newline at end of file":
            continue
        if line.startswith("+"):
            content.append(line[1:])
        elif line.startswith(" "):
            # A new file should have no context lines.
            raise CloudTaskError("Unexpected context in new answer-file diff")
        elif line.startswith("-"):
            raise CloudTaskError("Unexpected deletion in new answer-file diff")
        else:
            raise CloudTaskError(f"Malformed answer-file diff line: {line[:80]!r}")

    if not saw_hunk:
        raise CloudTaskError("Cloud diff contains no answer-file content")
    answer = "\n".join(content)
    if not answer.strip():
        raise CloudTaskError("Cloud task created an empty answer file")
    if section.endswith("\n") and not section.rstrip("\n").endswith(
        r"\ No newline at end of file"
    ):
        answer += "\n"
    return answer


class CodexCloudClient:
    """Small wrapper around the official ``codex cloud`` CLI."""

    def __init__(
        self,
        *,
        codex_binary: str = "codex",
        cwd: Path | None = None,
        command_timeout: float = 120.0,
    ) -> None:
        self.codex_binary = codex_binary
        self.cwd = cwd
        self.command_timeout = command_timeout

    def _run(
        self,
        args: list[str],
        *,
        stdin: str | None = None,
        allow_failure: bool = False,
    ) -> CommandResult:
        try:
            completed = subprocess.run(
                [self.codex_binary, *args],
                cwd=self.cwd,
                input=stdin,
                text=True,
                capture_output=True,
                timeout=self.command_timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise CloudTaskError(
                f"Codex CLI not found at {self.codex_binary!r}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise CloudTaskError(
                f"Codex cloud command timed out after {self.command_timeout:g}s"
            ) from exc

        result = CommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        if result.returncode and not allow_failure:
            detail = strip_ansi(result.stderr or result.stdout).strip()
            raise CloudTaskError(
                f"Codex cloud command failed ({result.returncode}): {detail[:500]}"
            )
        return result

    def preflight(self) -> None:
        """Verify that the installed CLI exposes cloud task submission."""

        result = self._run(["cloud", "exec", "--help"])
        if "--env" not in result.stdout or "--branch" not in result.stdout:
            raise CloudTaskError(
                "Installed Codex CLI does not expose the required cloud exec options"
            )

    def submit(
        self,
        *,
        prompt: str,
        environment: str,
        branch: str,
    ) -> TaskSubmission:
        """Submit one cloud task, passing the full problem prompt over stdin."""

        result = self._run(
            [
                "cloud",
                "exec",
                "--env",
                environment,
                "--attempts",
                "1",
                "--branch",
                branch,
                "-",
            ],
            stdin=prompt,
        )
        return parse_task_submission(result.stdout)

    def status(self, task_id: str) -> str:
        """Return PENDING, READY, APPLIED, or ERROR."""

        # The official CLI exits 1 for every non-READY state, including PENDING,
        # so stdout is authoritative and nonzero is intentionally allowed.
        result = self._run(
            ["cloud", "status", task_id],
            allow_failure=True,
        )
        return parse_task_state(result.stdout or result.stderr)

    def diff(self, task_id: str) -> str:
        """Fetch the first (and only) attempt's unified diff."""

        result = self._run(["cloud", "diff", task_id, "--attempt", "1"])
        return result.stdout
