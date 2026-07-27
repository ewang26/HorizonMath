"""Configuration shared by the trusted launcher and the untrusted-agent runtime."""

from __future__ import annotations

from pathlib import Path

APP_NAME = "horizonmath-codex-agent-eval"
STATE_VOLUME_NAME = "horizonmath-codex-agent-state-v1"

DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_EFFORT = "xhigh"
DEFAULT_CONCURRENCY = 4
PROBLEM_TIMEOUT_SECONDS = 3 * 60 * 60
PERMISSIBILITY_MODEL = "gpt-5.6-terra"
PERMISSIBILITY_EFFORT = "high"
PERMISSIBILITY_ROUNDS = 3
PERMISSIBILITY_CONCURRENCY = 8
PERMISSIBILITY_TIMEOUT_SECONDS = 20 * 60
MAX_SANDBOX_SECONDS = 23 * 60 * 60
RUNTIME_SAFETY_MARGIN_SECONDS = 90 * 60

REMOTE_AUTH_ROOT = Path("/codex-home")
REMOTE_STATE_ROOT = Path("/state")
REMOTE_WORK_ROOT = Path("/workspaces")
REMOTE_RUNTIME_ROOT = Path("/opt/horizonmath_agent_runtime")
REMOTE_FORBIDDEN_ROOT = Path("/forbidden")

OPENAI_EGRESS_ALLOWLIST = (
    "chatgpt.com",
    "*.chatgpt.com",
    "openai.com",
    "*.openai.com",
)

AGENT_PERMISSION_PROFILE = "horizonmath-agent"


def codex_config_toml() -> str:
    """Return the clean, least-privilege Codex configuration used on Modal."""

    return (Path(__file__).parent / "runtime" / "config.toml").read_text()


def sandbox_timeout_seconds(
    problem_count: int,
    concurrency: int,
    problem_timeout_seconds: int,
) -> int:
    """Compute a bounded outer Sandbox lifetime for a batch."""

    if problem_count < 1:
        raise ValueError("problem_count must be positive")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if problem_timeout_seconds < 1:
        raise ValueError("problem_timeout_seconds must be positive")

    waves = (problem_count + concurrency - 1) // concurrency
    requested = waves * problem_timeout_seconds + RUNTIME_SAFETY_MARGIN_SECONDS
    if requested > MAX_SANDBOX_SECONDS:
        raise ValueError(
            "This batch could exceed Modal's 24-hour Sandbox limit. "
            "Select fewer problems or raise concurrency."
        )
    return requested
