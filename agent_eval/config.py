"""Configuration shared by the trusted launcher and the untrusted-agent runtime."""

from __future__ import annotations

from pathlib import Path

APP_NAME = "horizonmath-codex-agent-eval"
AUTH_VOLUME_NAME = "horizonmath-codex-auth-v1"
STATE_VOLUME_NAME = "horizonmath-codex-agent-state-v1"

DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_EFFORT = "xhigh"
DEFAULT_CONCURRENCY = 4
PROBLEM_TIMEOUT_SECONDS = 3 * 60 * 60
MAX_SANDBOX_SECONDS = 23 * 60 * 60
RUNTIME_SAFETY_MARGIN_SECONDS = 30 * 60

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

    return f"""\
model = "{DEFAULT_MODEL}"
model_reasoning_effort = "{DEFAULT_EFFORT}"
approval_policy = "never"
web_search = "disabled"
default_permissions = "{AGENT_PERMISSION_PROFILE}"
cli_auth_credentials_store = "file"
check_for_update_on_startup = false

[history]
persistence = "save-all"

[features]
apps = false
multi_agent = false
memories = false
remote_plugin = false
skill_mcp_dependency_install = false
web_search = false
web_search_cached = false
web_search_request = false

[tools]
view_image = false

[shell_environment_policy]
inherit = "core"
exclude = [
  "*API_KEY*",
  "*ACCESS_TOKEN*",
  "*AUTH*",
  "*CREDENTIAL*",
  "*SECRET*",
  "*TOKEN*",
  "CODEX_*",
  "MODAL_*",
]

[permissions.{AGENT_PERMISSION_PROFILE}]
description = "Write only inside the active problem workspace; no network or evaluator access."

[permissions.{AGENT_PERMISSION_PROFILE}.filesystem]
":minimal" = "read"
"{REMOTE_AUTH_ROOT}" = "deny"
"{REMOTE_STATE_ROOT}" = "deny"
"{REMOTE_RUNTIME_ROOT}" = "deny"
"{REMOTE_FORBIDDEN_ROOT}" = "deny"

[permissions.{AGENT_PERMISSION_PROFILE}.filesystem.":workspace_roots"]
"." = "write"
"**/*.env" = "deny"

[permissions.{AGENT_PERMISSION_PROFILE}.network]
enabled = false
"""


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
