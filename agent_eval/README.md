# Modal Codex agent evaluation

This runner evaluates HorizonMath with long-horizon Codex coding agents using
ChatGPT-managed Codex usage rather than an OpenAI Platform API key.

Each problem receives an independent Codex thread, a clean Git repository,
local Python/compiler tools, and a hard 10,800-second limit. The default run is
`gpt-5.6-sol` at `xhigh`, with four concurrent threads inside one Codex
app-server process.

## Security boundary

The trusted local launcher reads `data/problems_full.json`, then uploads a
sanitized manifest containing only:

- problem id and index;
- the public problem prompt and evaluation mode;
- benchmark/agent instructions; and
- a hash of the public prompt.

Numeric answers, hidden test points, baselines, source URLs, validators,
numerics scripts, and evaluator source are never included in the agent image or
state manifest. The image is built with only `agent_eval/runtime/`.

The outer Modal Sandbox:

- runs under Modal's Sandbox isolation;
- can reach only `chatgpt.com`, `openai.com`, and their subdomains so the Codex
  controller can authenticate and request inference;
- keeps the device-authorized Codex session only on its ephemeral filesystem,
  while mounting a separate state Volume for non-secret checkpoints; and
- has no benchmark source tree or Modal secrets.

Codex command subprocesses use a custom permission profile that denies the
ephemeral controller auth directory, state Volume, runtime source, and a canary
path. It grants writes only to the active problem workspace. Every tool shell
starts through a compiled seccomp launcher that sets `NO_NEW_PRIVS` and denies
socket syscalls before Bash starts; all descendants inherit the restriction.
This is necessary because Modal's gVisor kernel does not support the nested
network namespace setup used by Codex's Linux sandbox. The controller remains
outside that filter and is separately limited to OpenAI/ChatGPT domains.
Web search, apps, plugins, memories, image tools, and multi-agent tools are
disabled.

Before starting any model thread, the worker runs a deterministic nested
sandbox self-test. The run aborts unless a sandboxed command can write its
workspace but cannot read protected mounts or open a direct-IP outbound socket.

Candidate solutions are scored separately. `cloud_scorer.py` sends only code
and public function arguments to a fresh `block_network=True` Modal Sandbox.
Expected values remain in the trusted scorer. Validators and ground truth are
never mounted into candidate execution.

## Authentication

Install the evaluation dependencies:

```bash
uv sync --group agent-eval --group dev
```

The launcher never reads or uploads the local Codex credential. It starts a
fresh Modal Sandbox and displays an OpenAI device-login URL and one-time code.
Complete that login interactively. OpenAI then issues a Codex session directly
to that Sandbox. The session is stored only under the Sandbox's ephemeral
`/codex-home`; it is not mounted from or copied to a Modal Volume and disappears
when the Sandbox terminates.

Each new or resumed Sandbox therefore requires a new device login. Only the
sanitized manifest, progress checkpoints, thread ids, and responses persist in
the state Volume.

No `OPENAI_API_KEY` or `CODEX_API_KEY` is accepted by the worker.

## Preflight and launch

An optional standalone preflight performs its own ephemeral device login:

```bash
uv run --group agent-eval python -m agent_eval.modal_runner preflight
```

Launch the first ten zero-based dataset entries. This command stays attached
until device login and all security preflights succeed, then detaches after the
worker has started:

```bash
uv run --group agent-eval python -m agent_eval.modal_runner launch \
  --start 0 \
  --count 10 \
  --concurrency 4
```

The command prints a run id and Sandbox id and writes a gitignored launch
record under `results/modal_<run-id>/launch.json`.

Inspect progress:

```bash
uv run --group agent-eval python -m agent_eval.modal_runner status \
  --run-id <run-id>
```

Download checkpoints and responses:

```bash
uv run --group agent-eval python -m agent_eval.modal_runner download \
  --run-id <run-id>
```

Score a completed run with networkless Modal candidate execution:

```bash
uv run --group agent-eval python -m agent_eval.cloud_scorer \
  --run-id <run-id>
```

## Full dataset and recovery

Modal Sandboxes have a 24-hour maximum lifetime. At four-way concurrency, use
batches of at most 28 problems; smaller batches provide more recovery margin.
Each batch uses one app-server and one ephemeral device-authorized session.
Run batches sequentially to keep subscription load controlled and make the
usage attributable.

Every problem checkpoint is committed to the state Volume when it starts,
receives a Codex thread/turn id, and finishes. Re-running the same run id after
a new device login skips completed and timed-out entries and can resume stored
thread ids. Failed entries remain visible for deliberate retry or diagnosis.

## Validation

```bash
GOOGLE_API_KEY=dummy uv run --group agent-eval --group dev pytest -q
```

Tests enforce the exact model, `xhigh` effort, three-hour problem limit,
sanitized manifest schema, permission policy, Modal lifetime calculation, and
removal of expected values from both local and cloud candidate-execution
payloads.
