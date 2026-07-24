# HorizonMath Codex agent instructions

## Default coding-agent evaluation workflow

When a user asks Codex to evaluate coding agents on HorizonMath, use
`scripts/run_agent_benchmark.py`. It is the supported Codex cloud workflow.

The architecture has two deliberately different runtimes:

- The coordinator and trusted evaluator run from this HorizonMath checkout.
- Every scored solver runs as a separate `codex cloud exec` task in an
  OpenAI-managed Codex cloud container.

Local orchestration is allowed. Local solving is not. Do not use `create_thread`,
projectless tasks, worktrees, or local collaboration subagents as scored solvers.
Model inference being remote does not make a local Codex task a cloud task.

Codex cloud checks out a repository. There is no supported repository-free Codex
cloud-task target in the current CLI/app workflow. Never claim otherwise. Use the
separate verifier-free companion repository created by
`scripts/create_agent_workspace.py`; never attach scored agents to the trusted
HorizonMath repository.

## Required setup

Before a scored run, confirm all of the following:

1. The companion repository has independent git history and contains only the
   generated allowlisted seed files.
2. A Codex cloud environment is connected to that companion GitHub repository.
3. Agent-phase internet access is **Off**.
4. The environment setup, maintenance script, and cache contain no HorizonMath
   checkout, validators, ground truth, baselines, or evaluation artifacts.
5. The cloud runtime exposes the required Codex goal tools.
6. A one-problem canary has completed and been evaluated in that exact
   environment before a multi-problem run.

The runner checks the local and remote companion-repository history and records
the required operator attestations. If any prerequisite cannot be established,
stop before submitting scored tasks.

## Required run behavior

1. Use one fresh Codex cloud task per selected problem. The runner uses one
   attempt per problem and may submit tasks in parallel.
2. Pass the requested model and reasoning effort explicitly to every
   `codex cloud exec` submission using the runner's `--model` and
   `--reasoning-effort` options.
3. The runner must generate `prompts.jsonl` before task submission. Each cloud
   prompt must contain the complete canonical system message and problem prompt
   without paraphrasing, shortening, reconstruction, or correction.
4. The primary cloud solver may spawn subagents inside its hosted environment.
   Do not supplement it with local collaboration agents.
5. Wait for every selected task to reach a terminal outcome. Record only the
   answer produced by the primary cloud task in `responses.jsonl`.
6. Run `scripts/evaluate_responses.py` in the trusted checkout after generation.
   Agentic numeric runs require the Gemini compliance check and fail closed if
   its credential is absent or the check errors.
7. Treat hard-coded numerical targets, rationalized decimal approximations,
   numerical integration, and every other prohibited construction as failures
   even when numerically correct.
8. Report per-problem results, the compliant pass rate, generation failures, and
   links to `config.json`, `prompts.jsonl`, `cloud_tasks.jsonl`,
   `responses.jsonl`, `generation_errors.jsonl`, `evaluation.jsonl`, and
   `summary.json`.

## Model-verification limitation

The runner passes `model` and `model_reasoning_effort` as explicit Codex CLI
configuration overrides and records those requested settings. The current cloud
CLI returns a task ID and URL, but does not echo the resolved model or reasoning
effort in machine-readable task metadata. Do not describe the recorded request as
independent runtime attestation.

If the user requires machine-confirmed resolved model settings, fail closed and
state that the current `codex cloud` CLI cannot provide that evidence. If explicit
submission settings plus the cloud task record are acceptable, proceed.

## Example: dataset indices 1 through 10

`--range` is zero-based and inclusive. To run indices 1–10 with ten concurrent
cloud tasks:

```bash
uv run scripts/run_agent_benchmark.py \
  --env YOUR_CODEX_CLOUD_ENVIRONMENT \
  --agent-workspace ../HorizonMath-agent-workspace \
  --agent-repo-url git@github.com:YOUR_ORG/HorizonMath-agent-workspace.git \
  --model gpt-5.6-sol \
  --reasoning-effort ultra \
  --range 1-10 \
  --parallel 10 \
  --confirm-environment-isolated \
  --confirm-goal-tools-available \
  --confirm-agent-internet-off \
  --confirm-live-canary
```

Do not substitute `--range 0-9`; that selects different dataset indices.
