# Evaluating Codex agents in OpenAI-hosted cloud containers

This is the default agentic workflow when a user asks Codex to evaluate Codex on
HorizonMath. It is distinct from both local Codex conversations and the opt-in
GitHub-backed cloud CLI workflow.

## What “Codex cloud” means

A valid scored agent runs in an isolated VM/container hosted by OpenAI. The
container—not the user's machine—executes:

- Python, mpmath, and symbolic calculations;
- shell commands and subprocesses;
- tool calls and scratch-file work; and
- primary/subagent coordination.

Remote model inference is not sufficient. These are local and invalid:

- a task whose creation result contains `hostId: "local"`;
- a projectless task with an output or working directory on the user's machine;
- a local Codex thread, even if its model inference is cloud-based;
- a local collaboration subagent; and
- a task merely described as “cloud” without execution-runtime confirmation.

## Mandatory preflight: prove the runtime before sending prompts

Inspect the current task-creation capability before creating any scored task. A
usable creator must return metadata that explicitly confirms:

| Required fact | Required normalized value |
|---|---|
| Execution location | `openai-hosted` |
| Runtime kind | `codex-cloud-container` |
| Host | Non-empty and not `local`, `localhost`, or a loopback address |
| Cloud environment | Non-empty environment ID |
| Repository attachment | `none` |
| Model | Exact requested model |
| Reasoning effort | Exact requested effort |

Terms such as `projectless`, `conversation`, or `background` do not establish any
of these facts. Acceptance of `model` and `thinking` arguments is also not proof
that the created task retained them; the creation result must confirm them.

If the creator cannot return all required metadata, stop before sending a
benchmark prompt. Do not try a local task as a canary, do not use local subagents,
and do not replace the requested condition with ChatGPT Work or a direct model
API. Report that this Codex surface cannot create the required runtime.

## Prepare exact prompts

For the first ten problems using `gpt-5.6-sol` with ultra reasoning:

```bash
uv run scripts/prepare_conversation_benchmark.py \
  --range 0-9 \
  --model gpt-5.6-sol \
  --reasoning-effort ultra
```

The generated, timestamped directory contains:

```text
config.json
prompts.jsonl
cloud_runtime.jsonl
responses.jsonl
```

Every `prompts.jsonl` record contains the unmodified `system_message`, the
unmodified problem `prompt`, hashes for both, and the complete
`agent_task_prompt`. Send that complete string verbatim to one new primary Codex
cloud task. Never reconstruct, shorten, paraphrase, or correct it.

The canonical system text is embedded in the task unless the creation surface
exposes a custom system-role field. Such a run is text-identical but not
role-identical to the single-shot API condition; preserve that distinction in
the run metadata.

## Create, wait, and collect

Create one fresh repository-free OpenAI-hosted Codex task per manifest record.
The primary agent may create cloud subagents inside its cloud environment, but
must not delegate to agents executing on the user's machine.

Wait for every primary task to reach a terminal state. Record only the primary
agent's final response in `responses.jsonl`:

```json
{
  "problem_id": "example_problem",
  "problem_index": 0,
  "provider": "codex-conversation",
  "model": "gpt-5.6-sol",
  "response": "the primary agent's complete final response",
  "timestamp": "2026-01-01T00:00:00+00:00"
}
```

For each selected problem, also append exactly one terminal record to
`cloud_runtime.jsonl`. Copy the task-creation confirmation; do not infer it from
the prompt or fill fields from intended settings.

Successful task:

```json
{
  "problem_id": "example_problem",
  "problem_index": 0,
  "terminal_outcome": "completed",
  "task_created": true,
  "execution_location": "openai-hosted",
  "runtime_kind": "codex-cloud-container",
  "host_id": "non-local-host-id-from-creation-result",
  "cloud_environment_id": "environment-id-from-creation-result",
  "task_id": "task-id-from-creation-result",
  "repository_attachment": "none",
  "local_execution_used": false,
  "model": "gpt-5.6-sol",
  "reasoning_effort": "ultra",
  "creation_confirmation_source": "task-creation-response",
  "agent_task_prompt_sha256": "sha256 of the complete agent_task_prompt",
  "response_source": "primary-final-response",
  "response_sha256": "sha256 of the exact responses.jsonl response string"
}
```

If cloud task creation fails before a task exists, record a generation error and
the attempted cloud condition without claiming runtime proof:

```json
{
  "problem_id": "example_problem",
  "problem_index": 0,
  "terminal_outcome": "generation_error",
  "task_created": false,
  "error_stage": "task_creation",
  "attempted_runtime_kind": "codex-cloud-container",
  "local_execution_used": false,
  "model": "gpt-5.6-sol",
  "reasoning_effort": "ultra",
  "agent_task_prompt_sha256": "sha256 of the complete agent_task_prompt"
}
```

If an already-confirmed cloud task fails later, use
`terminal_outcome: "generation_error"`, `task_created: true`, and include all
confirmed cloud-runtime fields from the successful creation response.

## Validate and evaluate

Run the provenance gate first:

```bash
uv run scripts/validate_conversation_run.py \
  results/codex-conversation_<model>_<timestamp>/
```

It rejects:

- missing or duplicate terminal records;
- any scored response without confirmed OpenAI-hosted execution;
- `host_id: "local"` or another local/loopback host;
- missing cloud environment or task IDs;
- repository attachment;
- model or reasoning-effort mismatch;
- prompt hash mismatch;
- response hash mismatch; and
- a response not identified as the primary agent's final response.

Then run the trusted evaluator:

```bash
uv run scripts/evaluate_responses.py \
  results/codex-conversation_<model>_<timestamp>/
```

The evaluator repeats the provenance gate before executing any proposed solution.
Agentic numeric answers also require the Gemini compliance check. A missing
credential, malformed compliance response, API failure, hard-coded numerical
target, prohibited numerical integration, rationalized decimal approximation, or
other noncompliant construction fails the problem even when the number matches.

## GitHub-backed alternative

`scripts/run_agent_benchmark.py` uses a separate verifier-free Git repository and
the Codex cloud CLI. Use it only when the user explicitly asks for that
GitHub-backed implementation. Never silently substitute it for the repository-free
workflow above.
