# Evaluating Codex cloud coding agents

HorizonMath supports an agentic generation mode in addition to the single-shot
model providers. Each problem becomes an independent Codex cloud task with terminal
and tool access. The cloud agent does its reasoning and computation remotely, writes
one answer file, and the existing trusted evaluator scores that answer locally.

This repository-backed workflow is the supported way to automate Codex cloud
evaluations. The current app's repository-free/projectless Codex tasks run on the
calling host, while Codex cloud tasks check out a connected repository. Do not
describe projectless local tasks as cloud execution.

## Trust boundary

Do not attach the cloud environment to `ewang26/HorizonMath`. That repository
contains `validators/`, numerical ground-truth programs, baselines, compliance code,
and hidden test points. Deleting those files on a branch is insufficient because
they remain accessible through git history.

Use a separate companion repository with independent history:

```text
trusted HorizonMath checkout
  reads public prompt only
  submits task and later evaluates answer
              |
              v
verifier-free companion repo in Codex cloud
  full terminal and local tool access
  agent-phase internet access Off
  writes exactly answers/<run>/<problem>.md
```

Agent-phase internet must be **Off**. HorizonMath is public, so an internet-enabled
agent could fetch the validators even if its checked-out repository is clean. Codex
cloud setup scripts may still use the internet to install dependencies before the
agent phase.

## 1. Create the companion project

From the trusted HorizonMath checkout:

```bash
uv run scripts/create_agent_workspace.py ../HorizonMath-agent-workspace
cd ../HorizonMath-agent-workspace
git remote add origin git@github.com:YOUR_ORG/HorizonMath-agent-workspace.git
git push -u origin main
```

Create the GitHub repository separately before pushing. It can be private. Never
seed it from HorizonMath, fork HorizonMath, or merge HorizonMath history into it.

The generator creates only:

```text
.gitignore
AGENTS.md
README.md
answers/.gitkeep
```

The benchmark runner checks the entire object history and refuses any companion
repository that has ever tracked another path.

## 2. Configure Codex cloud

In [Codex environment settings](https://chatgpt.com/codex/settings/environments):

1. Create an environment for the companion GitHub repository.
2. Configure any packages or setup script needed for mathematical work. Audit setup
   and maintenance scripts and recreate any cached container so they cannot leave
   HorizonMath, validators, ground truth, or other benchmark-private files behind.
3. Set **Agent internet access** to **Off**.
4. Note the environment ID or a unique environment label.

Because the companion repository exists both locally and on GitHub, run the
benchmark with `--agent-workspace` pointing to the local companion checkout. The
official `codex cloud` CLI then launches the task in that project’s cloud
environment and branch.

## 3. Inspect exact prompts without launching tasks

```bash
uv run scripts/run_agent_benchmark.py \
  --debug \
  --env YOUR_ENVIRONMENT \
  --agent-workspace ../HorizonMath-agent-workspace \
  --agent-repo-url git@github.com:YOUR_ORG/HorizonMath-agent-workspace.git \
  --model gpt-5.6-sol \
  --reasoning-effort ultra \
  --problem w4_watson_integral
```

The generated `prompts.jsonl` contains:

- the exact evaluation-mode system-message text used by `run_benchmark.py`;
- the exact `problem["prompt"]` supplied to single-shot models;
- an agent wrapper that requires Codex goal creation and names the unique answer
  file.

No other problem fields are copied into the cloud prompt.

The cloud CLI has no separate system-message argument, so the canonical text is
embedded inside the cloud task prompt rather than installed at the same message
priority used by single-shot provider APIs. Codex cloud and `AGENTS.md` instructions
also remain in effect. Results should therefore be labeled as an agentic evaluation,
not treated as role-identical to the single-shot condition.

## 4. Run cloud agents

Run one problem first:

```bash
uv run scripts/run_agent_benchmark.py \
  --env YOUR_ENVIRONMENT \
  --agent-workspace ../HorizonMath-agent-workspace \
  --agent-repo-url git@github.com:YOUR_ORG/HorizonMath-agent-workspace.git \
  --model gpt-5.6-sol \
  --reasoning-effort ultra \
  --confirm-environment-isolated \
  --confirm-goal-tools-available \
  --confirm-agent-internet-off \
  --problem w4_watson_integral
```

Run the full benchmark with at most ten active cloud tasks:

```bash
uv run scripts/run_agent_benchmark.py \
  --env YOUR_ENVIRONMENT \
  --agent-workspace ../HorizonMath-agent-workspace \
  --agent-repo-url git@github.com:YOUR_ORG/HorizonMath-agent-workspace.git \
  --model gpt-5.6-sol \
  --reasoning-effort ultra \
  --confirm-environment-isolated \
  --confirm-goal-tools-available \
  --confirm-agent-internet-off \
  --confirm-live-canary \
  --parallel 10
```

Every task is instructed to create a Codex goal for its problem before reasoning,
keep it active while working, and complete it only after saving the answer. Each
task uses one attempt so that one problem produces one benchmark response.

`codex cloud exec` does not currently expose goal lifecycle events or environment
repository/cache metadata. The two confirmation flags are therefore fail-closed
operator attestations, recorded in `config.json`; prompt text alone is not presented
as machine-verifiable proof. Before a scored run, verify that the chosen cloud
runtime exposes goal tools and that the environment is attached to the companion
repository with a clean setup/cache. If machine-auditable goal lifecycle evidence is
mandatory, the current cloud CLI surface is insufficient.

`--model` and `--reasoning-effort` are passed to every `codex cloud exec`
submission as `model` and `model_reasoning_effort` configuration overrides. The
runner records the requested values in `config.json` and each response. The
current cloud CLI returns the task ID and URL but does not echo the resolved model
or reasoning effort in machine-readable task metadata. These records prove what
the runner requested, not independent confirmation of what the service resolved.
If independent runtime confirmation is mandatory, do not score the run.

Before a multi-problem run, complete the documented one-problem command and its
trusted evaluation in the exact environment. Inspect the saved prompt, task URL,
response, evaluation, and cloud task UI, then pass `--confirm-live-canary`.
Multi-problem runs fail closed without that attestation. This is the only part of
the workflow that cannot be exercised in repository tests because it requires the
operator’s real companion GitHub repository and Codex cloud environment.

The runner is resumable. It records the task ID immediately after submission:

```bash
uv run scripts/run_agent_benchmark.py \
  --env YOUR_ENVIRONMENT \
  --agent-workspace ../HorizonMath-agent-workspace \
  --agent-repo-url git@github.com:YOUR_ORG/HorizonMath-agent-workspace.git \
  --model gpt-5.6-sol \
  --reasoning-effort ultra \
  --confirm-environment-isolated \
  --confirm-goal-tools-available \
  --confirm-agent-internet-off \
  --confirm-live-canary \
  --resume results/codex-cloud_<label>_<timestamp>/
```

On resume it continues polling already-submitted cloud tasks rather than launching
duplicates. Use `--retry-errors` to submit new tasks for terminal failures.
Transient status and diff-retrieval failures are retried against the original task.
A timed-out task may still be running in Codex cloud; its task URL is retained in
the error ledger so an operator can inspect or cancel it before retrying.

Do not apply or merge cloud task diffs into the companion repository. The trusted
runner reads each diff directly into `responses.jsonl`; keeping the companion
branch unchanged ensures every problem starts from the same empty workspace.

## 5. Evaluate in the trusted checkout

The agent runner emits the same `responses.jsonl` schema as the existing providers:

```bash
uv run scripts/evaluate_responses.py \
  results/codex-cloud_<label>_<timestamp>/
```

Only after cloud execution is complete does this trusted phase load validators,
ground truth, baselines, or compliance checks.

Returned `proposed_solution()` code runs behind an OS-enforced boundary with a
scrubbed environment and no network. It cannot read the trusted checkout. Hidden
multi-point expected values are never placed in the answer process, and each hidden
point runs in a fresh process. Trusted evaluation requires:

- macOS: the built-in `sandbox-exec` and `/usr/bin/python3`;
- Linux: `bubblewrap` (`bwrap`).

Evaluation fails closed when the platform sandbox is unavailable.
The isolated answer runtime includes `mpmath` and NumPy, the libraries explicitly
used by dataset answer contracts. Execution also has bounded output, file size,
open files, CPU time, and process-group cleanup; Linux additionally enforces address
space and process-count limits.

For numeric coding-agent results, compliance evaluation also fails closed: set
`GOOGLE_API_KEY` (or `GEMINI_API_KEY`, which the script maps) before Phase 2.
Missing credentials, malformed compliance output, or compliance API failures do not
count as passing agent solutions.

Phase 2 refuses to score until every configured selected problem has exactly one
terminal outcome (a response or the latest generation error), and rejects duplicate
or unexpected response IDs. This prevents interrupted subset runs from being
reported as complete benchmark scores.

## Output files

```text
results/codex-cloud_<label>_<timestamp>/
├── config.json
├── prompts.jsonl
├── cloud_tasks.jsonl
├── responses.jsonl
├── generation_errors.jsonl
├── evaluation.jsonl
└── summary.json
```

`cloud_tasks.jsonl` is an append-only task ledger containing task URLs and lifecycle
events. The runner accepts a cloud result only when its diff creates exactly the
named answer file and changes no other tracked path.
`generation_errors.jsonl` is the Phase 1 failure ledger. Phase 2 keeps only the
latest error per problem, so retries cannot inflate the summary.

## Isolation checks

Before submitting a task, the runner verifies:

- the local companion checkout is a clean git repository;
- its `origin` matches `--agent-repo-url`;
- the repository is not `ewang26/HorizonMath`;
- all remote branches and tags have been fetched for inspection;
- the selected local and GitHub branch resolve to the same commit;
- the repository has exactly one reachable commit across all local and remote refs;
- the seed commit’s files and bytes exactly match the trusted minimal template;
- the dataset, canonical system messages, and selected problem IDs match the saved
  run fingerprints on resume and evaluation;
- the operator has confirmed that the cloud environment is bound to the companion
  repository and that setup, maintenance, and cached state contain no benchmark
  artifacts;
- the operator has confirmed that goal lifecycle tools are available to the cloud
  agent;
- the operator has explicitly confirmed agent-phase internet access is off.
- a real one-problem canary has completed and been evaluated in the same environment
  before any multi-problem run.

The Codex cloud CLI does not currently expose these environment and goal properties
for machine verification, so the final four items are operator attestations and
are recorded in `config.json`.

## Prompt for a fresh Codex coordinator session

Open the trusted HorizonMath checkout on this branch in a normal Codex session.
The coordinator may run locally; scored problem solving may not. Replace the three
placeholders before sending this prompt:

```text
Follow the root AGENTS.md exactly.

Run a new coding-agent evaluation on HorizonMath dataset indices 1–10 inclusive.
Use scripts/run_agent_benchmark.py with:
- Codex cloud environment: YOUR_CODEX_CLOUD_ENVIRONMENT
- verifier-free companion checkout: YOUR_AGENT_WORKSPACE_PATH
- companion GitHub repository: YOUR_AGENT_REPOSITORY_URL
- model: gpt-5.6-sol
- reasoning effort: ultra
- parallel cloud tasks: 10

Every scored solver must be a fresh codex cloud exec task in the configured
OpenAI-managed cloud environment. Do not use create_thread, projectless tasks,
worktrees, or local collaboration agents as scored solvers. Local coordination
and trusted evaluation are expected.

Use the required environment, goal-tool, internet-off, and completed-canary
confirmations only after verifying they are true. Generate prompts before
submission, wait for all ten terminal outcomes, run the trusted evaluator with
mandatory Gemini compliance checking, and report per-problem results, compliant
pass rate, generation failures, and links to every run artifact.

If the cloud environment or attestations cannot be verified, or if independent
machine confirmation of the resolved model settings is required, fail closed
before scored submission and report the exact blocker.
```

`--range` is zero-based and inclusive. The prompt above intentionally selects
dataset indices 1 through 10. Use `0–9` only when the intended indices are 0
through 9.
