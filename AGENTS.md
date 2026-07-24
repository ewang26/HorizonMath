# HorizonMath Codex agent instructions

## Default coding-agent evaluation workflow

When a user asks Codex from a conversation to evaluate Codex/coding agents on
HorizonMath, use **repository-free tasks running in OpenAI-hosted Codex cloud
VMs/containers** by default.

“Cloud” refers to the execution runtime, not only model inference. Every scored
agent's Python/mpmath work, symbolic calculations, subprocesses, tool calls, and
subagent coordination must run in the OpenAI-hosted container. A task created
with `hostId: local`, a path on the user's machine, or any other local runtime is
not a cloud task and is invalid.

Do not silently substitute:

- `create_thread` with `target.type: "projectless"` when it returns
  `hostId: "local"`;
- local collaboration subagents;
- a ChatGPT Work cloud task that does not explicitly confirm a Codex cloud
  container plus the requested Codex model and reasoning effort;
- direct model API calls; or
- the GitHub-backed `scripts/run_agent_benchmark.py` workflow.

### Required preflight

Before sending any scored prompt, inspect the available task-creation capability.
It must explicitly confirm all of the following in its creation result:

- execution location: `openai-hosted`;
- runtime kind: `codex-cloud-container`;
- a non-local host ID and cloud environment ID;
- no repository or worktree attachment; and
- the exact requested model and reasoning effort.

Never infer cloud execution from words such as `projectless`, `conversation`,
`background`, or `cloud inference`. Never infer the model settings merely because
the creation call accepted them. If the returned metadata does not confirm every
item above, stop before prompt submission and explain that this Codex surface
cannot run the benchmark condition. This is the required fail-closed behavior.

1. Run `scripts/prepare_conversation_benchmark.py` with the user's exact problem
   selection, model, and reasoning effort.
   Preparing the manifest is not completion: continue autonomously through thread
   creation, waiting, response collection, and evaluation in the same task. Do not
   ask the user to launch or copy prompts manually.
2. Read each complete `agent_task_prompt` from the generated `prompts.jsonl`.
   Send that string verbatim. Never reconstruct, shorten, paraphrase, or summarize
   the canonical `system_message` or `prompt`.
3. Create one fresh primary task per problem in an explicitly confirmed
   OpenAI-hosted Codex cloud container, with no repository or worktree attached.
   Never run a scored agent in the trusted HorizonMath checkout or anywhere else
   on the local machine.
4. Copy the task-creation confirmation and terminal outcome into
   `cloud_runtime.jsonl`, following
   `docs/codex_conversation_agents.md`. This file must contain exactly one record
   per selected problem. A successful record must bind the prompt hash, task ID,
   cloud environment, non-local host, model, reasoning effort, and final-response
   hash.
5. The primary cloud agent may spawn as many subagents as it needs inside its cloud
   environment. Subagents must receive only the exact benchmark text already
   present in the primary task. Do not use local collaboration agents.
6. Record only the primary cloud agent's final response in `responses.jsonl`,
   with one terminal outcome per selected problem. Wait for every task; do not
   declare the run complete from submission acknowledgements.
7. Run `scripts/validate_conversation_run.py` before evaluation. It rejects missing
   evidence, `hostId: local`, local execution, model/reasoning mismatches, prompt
   mismatches, response mismatches, and incomplete terminal coverage.
8. Evaluate with `scripts/evaluate_responses.py`, which repeats the same cloud
   provenance validation. Agentic numeric runs require the Gemini compliance
   check and fail closed when its credential is absent or the check errors. Never
   report numeric agreement alone as a benchmark pass.

`scripts/run_agent_benchmark.py` is a separate, opt-in GitHub-repository workflow.
Use it only when the user explicitly asks for that implementation.
