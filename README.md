# HorizonMath

Research project for evaluating LLMs on open mathematical problems.

## Setup

```bash
uv sync
```

Some benchmark validators require SageMath (not a Python package). Install it separately:

```bash
# Ubuntu/Debian
sudo apt-get install -y sagemath

# macOS (Homebrew)
brew install sage
```

If Sage is installed outside your PATH, set `SAGE_CMD` to the executable, e.g.
`SAGE_CMD=/opt/homebrew/bin/sage`.

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
```

## Project Structure

```
HorizonMath/
├── data/                   # JSON data files
│   ├── problems_full.json  # Problem definitions, prompts, and numeric values
│   └── baselines.json      # State-of-the-art baseline values for benchmark problems
├── numerics/               # Numerical solution scripts (one per problem)
├── validators/             # Validators for benchmark and construction problems
│   ├── utils.py            # Shared utilities (ValidationResult, etc.)
│   └── *.py                # Problem-specific validators (one per problem)
├── scripts/                # Utility scripts
│   ├── run_benchmark.py    # Generate LLM responses (Phase 1)
│   ├── evaluate_responses.py   # Evaluate saved responses (Phase 2)
│   ├── tmux_run.sh         # Run both phases in a tmux session
│   ├── evaluate.py         # Core evaluation logic (numeric + benchmark modes)
│   ├── validator_registry.py   # Auto-discovers validators
│   ├── baseline_comparator.py  # Compares results against baselines
│   ├── aggregate_results.py    # Aggregate evaluations across split runs
│   └── run_numerics.py     # Execute all numeric scripts
└── pyproject.toml          # Python dependencies
```

## Problem Taxonomy

Problems are classified with three fields in `data/problems_full.json`:

- `output_type`: the kind of artifact being produced (constant, formula, construction, etc.)
- `evaluation_mode`: how references are generated (ground-truth or benchmark)
- `solvability`: difficulty level (0 = already solved / calibration, 1 = likely solvable, 2 = challenging, 3 = possibly unsolvable)

### Solvability Levels

| Level | Description | Count |
| --- | --- | --- |
| 0 | Already solved; included as calibration and sanity checks | 10 |
| 1 | Likely solvable with known techniques or near-term AI | 23 |
| 2 | Challenging; may require novel insights or methods | 60 |
| 3 | Possibly unsolvable; included for experimental exploration | 8 |

Level 0 problems have known solutions and serve to verify that the evaluation pipeline and LLMs can correctly reproduce established results.

### Output Type Counts

| Output Type | Count |
| --- | --- |
| constant | 54 |
| function | 5 |
| formula_discovery | 3 |
| construction | 39 |

**Total: 101 problems**

### Evaluation Mode Counts

| Evaluation Mode | Count |
| --- | --- |
| ground_truth_computable | 59 |
| benchmark_best_known | 33 |
| new_construction | 9 |

### Domain Counts

| Domain | Count |
| --- | --- |
| number_theory | 17 |
| discrete_geometry | 15 |
| lattice_models | 15 |
| continuum_physics | 13 |
| combinatorics | 12 |
| integrals | 11 |
| coding_theory | 9 |
| mathematical_constants | 9 |

### Run numerical computations

Most `ground_truth_computable` problems have a corresponding Python script in `numerics/` that computes their numerical value to high precision. For problems without a numerics script, the ground-truth value was obtained from the reference papers listed for the corresponding problem in the dataset json file. To run all numerical scripts and update `problems_full.json`:

```bash
# Run all numeric scripts
uv run python scripts/run_numerics.py

# Run a specific problem
uv run python scripts/run_numerics.py --problem 0

# Dry run (show what would be done)
uv run python scripts/run_numerics.py --dry-run

# Force recompute even if value exists
uv run python scripts/run_numerics.py --force
```

This updates `data/problems_full.json` with `numeric_value` fields for each problem.

## Running the Benchmark

The benchmark has a two-phase pipeline:

1. **Phase 1 — Generate responses** (`run_benchmark.py`): Prompts models and saves raw responses to `responses.jsonl`.
2. **Phase 2 — Evaluate responses** (`evaluate_responses.py`): Evaluates saved responses against ground truths, validators, and baselines.

This separation means you can re-evaluate responses without re-prompting models, and long-running generation survives interruptions via `--resume`.

### Running with tmux (recommended)

The `tmux_run.sh` wrapper runs both phases in a detached tmux session, so benchmark runs survive SSH disconnects and laptop lid closures. Output is logged to `results/tmux_run_<timestamp>.log`.

```bash
# Full benchmark with defaults (OpenRouter gpt-5.2, 5 parallel)
./scripts/tmux_run.sh

# Specify provider and model
./scripts/tmux_run.sh --provider openai --model gpt-5.2-pro

# Single problem
./scripts/tmux_run.sh --problem diff_basis_upper

# Resume an interrupted run
./scripts/tmux_run.sh --resume results/openrouter_openai-gpt-5.2_20260205_143022/

# Override parallelism
./scripts/tmux_run.sh --provider anthropic --parallel 4

# Run only generation (skip evaluation)
./scripts/tmux_run.sh --phase generate --provider openai --model gpt-5.2

# Run only evaluation on existing results
./scripts/tmux_run.sh --phase evaluate --resume results/openrouter_openai-gpt-5.2_20260205_143022/

# Explicitly run both phases (default)
./scripts/tmux_run.sh --phase both
```

Monitor and manage the session:

```bash
tmux attach -t openmath       # Attach to see live output
# Ctrl-b d                    # Detach without stopping the run
tmux kill-session -t openmath # Abort the run
```

When both phases finish, the session drops into an interactive shell so you can inspect results.

### Running each phase manually

You can also run the phases separately:

**Phase 1 — Generate responses:**

```bash
# Full benchmark with OpenRouter gpt-5.2 (default)
uv run scripts/run_benchmark.py

# Single problem
uv run scripts/run_benchmark.py --problem w4_watson_integral

# Use OpenAI directly (with code execution tool)
uv run scripts/run_benchmark.py --provider openai --model gpt-5.2-pro

# Use Anthropic Claude
uv run scripts/run_benchmark.py --provider anthropic

# Use Gemini
uv run scripts/run_benchmark.py --provider gemini

# Parallel generation
uv run scripts/run_benchmark.py --parallel 10

# Resume an interrupted run
uv run scripts/run_benchmark.py --resume results/openrouter_openai-gpt-5.2_20260205_143022/
```

**Phase 2 — Evaluate responses:**

```bash
# Evaluate all responses in a results directory
uv run scripts/evaluate_responses.py results/openrouter_openai-gpt-5.2_20260205_143022/

# Re-evaluate (overwrites previous evaluation results)
uv run scripts/evaluate_responses.py results/openrouter_openai-gpt-5.2_20260205_143022/ --force
```

### Output Structure

Results are saved to timestamped folders in `results/`:

```
results/openai_gpt-5.2_20260205_143022/
├── config.json          # Run configuration
├── prompts.jsonl        # Problem prompts (saved before API calls)
├── responses.jsonl      # Raw LLM responses (Phase 1 output)
├── evaluation.jsonl     # Per-problem evaluation results (Phase 2 output)
└── summary.json         # Detailed statistics
```

### CLI Options for `run_benchmark.py`

| Option | Description |
|--------|-------------|
| `--provider` | LLM provider: `openrouter` (default), `openai`, `gemini`, or `anthropic` |
| `--model` | Model name (defaults: `openai/gpt-5.2` for OpenRouter, `gpt-5.2` for OpenAI, `gemini-3-pro-preview` for Gemini, `claude-opus-4-6` for Anthropic) |
| `--problem` | Run only a single problem by ID (supports partial match) |
| `--range` | Slice of problems to run by 0-based index, e.g. `0-24` (inclusive) |
| `--resume` | Resume from an interrupted run directory |
| `--parallel` | Number of problems to process in parallel (default: 1; `tmux_run.sh` uses 5) |
| `--data-file` | Path to problems JSON (default: `data/problems_full.json`) |
| `--debug` | Save prompts locally without calling the model API |

### CLI Options for `evaluate_responses.py`

| Option | Description |
|--------|-------------|
| `--force` | Re-evaluate even if `evaluation.jsonl` already exists |
| `--data-file` | Path to problems JSON (default: `data/problems_full.json`) |
| `--baselines-file` | Path to baselines JSON (default: `data/baselines.json`) |

### Aggregating results across split runs

When a benchmark run is split across multiple result directories (e.g. using `--range` to parallelize generation), use `aggregate_results.py` to merge evaluations into a single report:

```bash
# Aggregate all directories matching a glob
uv run scripts/aggregate_results.py results/openai_gpt-5.2-pro_*/

# Save combined evaluation.jsonl, summary.json, and provenance.json
uv run scripts/aggregate_results.py results/openai_gpt-5.2-pro_*/ -o results/gpt-5.2-pro_combined/
```

If a problem appears in multiple directories (e.g. from retried runs), the latest directory wins (by lexicographic/timestamp order). The output directory contains:

```
results/gpt-5.2-pro_combined/
├── evaluation.jsonl   # All deduplicated per-problem evaluation results
├── summary.json       # Aggregate statistics (pass rates, digit counts, etc.)
└── provenance.json    # Maps each problem_id to its source directory
```

## Evaluating LLM Solutions

The evaluation script (`scripts/evaluate.py`) supports three modes:

### Numeric Mode (ground_truth_computable problems)

For problems with known numeric answers, the LLM outputs a `proposed_solution()` function that returns a number. The evaluator compares digits against the ground truth.

```bash
# Evaluate a numeric problem
uv run python scripts/evaluate.py --llm-output solution.txt --problem-index 0
```

### Benchmark Mode (benchmark_best_known problems)

For optimization/construction problems, the LLM outputs a `proposed_solution()` function that returns a JSON-serializable construction (dict/list). The evaluator:
1. Executes the function to get the construction
2. Validates it using problem-specific validators
3. Compares metrics against state-of-the-art baselines

```bash
# Evaluate a benchmark problem (auto-detects mode)
uv run python scripts/evaluate.py --llm-output solution.txt --problem-index 34

# Explicit benchmark mode
uv run python scripts/evaluate.py --llm-output solution.txt --problem-id diff_basis_upper --mode benchmark

# List all benchmark problems
uv run python scripts/evaluate.py --list-problems --mode benchmark

# Get JSON output
uv run python scripts/evaluate.py --llm-output solution.txt --problem-index 34 --json
```

### Construction Mode (new_construction problems)

For binary existence problems (find a valid object or don't), the LLM outputs a `proposed_solution()` function returning a construction. The evaluator validates it using problem-specific validators but performs no baseline comparison — the result is simply valid or invalid.

```bash
# Evaluate a construction problem (auto-detects mode)
uv run python scripts/evaluate.py --llm-output solution.txt --problem-index 78

# List all construction problems
uv run python scripts/evaluate.py --list-problems --mode construction
```

Example output:
```
✓ VALID [34] diff_basis_upper
      Problem ID: diff_basis_upper
      Message: Verified difference basis for n=10: |B|=5, |B|²/n = 2.500000
      Metrics: n=10, basis_size=5, ratio=2.5
      Baseline: beats_baseline
        Achieved: 2.5
        Baseline: 2.639 (minimize)
        Improvement: +5.27%
```

### LLM Output Format

LLMs should output a `proposed_solution()` function:

```python
def proposed_solution():
    # For numeric problems: return a number
    # For benchmark problems: return a JSON-serializable dict/list
    return {"n": 10, "basis": [0, 1, 2, 6, 9]}
```

Each validator documents its expected input format in its docstring. Common formats include:
- Sequences: `{"sequence": [1, -1, 1, ...]}`
- Graphs: `{"edges": [[0,1], [1,2], ...]}` or `{"adjacency": [[...], ...]}`
- Points: `{"points": [[x, y, z], ...]}`
- Matrices: `{"matrix": [[...], ...]}`

## Contributing Problems

We welcome new problem contributions! To propose a problem, open a GitHub issue with the following:

1. **Problem description** — a clear mathematical statement, including the source (paper, Math Stack Exchange, etc.)
2. **Classification** — the proposed `output_type`, `domain`, `evaluation_mode`, and `solvability` level (see [Problem Taxonomy](#problem-taxonomy))
3. **Full implementation** — depending on the evaluation mode, include:

| Evaluation Mode | What to provide |
|---|---|
| `ground_truth_computable` | A numerics script that computes the answer to at least 50 digits of precision, or a reliable academic source providing a pre-computed numerical value |
| `benchmark_best_known` | A validator script **and** a baseline value with source citation |
| `new_construction` | A validator script |

You must also provide a justification of the numerics or validator script that you provide below, explaining the method(s) used.

### Numerics scripts

A numerics script computes the ground-truth value to high precision. It should be a standalone Python file:

```python
from mpmath import mp
mp.dps = 110  # at least 100 digits of precision

def compute():
    # Your computation here
    return result

if __name__ == "__main__":
    print(str(compute()))
```

### Validators

A validator checks whether a proposed construction is mathematically valid and returns metrics. It should export a single `validate(solution)` function:

```python
from . import ValidationResult, success, failure

def validate(solution):
    """
    Expected input format:
        {"basis": [b0, b1, b2, ...]}
    """
    # 1. Parse the input
    if isinstance(solution, dict) and 'basis' in solution:
        basis = solution['basis']
    else:
        return failure("Expected dict with 'basis' key")

    # 2. Check mathematical validity
    if not all_differences_covered(basis):
        return failure("Not all differences covered", basis_size=len(basis))

    # 3. Return success with metrics
    return success(
        f"Valid basis of size {len(basis)}",
        basis_size=len(basis),
        ratio=len(basis)**2 / n
    )
```

- `success(message, **metrics)` and `failure(message, **metrics)` are the only return types needed.
- Document the expected input format in the docstring.
- For benchmark problems, return **metrics** as keyword arguments — one of these is compared against the baseline.
- Helper utilities available from the `validators` package: `parse_integer`, `parse_rational`, `load_solution`, `run_sage_script`.

### Baselines (benchmark problems only)

For `benchmark_best_known` problems, provide a baseline entry for `data/baselines.json`:

```json
{
  "problem_id": "diff_basis_upper",
  "baseline": {
    "value": "2.6390",
    "direction": "minimize",
    "metric": "ratio |B|^2/n for a difference basis of [1, n-1]",
    "metric_key": "ratio"
  }
}
```

- `direction`: `"minimize"` or `"maximize"` — whether lower or higher values are better.
- `metric_key`: which key from the validator's returned metrics to compare against the baseline.
- Include a source citation for the baseline value.
