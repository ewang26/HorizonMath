<h1 align="center">HorizonMath</h1>

<h3 align="center">Measuring AI Progress Toward Mathematical Discovery with Automatic Verification</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2603.15617"><img src="https://img.shields.io/badge/arXiv-2603.15617-b31b1b.svg" alt="arXiv"></a>
  <a href="https://ewang26.github.io/HorizonMath/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/squashenthus/HorizonMath"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Dataset-yellow.svg" alt="Dataset"></a>
</p>

<p align="center">
  Erik Y. Wang*, Sumeet Motwani, James V. Roggeveen, Eliot Hodges, Dulhan Jayalath,<br>
  Charles London, Kalyan Ramakrishnan, Flaviu Cipcigan, Philip Torr, Alessandro Abate
</p>

<p align="center">
  University of Oxford · Benchmark · Harvard University · Princeton University · Ellison Institute of Technology
</p>

<p align="center">
  *Correspondence: <a href="mailto:erik.wang@dtc.ox.ac.uk">erik.wang@dtc.ox.ac.uk</a>
</p>

## Setup

```bash
uv sync
```

Some validators require [SageMath](https://www.sagemath.org/). Install it separately (`brew install sage` on macOS, `sudo apt-get install -y sagemath` on Debian/Ubuntu). If Sage is not on your PATH, set `SAGE_CMD=/path/to/sage`.

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
```

## Problem Taxonomy

The benchmark contains **101 problems** across 8 domains, classified by three fields in `data/problems_full.json`:

- **`solvability`** (difficulty level): 0 (calibration, 10), 1 (likely solvable, 23), 2 (challenging, 60), 3 (possibly unsolvable, 8)
- **`output_type`** (artifact type): constant (54), function (5), formula_discovery (3), construction (39)
- **`evaluation_mode`** (how answers are checked): ground_truth_computable (59), benchmark_best_known (33), new_construction (9)

Solvability 0 problems have known solutions and serve as a verification step for the evaluation pipeline and a calibration for models.

## Running the Benchmark

The benchmark has a two-phase pipeline:

1. **Phase 1 — Generate responses** (`run_benchmark.py`): Prompts models and saves raw responses to `responses.jsonl`.
2. **Phase 2 — Evaluate responses** (`evaluate_responses.py`): Evaluates saved responses against ground truths, validators, and baselines.

This separation means you can re-evaluate responses without re-prompting models, and long-running generation survives interruptions via `--resume`.

### Running with tmux (recommended)

The `tmux_run.sh` wrapper runs both phases in a detached tmux session, so benchmark runs survive SSH disconnects. Output is logged to `results/tmux_run_<timestamp>.log`.

```bash
# Full benchmark with defaults (OpenRouter gpt-5.2, 5 parallel)
./scripts/tmux_run.sh

# Specify provider and model
./scripts/tmux_run.sh --provider openai --model gpt-5.2-pro

# Single problem
./scripts/tmux_run.sh --problem diff_basis_upper

# Resume an interrupted run
./scripts/tmux_run.sh --resume results/openrouter_openai-gpt-5.2_20260205_143022/

# Run only generation or evaluation
./scripts/tmux_run.sh --phase generate --provider openai --model gpt-5.2
./scripts/tmux_run.sh --phase evaluate --resume results/openrouter_openai-gpt-5.2_20260205_143022/
```

```bash
tmux attach -t openmath       # Attach to see live output
# Ctrl-b d                    # Detach without stopping
tmux kill-session -t openmath # Abort the run
```

### Running each phase manually

**Phase 1 — Generate responses:**

```bash
uv run scripts/run_benchmark.py                                    # Full benchmark (OpenRouter gpt-5.2)
uv run scripts/run_benchmark.py --provider openai --model gpt-5.2-pro  # Use OpenAI directly
uv run scripts/run_benchmark.py --problem w4_watson_integral       # Single problem
uv run scripts/run_benchmark.py --parallel 10                      # Parallel generation
uv run scripts/run_benchmark.py --resume results/<run_dir>/        # Resume interrupted run
```

**Phase 2 — Evaluate responses:**

```bash
uv run scripts/evaluate_responses.py results/<run_dir>/            # Evaluate all responses
uv run scripts/evaluate_responses.py results/<run_dir>/ --force    # Re-evaluate from scratch
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

Both scripts support additional options — run with `--help` for full details.

### Aggregating split runs

You can split a benchmark across parallel jobs using `--range` (0-based inclusive indices):

```bash
uv run scripts/run_benchmark.py --range 0-49 --provider openai --model gpt-5.2-pro
uv run scripts/run_benchmark.py --range 50-100 --provider openai --model gpt-5.2-pro
```

Then merge the result directories into a single report:

```bash
uv run scripts/aggregate_results.py results/openai_gpt-5.2-pro_*/ -o results/gpt-5.2-pro_combined/
```

## Evaluating Individual Solutions

The evaluation script (`scripts/evaluate.py`) can evaluate a single LLM solution file. The mode is auto-detected from the problem's `evaluation_mode`:

- **Numeric** (`ground_truth_computable`): compares returned digits against the ground truth.
- **Benchmark** (`benchmark_best_known`): validates the construction and compares metrics against baselines.
- **Construction** (`new_construction`): validates the construction (pass/fail, no baseline comparison).

```bash
uv run python scripts/evaluate.py --llm-output solution.txt --problem-index 34
uv run python scripts/evaluate.py --llm-output solution.txt --problem-id diff_basis_upper --json
uv run python scripts/evaluate.py --list-problems --mode benchmark
```

### LLM Output Format

LLMs should output a `proposed_solution()` function:

```python
def proposed_solution():
    # For numeric problems: return a number
    # For benchmark/construction problems: return a JSON-serializable dict/list
    return {"n": 10, "basis": [0, 1, 2, 6, 9]}
```

Each validator documents its expected input format in its docstring.

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

## Citation

```bibtex
@article{wang2026horizonmath,
  title     = {HorizonMath: Measuring AI Progress Toward Mathematical
               Discovery with Automatic Verification},
  author    = {Wang, Erik Y. and Motwani, Sumeet and Roggeveen, James V.
               and Hodges, Eliot and Jayalath, Dulhan and London, Charles
               and Ramakrishnan, Kalyan and Cipcigan, Flaviu
               and Torr, Philip and Abate, Alessandro},
  year      = {2026},
  note      = {Working Draft}
}
```
