#!/usr/bin/env python3
"""
Baseline comparator for OpenMath benchmark.

Compares validator metrics against baselines from baselines.json.
Handles direction-aware comparison (minimize vs maximize).
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class BaselineComparison:
    """Result of comparing achieved metric against baseline."""
    result: str  # 'beats_baseline', 'matches_baseline', 'below_baseline', 'no_baseline'
    achieved_value: Optional[float]
    baseline_value: Optional[float]
    direction: Optional[str]  # 'minimize' or 'maximize'
    metric_name: Optional[str]
    improvement_percent: Optional[float]
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'result': self.result,
            'achieved_value': self.achieved_value,
            'baseline_value': self.baseline_value,
            'direction': self.direction,
            'metric_name': self.metric_name,
            'improvement_percent': self.improvement_percent,
            'notes': self.notes
        }


def load_baselines(baselines_path: Path) -> dict[str, dict]:
    """
    Load baselines from JSON file.

    Returns:
        Dict mapping problem_id to baseline data
    """
    if not baselines_path.exists():
        return {}

    with open(baselines_path) as f:
        baselines_list = json.load(f)

    # Convert list to dict keyed by problem_id
    return {b['problem_id']: b for b in baselines_list}


def parse_baseline_value(value_str: str) -> Optional[float]:
    """
    Parse a baseline value string to float.

    Handles formats like:
    - "7.103205334137..."
    - "2.6390"
    - "1.28"
    - "> 6.5" (extracts 6.5)
    - "1/64" (simple fractions)
    - "-3/4" (negative fractions)
    """
    if not value_str:
        return None

    # Remove ellipsis and trailing dots
    value_str = re.sub(r'\.\.\.?$', '', str(value_str))

    # Handle inequality signs
    value_str = re.sub(r'^[<>≤≥]\s*', '', value_str)

    stripped = value_str.strip()

    # Try simple fraction a/b
    frac_match = re.match(r'^(-?\d+)\s*/\s*(\d+)$', stripped)
    if frac_match:
        numer = int(frac_match.group(1))
        denom = int(frac_match.group(2))
        if denom != 0:
            return numer / denom
        return None

    # Try plain numeric value
    match = re.match(r'^-?\d+\.?\d*', stripped)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None

    return None


def get_metric_value(metrics: dict, baseline_data: dict) -> Optional[float]:
    """
    Extract the relevant metric value from validator metrics.

    Resolution order:
    1. Explicit metric_key from baseline config (most reliable)
    2. Direct match of baseline metric name to validator key
    3. Common metric name mappings
    4. Fall back to 'ratio' key
    5. Fall back to first numeric value that's not a problem-size parameter
    """
    if not metrics:
        return None

    baseline_info = baseline_data.get('baseline', {})

    # 1. Explicit metric_key (highest priority - unambiguous mapping)
    metric_key = baseline_info.get('metric_key')
    if metric_key and metric_key in metrics and isinstance(metrics[metric_key], (int, float)):
        return float(metrics[metric_key])

    baseline_metric = baseline_info.get('metric', '').lower()

    # 2. Direct match of baseline metric name to validator key
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if key.lower() == baseline_metric.replace(' ', '_'):
                return float(value)

    # 3. Common metric mappings - order matters, more specific first
    metric_mappings = {
        'upper bound': ['ratio', 'constant', 'c', 'limit_constant', 'upper_bound'],
        'lower bound': ['ratio', 'constant', 'c', 'limit_constant', 'lower_bound', 'value'],
        'limit constant': ['ratio', 'constant', 'c', 'limit_constant'],
        'ratio': ['ratio', 'constant', 'c'],
        'irrationality measure': ['irrationality_measure', 'mu', 'measure', 'ratio'],
        'merit factor': ['merit_factor', 'merit', 'mf'],
        'size': ['size', 'basis_size', 'count'],
        'bound': ['bound', 'upper_bound', 'lower_bound', 'value', 'ratio'],
    }

    for base_key, aliases in metric_mappings.items():
        if base_key in baseline_metric:
            for alias in aliases:
                if alias in metrics and isinstance(metrics[alias], (int, float)):
                    return float(metrics[alias])

    # 4. Fall back to 'ratio' if available (common metric for optimization problems)
    if 'ratio' in metrics and isinstance(metrics['ratio'], (int, float)):
        return float(metrics['ratio'])

    # 5. Fall back to first numeric value that's not a problem-size parameter
    # Exclude keys that represent problem parameters rather than optimization metrics
    PARAMETER_KEYS = {
        'n', 'missing_count', 'basis_size', 'size', 'count',
        'num_points', 'dimension', 'prime', 'length', 'field_size',
        'num_directions', 'num_segments', 'regularity', 'torsion_order',
        'num_vertices', 'design_degree', 'max_lag', 'order',
    }
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and key not in PARAMETER_KEYS:
            return float(value)

    return None


def compare_against_baseline(
    problem_id: str,
    metrics: dict,
    baselines: dict[str, dict]
) -> BaselineComparison:
    """
    Compare achieved metric against baseline value.

    Args:
        problem_id: The problem identifier
        metrics: Dict of metrics from validator
        baselines: Dict of baseline data keyed by problem_id

    Returns:
        BaselineComparison with result and improvement percentage
    """
    if problem_id not in baselines:
        return BaselineComparison(
            result='no_baseline',
            achieved_value=None,
            baseline_value=None,
            direction=None,
            metric_name=None,
            improvement_percent=None,
            notes=f"No baseline found for problem {problem_id}"
        )

    baseline_data = baselines[problem_id]
    baseline_info = baseline_data.get('baseline', {})

    direction = baseline_info.get('direction')
    metric_name = baseline_info.get('metric')
    baseline_value_str = baseline_info.get('value')

    baseline_value = parse_baseline_value(baseline_value_str)
    if baseline_value is None:
        return BaselineComparison(
            result='no_baseline',
            achieved_value=None,
            baseline_value=None,
            direction=direction,
            metric_name=metric_name,
            improvement_percent=None,
            notes=f"Could not parse baseline value: {baseline_value_str}"
        )

    achieved_value = get_metric_value(metrics, baseline_data)
    if achieved_value is None:
        return BaselineComparison(
            result='no_baseline',
            achieved_value=None,
            baseline_value=baseline_value,
            direction=direction,
            metric_name=metric_name,
            improvement_percent=None,
            notes=f"Could not extract relevant metric from: {metrics}"
        )

    # Compare based on direction
    def _values_match(a, b, rtol=1e-12):
        if a == b:
            return True
        denom = max(abs(a), abs(b))
        return denom > 0 and abs(a - b) / denom < rtol

    if direction == 'minimize':
        if baseline_value == 0:
            if _values_match(achieved_value, 0):
                return BaselineComparison(
                    result='matches_baseline',
                    achieved_value=achieved_value,
                    baseline_value=baseline_value,
                    direction=direction,
                    metric_name=metric_name,
                    improvement_percent=0.0
                )
            else:
                return BaselineComparison(
                    result='below_baseline',
                    achieved_value=achieved_value,
                    baseline_value=baseline_value,
                    direction=direction,
                    metric_name=metric_name,
                    improvement_percent=None
                )
        if achieved_value < baseline_value and not _values_match(achieved_value, baseline_value):
            improvement = ((baseline_value - achieved_value) / baseline_value) * 100
            return BaselineComparison(
                result='beats_baseline',
                achieved_value=achieved_value,
                baseline_value=baseline_value,
                direction=direction,
                metric_name=metric_name,
                improvement_percent=round(improvement, 2)
            )
        elif _values_match(achieved_value, baseline_value):
            return BaselineComparison(
                result='matches_baseline',
                achieved_value=achieved_value,
                baseline_value=baseline_value,
                direction=direction,
                metric_name=metric_name,
                improvement_percent=0.0
            )
        else:
            deficit = ((achieved_value - baseline_value) / baseline_value) * 100
            return BaselineComparison(
                result='below_baseline',
                achieved_value=achieved_value,
                baseline_value=baseline_value,
                direction=direction,
                metric_name=metric_name,
                improvement_percent=round(-deficit, 2)
            )

    elif direction == 'maximize':
        if baseline_value == 0:
            if _values_match(achieved_value, 0):
                return BaselineComparison(
                    result='matches_baseline',
                    achieved_value=achieved_value,
                    baseline_value=baseline_value,
                    direction=direction,
                    metric_name=metric_name,
                    improvement_percent=0.0
                )
            else:
                return BaselineComparison(
                    result='beats_baseline',
                    achieved_value=achieved_value,
                    baseline_value=baseline_value,
                    direction=direction,
                    metric_name=metric_name,
                    improvement_percent=None
                )
        if achieved_value > baseline_value and not _values_match(achieved_value, baseline_value):
            improvement = ((achieved_value - baseline_value) / baseline_value) * 100
            return BaselineComparison(
                result='beats_baseline',
                achieved_value=achieved_value,
                baseline_value=baseline_value,
                direction=direction,
                metric_name=metric_name,
                improvement_percent=round(improvement, 2)
            )
        elif _values_match(achieved_value, baseline_value):
            return BaselineComparison(
                result='matches_baseline',
                achieved_value=achieved_value,
                baseline_value=baseline_value,
                direction=direction,
                metric_name=metric_name,
                improvement_percent=0.0
            )
        else:
            deficit = ((baseline_value - achieved_value) / baseline_value) * 100
            return BaselineComparison(
                result='below_baseline',
                achieved_value=achieved_value,
                baseline_value=baseline_value,
                direction=direction,
                metric_name=metric_name,
                improvement_percent=round(-deficit, 2)
            )

    else:
        # Unknown direction, just report values
        return BaselineComparison(
            result='below_baseline',
            achieved_value=achieved_value,
            baseline_value=baseline_value,
            direction=direction,
            metric_name=metric_name,
            improvement_percent=None,
            notes=f"Unknown optimization direction: {direction}"
        )


def get_default_baselines_path() -> Path:
    """Get the default path to baselines.json."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    return project_root / "data" / "baselines.json"


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Compare metrics against baselines")
    parser.add_argument("problem_id", help="Problem ID to compare")
    parser.add_argument("--metrics", type=str, required=True,
                        help="JSON string of metrics to compare")
    parser.add_argument("--baselines-file", type=str,
                        help="Path to baselines.json")
    args = parser.parse_args()

    # Load baselines
    baselines_path = Path(args.baselines_file) if args.baselines_file else get_default_baselines_path()
    baselines = load_baselines(baselines_path)

    # Parse metrics
    try:
        metrics = json.loads(args.metrics)
    except json.JSONDecodeError as e:
        print(f"Error parsing metrics JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Compare
    result = compare_against_baseline(args.problem_id, metrics, baselines)
    print(json.dumps(result.to_dict(), indent=2))
