#!/usr/bin/env python3
"""
Validator registry for OpenMath benchmark.

Maps problem IDs to validator modules via auto-discovery of files in validators/
directory matching the pattern {problem_id}.py.

Each validator module must have a `validate(solution)` function that returns
a ValidationResult dataclass.
"""

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional

# Add project root to path for imports
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from validators import ValidationResult


# Type alias for validator functions
ValidatorFunc = Callable[[Any], ValidationResult]


def get_validators_dir() -> Path:
    """Get the validators directory path."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    return project_root / "validators"


def problem_id_from_filename(filename: str) -> Optional[str]:
    """
    Extract problem ID from a validator filename.

    Expected format: {problem_id}.py
    Example: diff_basis_upper.py -> diff_basis_upper
    """
    if filename.startswith('_') or filename == 'utils.py':
        return None
    match = re.match(r'^(.+)\.py$', filename)
    if match:
        return match.group(1)
    return None


def discover_validators() -> dict[str, Path]:
    """
    Discover all validator modules in the validators/ directory.

    Returns:
        Dict mapping problem_id to validator file path
    """
    validators_dir = get_validators_dir()
    validators = {}

    if not validators_dir.exists():
        return validators

    for file_path in validators_dir.glob("*.py"):
        if file_path.name.startswith('_') or file_path.name == 'utils.py':
            continue
        problem_id = problem_id_from_filename(file_path.name)
        if problem_id:
            validators[problem_id] = file_path

    return validators


def load_validator_module(file_path: Path) -> Optional[Any]:
    """
    Dynamically load a validator module from a file path.

    Returns:
        The loaded module or None if loading fails
    """
    try:
        # Ensure the validators package is importable
        validators_dir = file_path.parent
        project_root = validators_dir.parent

        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Import as part of the validators package
        module_name = file_path.stem  # e.g., "diff_basis_upper"
        full_module_name = f"validators.{module_name}"

        # Check if already imported
        if full_module_name in sys.modules:
            return sys.modules[full_module_name]

        # Import the module
        import importlib
        module = importlib.import_module(full_module_name)
        return module

    except Exception as e:
        print(f"Warning: Failed to load validator {file_path}: {e}", file=sys.stderr)
        return None


def get_validator(problem_id: str) -> Optional[ValidatorFunc]:
    """
    Get the validator function for a problem.

    Searches validators/ directory for {problem_id}.py

    Args:
        problem_id: The problem identifier (e.g., "033_irrationality_measure_pi")

    Returns:
        Validator function or None if no validator exists
    """
    validators = discover_validators()
    if problem_id in validators:
        module = load_validator_module(validators[problem_id])
        if module and hasattr(module, 'validate'):
            return getattr(module, 'validate')
    return None


def list_validated_problems() -> list[str]:
    """
    List all problem IDs that have validators.

    Returns:
        Sorted list of problem IDs with available validators
    """
    validators = discover_validators()
    return sorted(validators.keys())


def get_validator_path(problem_id: str) -> Optional[Path]:
    """
    Get the file path of a validator for a problem ID.

    Returns:
        Path to validator file or None if no validator exists
    """
    validators = discover_validators()
    return validators.get(problem_id)


def has_validator(problem_id: str) -> bool:
    """Check if a validator exists for the given problem ID."""
    validators = discover_validators()
    return problem_id in validators


if __name__ == "__main__":
    # CLI for listing validators
    import argparse

    parser = argparse.ArgumentParser(description="List available validators")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show validator file paths")
    parser.add_argument("--check", type=str, metavar="PROBLEM_ID",
                        help="Check if validator exists for a specific problem")
    args = parser.parse_args()

    if args.check:
        if has_validator(args.check):
            path = get_validator_path(args.check)
            print(f"Validator found: {path}")
            sys.exit(0)
        else:
            print(f"No validator found for: {args.check}")
            sys.exit(1)

    problems = list_validated_problems()
    print(f"Found {len(problems)} validators:\n")

    for pid in problems:
        if args.verbose:
            path = get_validator_path(pid)
            print(f"  {pid}")
            print(f"    -> {path}")
        else:
            print(f"  {pid}")
