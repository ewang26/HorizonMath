#!/usr/bin/env python3
"""Create a minimal verifier-free repository for HorizonMath cloud agents."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


AGENTS_MD = """# HorizonMath cloud-agent workspace

This repository is an intentionally minimal workspace for mathematical discovery
benchmark tasks.

- Use all locally installed tools needed to reason, calculate, and test conjectures.
- At the beginning of each task, create the goal requested by the task prompt.
- Write the final answer only to the exact path under `answers/` named in the prompt.
- Use `/tmp` for scratch work. Do not commit scratch artifacts.
- Do not seek HorizonMath validators, evaluator implementations, hidden test points,
  numeric ground truth, or other private evaluation artifacts.
- Agent-phase internet access must remain disabled for this environment.
"""

README_MD = """# HorizonMath coding-agent workspace

This repository contains no benchmark evaluators or hidden answers. It is the
isolated working repository used by Codex cloud tasks. The trusted HorizonMath
repository submits one problem at a time and retrieves only the resulting file
under `answers/`.
"""

GITIGNORE = """# Agent scratch files must live outside the repository.
*
!.gitignore
!AGENTS.md
!README.md
!answers/
!answers/**
"""


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def create_workspace(destination: Path, *, initialize_git: bool) -> None:
    destination = destination.expanduser().resolve()
    trusted_root = Path(__file__).resolve().parent.parent
    if destination == trusted_root or is_within(destination, trusted_root):
        raise ValueError(
            "The agent workspace must be outside the trusted HorizonMath checkout "
            "so it cannot inherit verifier-containing git history."
        )
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Destination is not empty: {destination}")

    (destination / "answers").mkdir(parents=True, exist_ok=True)
    (destination / "AGENTS.md").write_text(AGENTS_MD, encoding="utf-8")
    (destination / "README.md").write_text(README_MD, encoding="utf-8")
    (destination / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    (destination / "answers" / ".gitkeep").write_text("", encoding="utf-8")

    if initialize_git:
        commands = [
            ["git", "init", "-b", "main"],
            ["git", "add", ".gitignore", "AGENTS.md", "README.md", "answers/.gitkeep"],
            [
                "git",
                "-c",
                "user.name=HorizonMath Benchmark",
                "-c",
                "user.email=horizonmath@example.invalid",
                "commit",
                "-m",
                "Initialize verifier-free coding-agent workspace",
            ],
        ]
        for command in commands:
            subprocess.run(command, cwd=destination, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a minimal repository for Codex cloud tasks. The destination "
            "must be outside this HorizonMath checkout."
        )
    )
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="Create files without initializing independent git history",
    )
    args = parser.parse_args()

    try:
        create_workspace(args.destination, initialize_git=not args.no_git)
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Created verifier-free agent workspace: {args.destination.resolve()}")
    if not args.no_git:
        print("Next: create a separate GitHub repository, add it as origin, and push main.")


if __name__ == "__main__":
    main()
