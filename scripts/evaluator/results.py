"""Result types for evaluation outputs."""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .code_extraction import ExtractionResult, ExtractionStatus
from .sandbox import ExecutionResult, ExecutionStatus
from .scoring import Score, OverallGrade


@dataclass
class EvaluationResult:
    """Result of evaluating a single problem."""
    problem_id: str
    problem_index: int
    extraction: Optional[ExtractionResult] = None
    execution: Optional[ExecutionResult] = None
    score: Optional[Score] = None
    success: bool = False
    error_stage: Optional[str] = None  # 'extraction', 'execution', 'scoring'
    error_message: Optional[str] = None
    expected_value: Optional[str] = None
    actual_value: Optional[str] = None
    # Additional fields for compatibility with scripts/evaluate.py EvaluationResult
    problem_title: str = ""
    matching_digits: Optional[int] = None  # Convenience field (also in score.matching_digits)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "problem_id": self.problem_id,
            "problem_index": self.problem_index,
            "problem_title": self.problem_title,
            "success": self.success,
            "error_stage": self.error_stage,
            "error_message": self.error_message,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "matching_digits": self.matching_digits,
        }

        if self.extraction:
            result["extraction"] = {
                "status": self.extraction.status.value,
                "error_message": self.extraction.error_message,
            }

        if self.execution:
            result["execution"] = {
                "status": self.execution.status.value,
                "error_message": self.execution.error_message,
                "execution_time_ms": self.execution.execution_time_ms,
            }

        if self.score is not None:
            result["score"] = {
                "grade": self.score.grade.value,
                "matching_digits": self.score.matching_digits,
                "required_digits": self.score.required_digits,
            }

        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class BatchResults:
    """Results from batch evaluation of multiple problems."""
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model: str = ""
    provider: str = ""
    results: list[EvaluationResult] = field(default_factory=list)

    def add_result(self, result: EvaluationResult) -> None:
        """Add an evaluation result."""
        self.results.append(result)

    def summary(self) -> dict:
        """Generate summary statistics."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = total - passed

        # Count by error stage
        by_error_stage: dict[str, int] = {}
        for r in self.results:
            if r.error_stage:
                by_error_stage[r.error_stage] = by_error_stage.get(r.error_stage, 0) + 1

        # Calculate pass rate
        pass_rate = f"{(passed / total * 100):.1f}%" if total > 0 else "N/A"

        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "model": self.model,
            "provider": self.provider,
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": pass_rate,
            "by_error_stage": by_error_stage,
        }

    def save(self, output_path: Path) -> tuple[Path, Path]:
        """
        Save results to JSONL and summary JSON files.

        Args:
            output_path: Base path for output files (without extension)
                        Will create {path}.jsonl and {path}.summary.json

        Returns:
            Tuple of (jsonl_path, summary_path)
        """
        # Ensure output_path has no extension for base path
        base_path = output_path.with_suffix('')

        # Save per-problem results as JSONL
        jsonl_path = Path(str(base_path) + ".eval.jsonl")
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        with open(jsonl_path, 'w') as f:
            for result in self.results:
                f.write(result.to_json() + '\n')

        # Save summary as JSON
        summary_path = Path(str(base_path) + ".eval.summary.json")
        with open(summary_path, 'w') as f:
            json.dump(self.summary(), f, indent=2)

        return jsonl_path, summary_path

    @classmethod
    def from_jsonl(cls, jsonl_path: Path) -> "BatchResults":
        """Load batch results from a JSONL file."""
        results = cls()
        with open(jsonl_path) as f:
            for line in f:
                data = json.loads(line)
                # Reconstruct EvaluationResult (simplified - without nested objects)
                result = EvaluationResult(
                    problem_id=data["problem_id"],
                    problem_index=data["problem_index"],
                    success=data["success"],
                    error_stage=data.get("error_stage"),
                    error_message=data.get("error_message"),
                    expected_value=data.get("expected_value"),
                    actual_value=data.get("actual_value"),
                    problem_title=data.get("problem_title", ""),
                    matching_digits=data.get("matching_digits"),
                )
                results.add_result(result)
        return results
