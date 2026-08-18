"""Regression tests for provenance-screened GPT-5.6 reporting."""

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_gpt56_pro_solution_report import (  # noqa: E402
    accepted_solution_classification,
)
from baseline_comparator import (  # noqa: E402
    compare_against_baseline,
    load_baselines,
)
from validators.autocorr_signed_upper import (  # noqa: E402
    validate as validate_signed_autocorr,
)
from validators.autocorr_upper import (  # noqa: E402
    validate as validate_autocorr,
)


REPORT_DIR = ROOT / "reports/gpt56_pro_final_solutions"


def test_pre_existing_certificates_do_not_receive_new_solution_credit():
    assert (
        accepted_solution_classification("autocorr_upper", 1)
        == "pre_existing_certificate"
    )
    assert (
        accepted_solution_classification("autocorr_signed_upper", 2)
        == "pre_existing_certificate"
    )
    assert accepted_solution_classification("airy_moment_a5", 2) == (
        "new_solution"
    )
    assert accepted_solution_classification("airy_moment_a4", 0) == (
        "tier_0_calibration"
    )


def test_generated_report_excludes_pre_existing_certificates_from_score():
    artifact = json.loads(
        (REPORT_DIR / "gpt56_pro_final_solutions.json").read_text()
    )
    score = artifact["final_score"]

    assert score["passed"] == 15
    assert score["total"] == 113
    assert score["pass_rate_percent"] == 13.3
    assert score["raw_accepted_outputs"] == 17
    assert score["new_solutions_tiers_1_to_3"] == 8
    assert score["pre_existing_certificates_excluded"] == 2

    scored_ids = {row["problem_id"] for row in artifact["solutions"]}
    certificates = {
        row["problem_id"]: row
        for row in artifact["pre_existing_certificates"]
    }
    expected = {"autocorr_upper", "autocorr_signed_upper"}

    assert scored_ids.isdisjoint(expected)
    assert certificates.keys() == expected
    assert all(not row["counted_in_score"] for row in certificates.values())
    assert all(
        "submitted_response" not in row and "original_submission" not in row
        for row in certificates.values()
    )
    assert all(
        row["original_submission_retained_in_source_archive"]
        for row in certificates.values()
    )
    assert artifact["review_scope"] == {
        "provenance_adjustment": [
            "autocorr_signed_upper",
            "autocorr_upper",
        ],
        "other_accepted_candidates_re_adjudicated": False,
    }
    assert artifact["pipeline"]["deterministic_benchmark_improvements"] == 2
    assert artifact["by_tier"]["1"]["passed"] == 3
    assert artifact["by_tier"]["2"]["passed"] == 4


def test_fixed_certificates_match_reported_hashes_and_sizes():
    expected = {
        "autocorr_upper": (
            90_000,
            "a2d2c953704be161f34a421269464ba9e48ba0fe17a4fd81ff0fd69b26d70d80",
        ),
        "autocorr_signed_upper": (
            400,
            "0e86498ba294fb7a45606e3b8aa62765830fe13ed0539c8e57a8e8b0e49c9fae",
        ),
    }

    for problem_id, (size, digest) in expected.items():
        path = REPORT_DIR / "certificates" / f"{problem_id}.json"
        payload = path.read_bytes()
        certificate = json.loads(payload)
        assert len(certificate["values"]) == size
        assert hashlib.sha256(payload).hexdigest() == digest


def test_fixed_certificates_pass_and_results_are_json_serializable():
    validators = {
        "autocorr_upper": validate_autocorr,
        "autocorr_signed_upper": validate_signed_autocorr,
    }

    baselines = load_baselines(ROOT / "data/baselines.json")

    for problem_id, validate in validators.items():
        certificate = json.loads(
            (
                REPORT_DIR
                / "certificates"
                / f"{problem_id}.json"
            ).read_text()
        )
        result = validate(certificate)
        serialized = json.loads(result.to_json())
        assert result.valid
        assert serialized["metrics"]["improves_bound"] is False
        comparison = compare_against_baseline(
            problem_id, serialized["metrics"], baselines
        )
        assert comparison.result == "matches_baseline"
        assert comparison.improvement_percent == 0.0


def test_generated_certificate_evidence_matches_updated_baselines():
    artifact = json.loads(
        (REPORT_DIR / "gpt56_pro_final_solutions.json").read_text()
    )
    certificates = {
        row["problem_id"]: row
        for row in artifact["pre_existing_certificates"]
    }
    expected = {
        "autocorr_upper": 1.5028503020710076,
        "autocorr_signed_upper": 1.4545548626983325,
    }

    for problem_id, benchmark_value in expected.items():
        verification = certificates[problem_id]["verification"]
        metrics = verification["validator_metrics"]
        comparison = verification["baseline_comparison"]
        assert metrics["best_known_upper"] == benchmark_value
        assert metrics["improves_bound"] is False
        assert comparison["result"] == "matches_baseline"
        assert comparison["baseline_value"] == benchmark_value
        assert comparison["improvement_percent"] == 0.0


def test_public_report_uses_neutral_certificate_summary():
    markdown = (REPORT_DIR / "gpt56_pro_final_solutions.md").read_text()

    assert "## Pre-existing certificates — not scored (2)" in markdown
    assert "PRE-EXISTING CERTIFICATE — `autocorr_upper`" in markdown
    assert "PRE-EXISTING CERTIFICATE — `autocorr_signed_upper`" in markdown
    assert "urlopen" not in markdown
