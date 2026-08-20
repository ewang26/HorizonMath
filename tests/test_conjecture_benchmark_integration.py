import json
from pathlib import Path

import pytest

from conjectures import ValidationResult as CoreValidationResult
from scripts.validator_registry import get_validator, get_validator_path, has_validator
from validators import ValidationResult
from validators import grantham_challenge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTED_CONJECTURE_IDS_IN_ORDER = (
    "euler_sum_of_powers_k6",
    "bpsw_pseudoprime",
    "selfridge_fibonacci",
    "grantham_challenge",
    "bfw_enhanced",
    "wall_sun_sun",
    "complex_jacobian",
    "turyn_type_tt46",
    "skew_hadamard_356",
    "cocyclic_hadamard_188",
    "finite_magma_e677_not_e255",
    "line_graph_inertia",
    "dual_finite_magma",
    "max_laplacian_44",
    "max_laplacian_46",
    "distance_spectral_independence_inertia",
    "erdos_97",
    "erdos_811",
    "symmetric_conference_matrix_86",
    "rshcd_196_type_minus",
    "steiner_3_5_41",
    "steiner_3_6_46",
    "seymour_second_neighborhood",
)
SELECTED_CONJECTURE_IDS = set(SELECTED_CONJECTURE_IDS_IN_ORDER)


def test_selected_conjectures_are_grouped_at_the_end_of_the_dataset():
    problems = json.loads((PROJECT_ROOT / "data" / "problems_full.json").read_text())
    problem_ids = [problem["id"] for problem in problems]

    assert problem_ids[-len(SELECTED_CONJECTURE_IDS_IN_ORDER) :] == list(
        SELECTED_CONJECTURE_IDS_IN_ORDER
    )


def test_every_selected_conjecture_has_a_loadable_benchmark_adapter():
    problems = json.loads((PROJECT_ROOT / "data" / "problems_full.json").read_text())
    selected = {problem["id"] for problem in problems if problem["id"] in SELECTED_CONJECTURE_IDS}
    assert selected == SELECTED_CONJECTURE_IDS

    for problem_id in sorted(selected):
        assert has_validator(problem_id), problem_id
        assert get_validator_path(problem_id) is not None, problem_id
        assert callable(get_validator(problem_id)), problem_id


@pytest.mark.parametrize(
    ("problem_id", "witness"),
    [
        (
            "complex_jacobian",
            {"polynomials": ["x1", "x2"], "x": [0, 0], "y": [1, 1]},
        ),
        (
            "seymour_second_neighborhood",
            {"n": 3, "arcs": [[0, 1], [1, 2], [2, 0]]},
        ),
    ],
)
def test_core_adapters_return_the_horizonmath_result_contract(problem_id, witness):
    validator = get_validator(problem_id)
    assert validator is not None
    result = validator(witness)
    assert isinstance(result, ValidationResult)
    assert isinstance(result.valid, bool)
    assert isinstance(result.message, str)
    assert isinstance(result.metrics, dict)


def test_grantham_benchmark_uses_the_fixed_core_verifier(monkeypatch):
    observed = {}

    def fake_core(witness):
        observed["witness"] = witness
        return CoreValidationResult(
            True,
            "Grantham challenge",
            "valid counterexample witness",
            {"n": 15},
        )

    monkeypatch.setattr(grantham_challenge, "verify_grantham_challenge", fake_core)
    witness = {"n": 15, "divisor": 3}
    result = grantham_challenge.validate(witness)

    assert result.valid is True
    assert observed == {"witness": witness}
    assert result.metrics == {"n": 15}


def test_grantham_core_uses_the_published_polynomial(monkeypatch):
    from conjectures import validators_11

    observed = {}

    def fake_frobenius_test(n, P, Q, D):
        observed.update(n=n, P=P, Q=Q, D=D)
        return True

    monkeypatch.setattr(
        validators_11, "is_quadratic_frobenius_prp", fake_frobenius_test
    )
    result = validators_11.verify_grantham_challenge({"n": 15, "divisor": 3})

    assert result.valid is True
    assert observed == {"n": 15, "P": -5, "Q": 5, "D": 5}
    assert result.normalized_witness == {"n": 15}
