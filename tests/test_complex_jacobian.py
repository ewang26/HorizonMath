import pytest

from conjectures import VerificationError, verify_complex_jacobian
from conjectures.validators_11 import VALIDATORS


ANNOUNCED_THREE_DIMENSIONAL_COUNTEREXAMPLE = {
    "dimension": 3,
    "polynomials": [
        "(1+x1*x2)**3*x3 + x2**2*(1+x1*x2)*(4+3*x1*x2)",
        "x2 + 3*x1*(1+x1*x2)**2*x3 + 3*x1*x2**2*(4+3*x1*x2)",
        "2*x1 - 3*x1**2*x2 - x1**3*x3",
    ],
    "x": [0, 0, "-1/4"],
    "y": [1, "-3/2", "13/2"],
}


def test_announced_counterexample_passes_exact_three_dimensional_check():
    result = verify_complex_jacobian(ANNOUNCED_THREE_DIMENSIONAL_COUNTEREXAMPLE, dimension=3)

    assert result.valid
    assert result.normalized_witness == {
        "dimension": 3,
        "form": "general",
        "jacobian_determinant": "-2",
    }


def test_benchmark_registry_is_fixed_to_dimension_two():
    with pytest.raises(VerificationError, match="candidate dimension does not match"):
        VALIDATORS["complex_jacobian"](ANNOUNCED_THREE_DIMENSIONAL_COUNTEREXAMPLE)


@pytest.mark.parametrize("scale", ["2", "sqrt(2)", "I"])
def test_nonunit_nonzero_constant_determinant_is_allowed(scale):
    result = verify_complex_jacobian(
        {
            "polynomials": [f"({scale})*x1", "x2"],
            "x": [0, 0],
            "y": [1, 0],
        }
    )

    assert not result.valid
    assert result.reason == "the supplied points do not collide under the map"


@pytest.mark.parametrize(
    "polynomials",
    [
        ["x1**2", "x2"],
        ["x1", "0"],
    ],
)
def test_nonconstant_or_zero_jacobian_is_rejected_even_when_points_collide(polynomials):
    result = verify_complex_jacobian(
        {
            "polynomials": polynomials,
            "x": [-1, 0],
            "y": [1, 0],
        }
    )

    assert not result.valid
    assert result.reason == "the Jacobian determinant is not a nonzero constant"


def test_algebraically_disguised_zero_determinant_is_rejected():
    algebraic_zero = "sqrt(5 + 2*sqrt(6)) - sqrt(2) - sqrt(3)"
    result = verify_complex_jacobian(
        {
            "polynomials": [f"({algebraic_zero})*x1", "x2"],
            "x": [0, 0],
            "y": [1, 0],
        }
    )

    assert not result.valid
    assert result.reason == "the Jacobian determinant is not a nonzero constant"


def test_algebraically_equal_points_are_not_treated_as_distinct():
    result = verify_complex_jacobian(
        {
            "polynomials": ["x1", "x2"],
            "x": ["sqrt(5 + 2*sqrt(6))", 0],
            "y": ["sqrt(2) + sqrt(3)", 0],
        }
    )

    assert not result.valid
    assert result.reason == "the two witness points are identical"


@pytest.mark.parametrize(
    "candidate",
    [
        {
            "polynomials": ["2**sqrt(2)*x1", "x2"],
            "x": [0, 0],
            "y": [1, 0],
        },
        {
            "polynomials": ["x1", "x2"],
            "x": ["2**sqrt(2)", 0],
            "y": [1, 0],
        },
    ],
)
def test_transcendental_certificate_values_are_rejected(candidate):
    with pytest.raises(VerificationError, match="exact algebraic number"):
        verify_complex_jacobian(candidate)
