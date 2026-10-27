"""Point-set validators must reject non-finite coordinates and keep scoring valid configurations as before."""

import importlib
import itertools

import numpy as np
import pytest

MODULES = [
    "tammes_n15",
    "kissing_number_dim5",
    "kissing_number_dim6",
    "kissing_number_dim9",
    "kissing_number_dim11",
    "heilbronn_n12",
    "spherical_7_design_minimal",
    "spherical_9_design_s2",
]


def _cross_polytope(dim):
    eye = np.eye(dim)
    return np.vstack([eye, -eye])


def _sphere_points(n, dim, seed):
    pts = np.random.default_rng(seed).normal(size=(n, dim))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _two_24_cells():
    pts = []
    for i, j in itertools.combinations(range(4), 2):
        for si in (1, -1):
            for sj in (1, -1):
                p = np.zeros(4)
                p[i], p[j] = si, sj
                pts.append(p / np.sqrt(2))
    for i in range(4):
        for s in (1, -1):
            p = np.zeros(4)
            p[i] = s
            pts.append(p)
    for signs in itertools.product((0.5, -0.5), repeat=4):
        pts.append(np.array(signs))
    return np.array(pts)


# A configuration each validator accepts (None: no accepted example is built here).
VALID = {
    "tammes_n15": _sphere_points(15, 3, 0),
    "kissing_number_dim5": _cross_polytope(5),
    "kissing_number_dim6": _cross_polytope(6),
    "kissing_number_dim9": _cross_polytope(9),
    "kissing_number_dim11": _cross_polytope(11),
    "heilbronn_n12": np.random.default_rng(1).random((12, 2)),
    "spherical_7_design_minimal": _two_24_cells(),
    "spherical_9_design_s2": None,
}
# A finite configuration of the right shape (accepted or not by the validator, but not for being non-finite).
FINITE = {
    "tammes_n15": _sphere_points(15, 3, 0),
    "kissing_number_dim5": _cross_polytope(5),
    "kissing_number_dim6": _cross_polytope(6),
    "kissing_number_dim9": _cross_polytope(9),
    "kissing_number_dim11": _cross_polytope(11),
    "heilbronn_n12": np.random.default_rng(1).random((12, 2)),
    "spherical_7_design_minimal": _two_24_cells(),
    "spherical_9_design_s2": _sphere_points(20, 3, 2),
}


def _validate(name, points):
    module = importlib.import_module(f"validators.{name}")
    return module.validate({"points": np.asarray(points).tolist()})


@pytest.mark.parametrize("name", MODULES)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_coordinate_is_rejected(name, bad):
    points = FINITE[name].copy()
    points[0, 0] = bad
    result = _validate(name, points)
    assert not result.valid
    assert "finite" in result.message.lower()


@pytest.mark.parametrize("name", MODULES)
def test_finite_input_is_not_rejected_as_non_finite(name):
    result = _validate(name, FINITE[name])
    assert "finite" not in result.message.lower()


@pytest.mark.parametrize("name", [m for m in MODULES if VALID[m] is not None])
def test_valid_configuration_still_passes(name):
    result = _validate(name, VALID[name])
    assert result.valid, result.message
