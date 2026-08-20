import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate as evaluate_module  # noqa: E402
from evaluator.sandbox import (  # noqa: E402
    ExecutionStatus,
    execute_sandboxed,
    load_json_result,
)


def execute_json_result(code: str):
    execution = execute_sandboxed(code, return_json=True)
    assert execution.status is ExecutionStatus.SUCCESS, execution.error_message
    assert execution.output is not None
    return load_json_result(execution.output)


def test_json_transport_preserves_native_containers_and_exact_values():
    result = execute_json_result(
        """
def proposed_solution():
    import numpy as np
    import sympy as sp
    from fractions import Fraction

    return {
        "array": np.array([[1, 2], [3, 4]]),
        "exact": sp.sqrt(2),
        "fraction": Fraction(2, 3),
        "complex": 2 - 3j,
        "bytes": b"Dhc",
        "generator": (value for value in range(3)),
        "tuple": (4, 5),
        "sympy_tuple": sp.Tuple(6, 7),
        "pair_mapping": {(1, 2): sp.Integer(3)},
    }
"""
    )

    assert result == {
        "array": [[1, 2], [3, 4]],
        "exact": "sqrt(2)",
        "fraction": "2/3",
        "complex": "2-3*I",
        "bytes": "Dhc",
        "generator": [0, 1, 2],
        "tuple": [4, 5],
        "sympy_tuple": ["6", "7"],
        "pair_mapping": {(1, 2): "3"},
    }


def test_json_transport_preserves_a_mapping_that_uses_the_reserved_tag_key():
    result = execute_json_result(
        """
def proposed_solution():
    return {
        "__horizonmath_transport_type__": "tuple",
        "items": [1, 2],
    }
"""
    )

    assert result == {
        "__horizonmath_transport_type__": "tuple",
        "items": [1, 2],
    }


def test_json_transport_does_not_overwrite_solution_globals():
    result = execute_json_result(
        """
sp = 7
nx = 8
Mapping = 9

def proposed_solution():
    return (value for value in (sp, nx, Mapping))
"""
    )

    assert result == [7, 8, 9]


def test_json_transport_rejects_mapping_keys_that_collapse_during_encoding():
    execution = execute_sandboxed(
        "def proposed_solution():\n    return {b'x': 1, 'x': 2}\n",
        return_json=True,
    )

    assert execution.status is ExecutionStatus.SUCCESS
    with pytest.raises(ValueError, match="duplicate mapping key"):
        load_json_result(execution.output or "")


def test_json_transport_converts_networkx_graphs_without_losing_structure():
    result = execute_json_result(
        """
def proposed_solution():
    import networkx as nx

    undirected = nx.Graph()
    undirected.add_nodes_from([3, 1, 2])
    undirected.add_edge(3, 1)

    directed = nx.DiGraph()
    directed.add_nodes_from([2, 0, 1])
    directed.add_edge(2, 0)

    parallel = nx.MultiGraph()
    parallel.add_nodes_from([0, 1])
    parallel.add_edge(0, 1)
    parallel.add_edge(0, 1)

    labeled = nx.Graph()
    labeled.add_nodes_from([(2, "b"), (1, "a")])
    labeled.add_edge((2, "b"), (1, "a"))
    return {
        "undirected": undirected,
        "directed": directed,
        "parallel": parallel,
        "labeled": labeled,
    }
"""
    )

    assert result["undirected"] == {"vertices": [1, 2, 3], "edges": [(3, 1)]}
    assert result["directed"] == {"vertices": [0, 1, 2], "arcs": [(2, 0)]}
    assert result["parallel"] == {
        "vertices": [0, 1],
        "edges": [(0, 1), (0, 1)],
    }
    assert result["labeled"] == {
        "vertices": [(1, "a"), (2, "b")],
        "edges": [((2, "b"), (1, "a"))],
    }


@pytest.mark.parametrize(
    ("return_expression", "expected_error"),
    [
        ("object()", "unsupported construction result type: object"),
        ("1.5 + 2j", "only complex values with integral components are supported"),
        ("b'\\xff'", "UnicodeDecodeError"),
    ],
)
def test_json_transport_rejects_unsupported_or_inexact_values(
    return_expression, expected_error
):
    execution = execute_sandboxed(
        f"def proposed_solution():\n    return {return_expression}\n",
        return_json=True,
    )

    assert execution.status is ExecutionStatus.RUNTIME_ERROR
    assert expected_error in (execution.error_message or "")


def test_complex_jacobian_python_native_witness_reaches_validator(monkeypatch):
    problem = next(
        problem
        for problem in json.loads((PROJECT_ROOT / "data" / "problems_full.json").read_text())
        if problem["id"] == "complex_jacobian"
    )
    response = """
def proposed_solution():
    import numpy as np
    import sympy as sp

    return {
        "polynomials": (
            {(1, 0): sp.Integer(1)},
            {(0, 1): sp.Integer(1)},
        ),
        "x": np.array([0, 0]),
        "y": sp.Tuple(1, 1),
    }
"""

    monkeypatch.setattr(
        evaluate_module,
        "run_validator_with_timeout",
        lambda validator, solution: validator(solution),
    )
    result = evaluate_module.evaluate_construction_problem(problem, 119, response)

    assert result.error_type is None
    assert result.valid is False
    assert result.validator_message == "the supplied points do not collide under the map"


def test_networkx_digraph_witness_reaches_validator(monkeypatch):
    problem = next(
        problem
        for problem in json.loads((PROJECT_ROOT / "data" / "problems_full.json").read_text())
        if problem["id"] == "seymour_second_neighborhood"
    )
    response = """
def proposed_solution():
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
    return graph
"""

    monkeypatch.setattr(
        evaluate_module,
        "run_validator_with_timeout",
        lambda validator, solution: validator(solution),
    )
    result = evaluate_module.evaluate_construction_problem(problem, 135, response)

    assert result.error_type is None
    assert result.valid is False
    assert result.validator_message == "vertex 0 has |N++|=1 >= |N+|=1"
