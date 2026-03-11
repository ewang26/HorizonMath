"""Unit tests for the compliance checker."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from evaluator.compliance import check_solution_compliance, ComplianceResult


def _mock_genai_response(text: str):
    """Create a mock genai response with the given text."""
    mock_response = MagicMock()
    mock_response.text = text
    return mock_response


@patch("evaluator.compliance.genai.Client")
def test_compliant_solution(mock_client_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_genai_response(
        '{"compliant": true, "reason": "uses known constants and gamma function"}'
    )
    mock_client_cls.return_value = mock_client

    result = check_solution_compliance("def proposed_solution():\n    return mp.gamma(mp.mpf('1')/4)")
    assert result.compliant is True
    assert "known constants" in result.reason


@patch("evaluator.compliance.genai.Client")
def test_non_compliant_solution(mock_client_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_genai_response(
        '{"compliant": false, "reason": "uses mp.quad for numerical integration"}'
    )
    mock_client_cls.return_value = mock_client

    result = check_solution_compliance("def proposed_solution():\n    return mp.quad(lambda x: x**2, [0, 1])")
    assert result.compliant is False
    assert "mp.quad" in result.reason


@patch("evaluator.compliance.genai.Client")
def test_unparseable_response_defaults_compliant(mock_client_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_genai_response(
        "I think this solution looks fine to me."
    )
    mock_client_cls.return_value = mock_client

    result = check_solution_compliance("def proposed_solution():\n    return mp.pi")
    assert result.compliant is True
    assert "parse error" in result.reason.lower()


@patch("evaluator.compliance.genai.Client")
def test_api_error_defaults_compliant(mock_client_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API connection failed")
    mock_client_cls.return_value = mock_client

    result = check_solution_compliance("def proposed_solution():\n    return mp.pi")
    assert result.compliant is True
    assert "api error" in result.reason.lower()


@patch("evaluator.compliance.genai.Client")
def test_markdown_fenced_json_response(mock_client_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_genai_response(
        '```json\n{"compliant": true, "reason": "valid closed-form expression"}\n```'
    )
    mock_client_cls.return_value = mock_client

    result = check_solution_compliance("def proposed_solution():\n    return mp.euler")
    assert result.compliant is True
    assert "closed-form" in result.reason
