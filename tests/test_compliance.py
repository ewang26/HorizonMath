"""Unit tests for the compliance checker."""

import os
import re
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from evaluator.compliance import (
    _COMPLIANCE_PROMPT,
    COMPLIANCE_MODEL,
    COMPLIANCE_THINKING_LEVEL,
    DEFAULT_COMPLIANCE_ROUNDS,
    OPENROUTER_COMPLIANCE_MODEL,
    check_solution_compliance,
    ComplianceResult,
)


def _mock_openrouter_response(text: str):
    """Create a mock OpenRouter chat-completions response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = text
    return mock_response


def _mock_genai_response(text: str):
    """Create a mock genai response with the given text."""
    mock_response = MagicMock()
    mock_response.text = text
    return mock_response


@patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
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
    call = mock_client.models.generate_content.call_args
    assert call.kwargs["model"] == COMPLIANCE_MODEL == "gemini-3.6-flash"
    assert (
        call.kwargs["config"].thinking_config.thinking_level
        == COMPLIANCE_THINKING_LEVEL
    )
    assert call.kwargs["config"].response_mime_type == "application/json"


@patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
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


@patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
@patch("evaluator.compliance.genai.Client")
def test_unparseable_response_is_indeterminate(mock_client_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_genai_response(
        "I think this solution looks fine to me."
    )
    mock_client_cls.return_value = mock_client

    result = check_solution_compliance("def proposed_solution():\n    return mp.pi")
    assert result.compliant is None
    assert result.status == "indeterminate"
    assert "parse/schema error" in result.reason.lower()


@patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
@patch("evaluator.compliance.genai.Client")
def test_invalid_response_schema_is_indeterminate(mock_client_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = _mock_genai_response(
        '{"reason": "missing verdict"}'
    )
    mock_client_cls.return_value = mock_client

    result = check_solution_compliance("def proposed_solution():\n    return mp.pi")
    assert result.compliant is None
    assert result.status == "indeterminate"
    assert "parse/schema error" in result.reason.lower()


@patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
@patch("evaluator.compliance.genai.Client")
def test_api_error_is_indeterminate(mock_client_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API connection failed")
    mock_client_cls.return_value = mock_client

    result = check_solution_compliance("def proposed_solution():\n    return mp.pi")
    assert result.compliant is None
    assert result.status == "indeterminate"
    assert "api error" in result.reason.lower()


@patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
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


@patch.dict(os.environ, {}, clear=True)
def test_missing_api_key_is_indeterminate():
    result = check_solution_compliance("def proposed_solution():\n    return mp.pi")
    assert result.compliant is None
    assert result.status == "indeterminate"
    assert "google_api_key is not set" in result.reason.lower()


@patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
@patch("evaluator.compliance.genai.Client")
def test_no_strict_majority_is_indeterminate(mock_client_cls):
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [
        _mock_genai_response(
            '{"compliant": true, "reason": "valid closed form"}'
        ),
        Exception("temporary API failure"),
        Exception("temporary API failure"),
    ]
    mock_client_cls.return_value = mock_client

    result = check_solution_compliance(
        "def proposed_solution():\n    return mp.pi", n=3
    )
    assert result.compliant is None
    assert result.status == "indeterminate"
    assert "no strict majority" in result.reason.lower()
    assert "1/3 compliant" in result.reason
    assert "2/3 indeterminate" in result.reason


@patch.dict(
    os.environ,
    {
        "COMPLIANCE_PROVIDER": "openrouter",
        "COMPLIANCE_MODEL": "stealth/ox-alpha",
        "COMPLIANCE_REASONING_EFFORT": "max",
        "OPENROUTER_API_KEY": "test-openrouter-key",
    },
    clear=True,
)
@patch("evaluator.compliance.openai.OpenAI")
def test_openrouter_reviewer_uses_chat_completions(mock_openai_cls):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openrouter_response(
        '{"compliant": true, "reason": "independent closed-form expression"}'
    )
    mock_openai_cls.return_value = mock_client

    result = check_solution_compliance("def proposed_solution():\n    return mp.pi")
    assert result.compliant is True
    assert result.provider == "openrouter"
    assert result.model == "stealth/ox-alpha"
    mock_openai_cls.assert_called_with(
        base_url="https://openrouter.ai/api/v1",
        api_key="test-openrouter-key",
        timeout=10 * 60,
    )
    call = mock_client.chat.completions.create.call_args
    assert call.kwargs["model"] == OPENROUTER_COMPLIANCE_MODEL == "stealth/ox-alpha"
    assert call.kwargs["extra_body"]["reasoning"]["effort"] == "max"
    assert call.kwargs["extra_body"]["reasoning"]["enabled"] is True


@patch.dict(
    os.environ,
    {"COMPLIANCE_PROVIDER": "openrouter"},
    clear=True,
)
def test_missing_openrouter_api_key_is_indeterminate():
    result = check_solution_compliance("def proposed_solution():\n    return mp.pi")
    assert result.compliant is None
    assert result.status == "indeterminate"
    assert "openrouter_api_key is not set" in result.reason.lower()
    assert result.model == OPENROUTER_COMPLIANCE_MODEL


def test_default_compliance_review_uses_five_rounds():
    assert DEFAULT_COMPLIANCE_ROUNDS == 5


def test_rubric_integrates_clarifications_into_original_structure():
    rule_markers = [
        "**Numerical integration**",
        "**Finite truncations of infinite series**",
        "**Numerical root-finding**",
        "**Restating the defining expression as a computational procedure**",
        "**Unevaluated infinite series/products/limits**",
        "**Hardcoded or encoded target values**",
        "**Circular / tautological identities**",
        "**Numerical parameter fitting / digit-matching constructions**",
    ]
    integrated_markers = [
        "**Task fulfillment**",
        "**Judge the mathematical representation, not library internals**",
        "exactly determined by an input integer",
        "integral, limit, differential equation, or other non-series object",
        "power-of-ten denominator",
        "defining local expansion, parameter derivative, normalization",
        "unexplained large coefficients",
        "Named functions of exactly specified finite matrices",
    ]

    for marker in rule_markers + integrated_markers:
        assert marker in _COMPLIANCE_PROMPT

    assert re.findall(r"(?m)^\d+\. \*\*", _COMPLIANCE_PROMPT) == [
        f"{number}. **" for number in range(1, 9)
    ]
    assert "ADDITIONAL ADJUDICATION RULES" not in _COMPLIANCE_PROMPT
