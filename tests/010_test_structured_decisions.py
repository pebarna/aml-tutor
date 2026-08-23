"""Baked-in check for lesson 010 — structured triage decisions.

No LLM call here: build_prompt and parse_triage_decision are pure functions over plain data. The
real API call is lesson 011's job.
"""
import pytest

TRANSACTION = {"step": 5, "type": "TRANSFER", "amount": 181.0}
RETRIEVED = [
    {"id": "TY-001", "title": "Structuring / smurfing", "text": "...", "score": 0.9},
    {"id": "TY-003", "title": "Rapid pass-through", "text": "...", "score": 0.4},
]


def test_prompt_includes_transaction_score_and_retrieved_titles():
    from aml_triage.triage_schema import build_prompt

    prompt = build_prompt(TRANSACTION, classifier_score=0.87, retrieved=RETRIEVED)
    assert "TRANSFER" in prompt
    assert "181.0" in prompt
    assert "0.87" in prompt
    assert "Structuring / smurfing" in prompt
    assert "Rapid pass-through" in prompt


def test_tool_schema_names_a_forced_tool_with_the_expected_fields():
    from aml_triage.triage_schema import TRIAGE_TOOL_SCHEMA

    properties = TRIAGE_TOOL_SCHEMA["input_schema"]["properties"]
    assert set(properties.keys()) == {"decision", "rationale", "cited_typology_ids"}
    assert properties["decision"]["enum"] == ["escalate", "monitor", "close"]


def test_parse_returns_a_clean_dict_for_a_valid_tool_call():
    from aml_triage.triage_schema import parse_triage_decision

    tool_input = {"decision": "escalate", "rationale": "Matches structuring.", "cited_typology_ids": ["TY-001"]}
    result = parse_triage_decision(tool_input, known_typology_ids={"TY-001", "TY-003"})
    assert result == tool_input


def test_parse_rejects_an_invalid_decision_enum():
    from aml_triage.triage_schema import parse_triage_decision

    tool_input = {"decision": "ignore", "rationale": "x", "cited_typology_ids": []}
    with pytest.raises(ValueError):
        parse_triage_decision(tool_input, known_typology_ids=set())


def test_parse_rejects_a_citation_the_model_was_not_shown():
    from aml_triage.triage_schema import parse_triage_decision

    tool_input = {"decision": "escalate", "rationale": "x", "cited_typology_ids": ["TY-005"]}
    with pytest.raises(ValueError):
        parse_triage_decision(tool_input, known_typology_ids={"TY-001", "TY-003"})
