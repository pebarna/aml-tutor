"""Baked-in check for lesson 011 — the end-to-end triage agent.

Uses a fake client injected through triage()'s client= parameter. No network call, no
ANTHROPIC_API_KEY needed to pass this test.
"""
from types import SimpleNamespace

import pytest


class _FakeMessages:
    def __init__(self, tool_input):
        self._tool_input = tool_input
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = SimpleNamespace(type="tool_use", input=self._tool_input)
        return SimpleNamespace(content=[block])


class _FakeClient:
    def __init__(self, tool_input):
        self.messages = _FakeMessages(tool_input)


def test_triage_forces_the_structured_tool_and_returns_the_parsed_decision(typologies_path):
    from aml_triage.retrieval import top_k_typologies_hybrid
    from aml_triage.triage import triage
    from aml_triage.triage_schema import TRIAGE_TOOL_SCHEMA

    transaction = {"step": 5, "type": "TRANSFER", "amount": 181.0}
    # Use the same query that triage() will build internally
    query = f"{transaction['type']} transaction of amount {transaction['amount']}"
    retrieved = top_k_typologies_hybrid(
        query, k=2, corpus_path=str(typologies_path)
    )
    valid_id = retrieved[0]["id"]

    fake_client = _FakeClient(
        {"decision": "escalate", "rationale": "Matches a known pattern.", "cited_typology_ids": [valid_id]}
    )
    result = triage(
        transaction, classifier_score=0.9, client=fake_client, k=2, corpus_path=str(typologies_path)
    )

    assert result["decision"] == "escalate"
    assert result["cited_typology_ids"] == [valid_id]
    assert result["retrieved"] == retrieved

    call_kwargs = fake_client.messages.calls[0]
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": TRIAGE_TOOL_SCHEMA["name"]}
    assert call_kwargs["tools"] == [TRIAGE_TOOL_SCHEMA]


def test_triage_raises_when_the_model_cites_something_it_was_not_shown(typologies_path):
    from aml_triage.triage import triage

    transaction = {"step": 5, "type": "TRANSFER", "amount": 181.0}
    fake_client = _FakeClient(
        {"decision": "escalate", "rationale": "x", "cited_typology_ids": ["TY-999"]}
    )
    with pytest.raises(ValueError):
        triage(
            transaction, classifier_score=0.9, client=fake_client, k=1, corpus_path=str(typologies_path)
        )
