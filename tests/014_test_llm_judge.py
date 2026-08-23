"""Baked-in check for lesson 014 — LLM-as-judge scoring.

Same injected-client convention as lesson 011: the baked-in test never calls a real model.
"""
from types import SimpleNamespace

RESULT = {
    "decision": "escalate",
    "rationale": "Matches structuring.",
    "cited_typology_ids": ["TY-001"],
    "retrieved": [{"id": "TY-001", "title": "Structuring / smurfing", "text": "...", "score": 0.9}],
}


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


def test_judge_forces_a_structured_tool_and_returns_its_verdict():
    from aml_triage.eval import llm_judge_score

    fake_client = _FakeClient({"agrees": True, "comment": "Rationale follows from the cited typology."})
    score = llm_judge_score(RESULT, client=fake_client)

    assert score == {"agrees": True, "comment": "Rationale follows from the cited typology."}
    call_kwargs = fake_client.messages.calls[0]
    assert call_kwargs["tool_choice"]["name"] == "submit_judge_verdict"


def test_judge_prompt_includes_the_rationale_and_cited_typology_text():
    from aml_triage.eval import llm_judge_score

    fake_client = _FakeClient({"agrees": False, "comment": "Weak link."})
    llm_judge_score(RESULT, client=fake_client)

    sent_prompt = fake_client.messages.calls[0]["messages"][0]["content"]
    assert "Matches structuring." in sent_prompt
    assert "Structuring / smurfing" in sent_prompt
