"""Baked-in check for lesson 013 — deterministic triage checks.

Citation *validity* (does the id exist and was it shown to the model) is already enforced by
lesson 010's parse_triage_decision guard, upstream of this function. deterministic_score only
checks citation *presence* — did the model cite anything at all.
"""

CASE_ESCALATE = {
    "transaction": {"step": 1, "type": "TRANSFER", "amount": 181.0},
    "classifier_score": 0.95,
    "label_decision": "escalate",
    "label_note": "Matches structuring: many small amounts, rapid succession.",
}


def test_decision_match_true_when_decisions_agree():
    from aml_triage.eval import deterministic_score

    result = {"decision": "escalate", "rationale": "x", "cited_typology_ids": ["TY-001"]}
    score = deterministic_score(CASE_ESCALATE, result)
    assert score == {"decision_match": True, "citation_present": True}


def test_decision_match_false_when_decisions_disagree():
    from aml_triage.eval import deterministic_score

    result = {"decision": "close", "rationale": "x", "cited_typology_ids": ["TY-001"]}
    score = deterministic_score(CASE_ESCALATE, result)
    assert score["decision_match"] is False


def test_citation_present_false_when_nothing_was_cited():
    from aml_triage.eval import deterministic_score

    result = {"decision": "escalate", "rationale": "x", "cited_typology_ids": []}
    score = deterministic_score(CASE_ESCALATE, result)
    assert score["citation_present"] is False
