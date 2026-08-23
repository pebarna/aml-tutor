"""Baked-in check for lesson 015 — the agreement-rate report."""
import pytest

CASES = [{"label_decision": "escalate"}, {"label_decision": "close"}]
RESULTS = [{"decision": "escalate"}, {"decision": "escalate"}]
DETERMINISTIC = [
    {"decision_match": True, "citation_present": True},
    {"decision_match": False, "citation_present": False},
]
JUDGE = [{"agrees": True, "comment": "ok"}, {"agrees": False, "comment": "weak"}]


def test_report_computes_the_three_rates():
    from aml_triage.eval import report

    result = report(CASES, RESULTS, DETERMINISTIC, JUDGE)
    assert result["n_cases"] == 2
    assert result["decision_agreement_rate"] == pytest.approx(0.5)
    assert result["citation_present_rate"] == pytest.approx(0.5)
    assert result["judge_agreement_rate"] == pytest.approx(0.5)


def test_report_raises_on_mismatched_list_lengths():
    from aml_triage.eval import report

    with pytest.raises(ValueError):
        report(CASES, RESULTS[:1], DETERMINISTIC, JUDGE)
