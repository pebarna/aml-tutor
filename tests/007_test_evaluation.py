"""Baked-in check for lesson 007 — evaluation and threshold.

Uses fixed y_true/scores arrays, not a freshly-trained model — see the plan's Task 7 rationale:
exact metric values are only meaningful against a known, static input.
"""
import pytest

Y_TRUE = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
SCORES = [0.05, 0.10, 0.12, 0.20, 0.30, 0.55, 0.60, 0.65, 0.80, 0.95]
OBJECTIVE = {"min_precision": 0.90}


def test_report_returns_the_expected_keys():
    from aml_triage.evaluate import report

    result = report(Y_TRUE, SCORES, OBJECTIVE)
    assert set(result.keys()) == {"precision", "recall", "pr_auc", "threshold"}


def test_threshold_satisfies_the_min_precision_objective():
    from aml_triage.evaluate import report

    result = report(Y_TRUE, SCORES, OBJECTIVE)
    assert result["precision"] >= OBJECTIVE["min_precision"]


def test_recall_is_maximized_among_thresholds_meeting_the_objective():
    from sklearn.metrics import precision_recall_curve

    from aml_triage.evaluate import report

    result = report(Y_TRUE, SCORES, OBJECTIVE)
    precision, recall, _ = precision_recall_curve(Y_TRUE, SCORES)
    best_possible_recall = max(
        (r for p, r in zip(precision, recall) if p >= OBJECTIVE["min_precision"]),
        default=None,
    )
    assert best_possible_recall is not None, "test fixture data doesn't reach min_precision at all"
    assert result["recall"] == pytest.approx(best_possible_recall)


def test_raises_when_no_threshold_reaches_the_objective():
    from aml_triage.evaluate import report

    # min_precision above 1.0 is unreachable for any input, since precision is a ratio in
    # [0, 1] — deliberately data-independent, unlike a value merely close to 1.0 (0.9999 turns
    # out to be reachable on THIS fixture: SCORES has a clean gap between its highest negative
    # (0.60) and lowest positive (0.65), so the threshold at 0.65 yields perfect precision AND
    # perfect recall, and 0.9999 <= 1.0 does not raise).
    impossible_objective = {"min_precision": 1.5}
    with pytest.raises(ValueError):
        report(Y_TRUE, SCORES, impossible_objective)
