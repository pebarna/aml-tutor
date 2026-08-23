"""Baked-in check for lesson 012 — the hand-labeled triage eval set.

This test checks the STRUCTURE of the student's own aml-triage/data/triage_eval_set.jsonl: schema,
row count, allowed decision values, and that more than one decision type was actually used. Whether
the labels themselves are good judgment calls is checked by the tutor in conversation (see the
lesson's Checks section) — the same split every other lesson draws between code (pytest) and
understanding (conversation), applied here to labeling quality instead.
"""


def test_eval_set_has_one_row_per_candidate(eval_candidates_path):
    """Confirms the labeled file covers at least the whole candidate pool. Uses >=, not ==: the
    spec explicitly encourages labeling more cases than the 16-candidate floor, so a larger file
    is expected to pass too — only a *smaller* file (a candidate never labeled at all) should fail.
    """
    from aml_triage.eval import load_eval_set

    cases = load_eval_set("data/triage_eval_set.jsonl")
    with open(eval_candidates_path) as f:
        candidate_count = sum(1 for _ in f)
    assert len(cases) >= candidate_count


def test_every_case_has_the_expected_keys():
    from aml_triage.eval import load_eval_set

    cases = load_eval_set("data/triage_eval_set.jsonl")
    for case in cases:
        assert set(case.keys()) == {"transaction", "classifier_score", "label_decision", "label_note"}


def test_label_decisions_are_within_the_allowed_set():
    from aml_triage.eval import load_eval_set

    cases = load_eval_set("data/triage_eval_set.jsonl")
    assert {c["label_decision"] for c in cases}.issubset({"escalate", "monitor", "close"})


def test_more_than_one_decision_type_was_actually_used():
    from aml_triage.eval import load_eval_set

    cases = load_eval_set("data/triage_eval_set.jsonl")
    assert len({c["label_decision"] for c in cases}) >= 2


def test_every_label_note_is_a_real_sentence_not_left_blank():
    from aml_triage.eval import load_eval_set

    cases = load_eval_set("data/triage_eval_set.jsonl")
    for case in cases:
        assert len(case["label_note"].strip()) >= 10
