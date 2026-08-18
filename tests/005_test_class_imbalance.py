"""Baked-in check for lesson 005 — class imbalance."""
import pytest


def test_weight_equals_negative_over_positive_on_the_train_split(fixture_path, split_step):
    from aml_triage.data import load_transactions
    from aml_triage.split import temporal_split
    from aml_triage.imbalance import compute_scale_pos_weight

    df = load_transactions(str(fixture_path))
    train_df, _ = temporal_split(df, split_step)
    y_train = train_df["isFraud"]
    weight = compute_scale_pos_weight(y_train)
    expected = (y_train == 0).sum() / (y_train == 1).sum()
    assert weight == pytest.approx(expected)
    assert weight > 1, "fraud should be the minority class in the train split"


def test_raises_a_clear_error_with_no_positive_examples():
    from aml_triage.imbalance import compute_scale_pos_weight
    import pandas as pd

    all_negative = pd.Series([0, 0, 0])
    with pytest.raises(ValueError):
        compute_scale_pos_weight(all_negative)
