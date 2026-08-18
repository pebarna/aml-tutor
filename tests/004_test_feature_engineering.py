"""Baked-in check for lesson 004 — feature engineering."""
import pandas as pd

LEAKY_COLUMNS = ["nameOrig", "nameDest", "isFlaggedFraud", "type"]


def test_leaky_and_raw_type_columns_are_dropped(fixture_path):
    from aml_triage.data import load_transactions
    from aml_triage.features import add_features

    df = load_transactions(str(fixture_path))
    featured = add_features(df)
    for column in LEAKY_COLUMNS:
        assert column not in featured.columns, f"{column} should have been dropped"


def test_derived_columns_exist_and_are_numeric(fixture_path):
    from aml_triage.data import load_transactions
    from aml_triage.features import add_features

    df = load_transactions(str(fixture_path))
    featured = add_features(df)
    for column in ["orig_balance_delta", "dest_balance_delta", "is_transfer", "is_cash_out"]:
        assert column in featured.columns
        assert pd.api.types.is_numeric_dtype(featured[column])


def test_balance_deltas_are_correct_on_known_rows(fixture_path):
    from aml_triage.data import load_transactions
    from aml_triage.features import add_features

    df = load_transactions(str(fixture_path))
    featured = add_features(df)
    expected_orig_delta = df["newbalanceOrig"] - df["oldbalanceOrg"]
    expected_dest_delta = df["newbalanceDest"] - df["oldbalanceDest"]
    assert (featured["orig_balance_delta"] == expected_orig_delta).all()
    assert (featured["dest_balance_delta"] == expected_dest_delta).all()


def test_zero_balance_row_does_not_produce_a_crash_or_nan(fixture_path):
    from aml_triage.data import load_transactions
    from aml_triage.features import add_features

    df = load_transactions(str(fixture_path))
    zero_balance_mask = (df["oldbalanceOrg"] == 0) & (df["newbalanceOrig"] == 0)
    assert zero_balance_mask.sum() >= 1, "fixture should contain a zero-balance row (Task 1)"
    featured = add_features(df)
    assert not featured.loc[zero_balance_mask, "orig_balance_delta"].isna().any()
    assert (featured.loc[zero_balance_mask, "orig_balance_delta"] == 0).all()


def test_type_flags_match_the_raw_type_column(fixture_path):
    from aml_triage.data import load_transactions
    from aml_triage.features import add_features

    df = load_transactions(str(fixture_path))
    featured = add_features(df)
    assert (featured["is_transfer"] == (df["type"] == "TRANSFER").astype(int)).all()
    assert (featured["is_cash_out"] == (df["type"] == "CASH_OUT").astype(int)).all()
