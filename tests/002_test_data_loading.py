"""Baked-in check for lesson 002 — load and explore the data.

Run via `uv run pytest ../aml-tutor/tests/002_test_data_loading.py` with cwd=aml-triage, so
`import aml_triage` resolves to the student's package. fixture_path comes from tests/conftest.py
(Task 1).
"""
import pandas as pd

EXPECTED_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]


def test_columns_and_row_count_match_the_fixture(fixture_path):
    from aml_triage.data import load_transactions

    df = load_transactions(str(fixture_path))
    raw_row_count = sum(1 for _ in open(fixture_path)) - 1  # minus header
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == raw_row_count


def test_dtypes_are_numeric_where_expected(fixture_path):
    from aml_triage.data import load_transactions

    df = load_transactions(str(fixture_path))
    for column in ["step", "amount", "oldbalanceOrg", "newbalanceOrig",
                    "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]:
        assert pd.api.types.is_numeric_dtype(df[column]), f"{column} is not numeric"


def test_fraud_rate_matches_an_independent_recomputation(fixture_path):
    from aml_triage.data import load_transactions

    df = load_transactions(str(fixture_path))
    raw = pd.read_csv(fixture_path)
    assert df["isFraud"].mean() == raw["isFraud"].mean()
    assert 0 < df["isFraud"].mean() < 0.5, "fixture should be fraud-heavy but still a minority class"
