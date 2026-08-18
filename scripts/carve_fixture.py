#!/usr/bin/env python3
"""One-time script: carve a small, deterministic PaySim fixture for aml-tutor's baked-in tests.

Usage: python scripts/carve_fixture.py /path/to/full/paysim.csv

Never run against a re-hosted copy of the dataset — only against your own download from Kaggle's
"Synthetic Financial Datasets For Fraud Detection" (ealaxi/paysim1).
"""
import sys
from pathlib import Path

import pandas as pd

SEED = 20260818
NON_FRAUD_SAMPLE_SIZE = 1200
STEP_RANGE = (1, 50)  # bounded window; widened if it doesn't satisfy the checks below
FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "paysim_fixture.csv"
PROVENANCE_PATH = FIXTURE_PATH.parent / "PROVENANCE.md"


def carve(full_csv_path: str) -> None:
    df = pd.read_csv(full_csv_path)
    windowed = df[(df["step"] >= STEP_RANGE[0]) & (df["step"] <= STEP_RANGE[1])]

    fraud = windowed[windowed["isFraud"] == 1]
    fraud_types = set(fraud["type"].unique())
    if not {"TRANSFER", "CASH_OUT"}.issubset(fraud_types):
        raise SystemExit(
            f"step range {STEP_RANGE} does not contain both fraud types (found {fraud_types}); "
            "widen STEP_RANGE in this script and re-run."
        )

    zero_balance = windowed[(windowed["oldbalanceOrg"] == 0) & (windowed["newbalanceOrig"] == 0)]
    if zero_balance.empty:
        raise SystemExit(
            f"step range {STEP_RANGE} has no zero-balance row; widen STEP_RANGE and re-run."
        )

    non_fraud_pool = windowed[windowed["isFraud"] == 0]
    non_fraud = non_fraud_pool.sample(
        n=min(NON_FRAUD_SAMPLE_SIZE, len(non_fraud_pool)), random_state=SEED
    )

    steps_present = sorted(windowed["step"].unique())
    if len(steps_present) < 2:
        raise SystemExit(f"step range {STEP_RANGE} has fewer than 2 distinct steps; widen it.")
    split_boundary_step = steps_present[len(steps_present) // 2]
    step_after_boundary = next(s for s in steps_present if s > split_boundary_step)

    fixture = (
        pd.concat([fraud, zero_balance.head(1), non_fraud])
        .drop_duplicates()
        .sort_values(["step"])
        .reset_index(drop=True)
    )

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fixture.to_csv(FIXTURE_PATH, index=False)

    PROVENANCE_PATH.write_text(
        "# paysim_fixture.csv provenance\n\n"
        '- Source: Kaggle, "Synthetic Financial Datasets For Fraud Detection" (ealaxi/paysim1).\n'
        f"- Carved by: scripts/carve_fixture.py, seed={SEED}, step range {STEP_RANGE}.\n"
        f"- Rows: {len(fixture)}.\n"
        f"- Split-boundary step (lesson 003): {split_boundary_step} "
        f"(train: step <= {split_boundary_step}; test starts at step {step_after_boundary}).\n"
        "- License permits redistribution of a small slice; this fixture is not the full "
        "dataset and was never committed alongside it.\n"
    )
    print(f"Wrote {len(fixture)} rows to {FIXTURE_PATH}")
    print(f"Split-boundary step: {split_boundary_step} (next present step: {step_after_boundary})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/carve_fixture.py /path/to/full/paysim.csv")
    carve(sys.argv[1])
