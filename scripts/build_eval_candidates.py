#!/usr/bin/env python3
"""One-time script: select unlabeled triage eval-set candidates from the committed PaySim fixture.

Usage: python scripts/build_eval_candidates.py

This produces CANDIDATES, not labels -- lesson 012 has the student hand-label each one themselves.
classifier_score is a documented stand-in (0.95/0.05 by the source row's isFraud flag), not a real
Phase 1 model inference -- there is no persisted Phase 1 model artifact to reload here.
"""
import json
from pathlib import Path

import pandas as pd

SEED = 20260823
N_PER_CLASS = 8
FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "paysim_fixture.csv"
OUTPUT_PATH = FIXTURE_PATH.parent / "triage_eval_candidates.jsonl"

TRANSACTION_COLUMNS = [
    "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
]


def build() -> None:
    df = pd.read_csv(FIXTURE_PATH)
    fraud = df[df["isFraud"] == 1]
    non_fraud = df[df["isFraud"] == 0]

    if len(fraud) < N_PER_CLASS or len(non_fraud) < N_PER_CLASS:
        raise SystemExit(
            f"need >= {N_PER_CLASS} rows in each class, found {len(fraud)} fraud / "
            f"{len(non_fraud)} non-fraud; lower N_PER_CLASS or widen the source fixture, don't let "
            "this silently produce fewer than 16 candidates"
        )

    fraud_sample = fraud.sample(n=N_PER_CLASS, random_state=SEED)
    non_fraud_sample = non_fraud.sample(n=N_PER_CLASS, random_state=SEED)

    rows = []
    for _, row in pd.concat([fraud_sample, non_fraud_sample]).iterrows():
        is_fraud = bool(row["isFraud"])
        rows.append({
            "transaction": {col: row[col] for col in TRANSACTION_COLUMNS},
            "classifier_score": 0.95 if is_fraud else 0.05,
        })

    with OUTPUT_PATH.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n = N_PER_CLASS
    print(f"Wrote {len(rows)} candidates ({n} fraud-flagged, {n} not) to {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
