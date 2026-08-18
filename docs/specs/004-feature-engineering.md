# Feature engineering

Turn the raw PaySim columns into the numeric features a classifier can actually train on, and draw
an explicit line between what a model is allowed to see and what it isn't.

## Key concept

**Every column that survives into training either encodes information available at decision time,
or it doesn't belong.** Three columns get dropped outright:

- `nameOrig` and `nameDest` are transaction identifiers — free-form account strings like
  `C1305486145`, visible directly in the fixture. A model that memorizes which specific account IDs
  turned out to be fraudulent has learned nothing that generalizes to an account it has never seen;
  it has just memorized labels by another name.
- `isFlaggedFraud` is label-adjacent: it is itself a signal about whether a transaction was fraud,
  produced by inspecting the outcome the classifier is trying to predict. Keeping it in the feature
  set risks the model latching onto a near-copy of the target instead of learning the underlying
  transaction pattern — exactly the kind of shortcut that looks like a great score in training and
  falls apart the moment it meets a transaction this flag hasn't already judged.

Two more numeric features get derived and the column they come from gets dropped once they exist:
`orig_balance_delta` (`newbalanceOrig - oldbalanceOrg`) and `dest_balance_delta`
(`newbalanceDest - oldbalanceDest`). Both are decision-time-safe — everything they're computed from
is already known the instant the transaction is recorded, no peeking into the future required.

The delta is the deliberate choice here, not a ratio (`newbalance / oldbalanceOrg`). A ratio divides
by zero whenever `oldbalanceOrg` is zero, and that is not a rare edge case to hand-wave past —
running the loader against the tutorial's own fixture shows 392 of its 1791 rows have
`oldbalanceOrg == 0` and `newbalanceOrig == 0` at once (an account starting and ending a transaction
at zero balance). This is exactly why the fixture was built to deliberately contain a zero-balance
row in the first place: a ratio-based feature would produce `0/0`, undefined and typically rendered
as `NaN` by the arithmetic, on every one of those rows — and a `NaN` feature silently breaks
training two lessons from now. A delta has no such failure mode: `0 - 0` is simply `0`, a real
number that says "no balance change," which is the correct thing to say about that row.

## Implementation order

1. **Implement `add_features(df) -> DataFrame`** in `src/aml_triage/features.py`. Start by dropping
   the three leaky columns: `nameOrig`, `nameDest`, `isFlaggedFraud`.
2. **Derive the balance deltas.** Add `orig_balance_delta = newbalanceOrig - oldbalanceOrg` and
   `dest_balance_delta = newbalanceDest - oldbalanceDest` as new numeric columns.
3. **Derive the transaction-type flags.** PaySim's `type` column is a string category
   (`TRANSFER`, `CASH_OUT`, `PAYMENT`, `CASH_IN`, `DEBIT`). Add `is_transfer` (1 where
   `type == "TRANSFER"`, 0 otherwise) and `is_cash_out` (1 where `type == "CASH_OUT"`, 0 otherwise)
   as integer columns — fraud in PaySim concentrates in transfers and cash-outs, so these two flags
   carry most of what `type` has to offer even without keeping every category.
4. **Drop the raw `type` column** once both flags are derived from it — it has done its job and a
   raw string column has no place in a numeric feature matrix.

### If you ask the tutor to do this step for you

Like lesson 003's `temporal_split`, `add_features` is a small pure function with a signature already
fixed by this spec, and nothing about it requires a shell command or a file the doer can't see. If
asked, the doer writes `src/aml_triage/features.py` by hand: drop the three leaky columns, derive
the two balance deltas and the two type flags, then drop the raw `type` column — matching the
signature `add_features(df: pd.DataFrame) -> pd.DataFrame` exactly. The baked-in check below verifies
the result against the tutorial's own fixture, including its zero-balance rows.

## Checks

Ask the learner to answer these in their own words:

- Why are `nameOrig` and `nameDest` excluded — what specifically would a model do with them that a
  generalizing model shouldn't?
- Why is `isFlaggedFraud` excluded, and how is that reason different from the reason the two name
  columns are excluded?
- Why is `type` dropped even though it isn't leaky the way the other three columns are — what
  happens to the information it carried?
- Why does `newbalance - oldbalance` avoid a failure mode that `newbalance / oldbalance` doesn't,
  and which rows in the fixture would expose that failure mode if you used the ratio instead?

Then run the baked-in check:

```sh
uv run pytest ../aml-tutor/tests/004_test_feature_engineering.py
```

```json validation
[
  {
    "id": "004-feature-engineering",
    "label": "Feature engineering",
    "command": "uv",
    "args": ["run", "pytest", "../aml-tutor/tests/004_test_feature_engineering.py"],
    "cwd": "../aml-triage"
  }
]
```

## Pressure test

Lesson 005's class-imbalance weighting and lesson 006's baseline training both consume whatever
`add_features` hands back, directly, as the feature matrix they train on. Neither of those lessons
re-checks that every column is numeric or that none of them are `NaN` — they assume this lesson
already guaranteed it. If a feature added here is left as a string, or produces a `NaN` on some row
this lesson's own tests didn't happen to cover, the failure won't surface here: it will surface two
lessons later, inside `xgboost`'s own error message, looking like a training bug instead of what it
actually is — a features bug that training merely inherited.
