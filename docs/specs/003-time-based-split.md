# Time-based train/test split

Split the data by when it happened, not by chance, so the evaluation you build in later lessons
tells you something true about a model deployed in production tomorrow.

## Key concept

**A random split leaks the future into training.** `sklearn.model_selection.train_test_split`
with `shuffle=True` (its default) would happily put a row from step 40 in the training set and a
row from step 10 in the test set. That model learns from an outcome that, at the moment the step
10 transaction was actually flagged, had not happened yet — nobody deciding whether to block a
transaction at step 10 gets to see what the world looked like at step 40. Training on it makes the
model look better than it can possibly be at decision time, and the gap between that inflated
score and real performance only shows up after deployment, when it is expensive.

`step` is PaySim's clock: it counts hours since the simulation started, 1 through 743 across the
roughly 31-day (744-hour) run the dataset represents — confirmed against the real
`paysim_sample.csv` from lesson 002, not just the fixture's own carved-down window of steps 1–50.
A time-based split respects that clock: everything at or before some boundary step is "the past"
(train), everything strictly after it is "the future" (test). That mirrors the real deployment
question — given only what was known up to a point in time, how well does the model do on what
comes next?

The one rule that keeps this honest is the **tie rule: a `step` value is never divided between
train and test.** All rows sharing a step either land in train together or in test together, never
split across both. Without that rule, two transactions that happened in the same hour — arguably
the same instant, as far as this simulation's clock resolution goes — could end up on opposite
sides of the boundary, which reintroduces exactly the same-moment leakage the split exists to
prevent.

## Implementation order

1. **Implement the split.** Write `temporal_split(df, split_step) -> tuple[pd.DataFrame, pd.DataFrame]`
   in `src/aml_triage/split.py`. It partitions `df` into a train frame containing every row with
   `step <= split_step` and a test frame containing every row with `step > split_step`, and returns
   `(train_df, test_df)`.
2. **Reset the index on both halves.** Filtering a DataFrame keeps the original row labels, so
   `train_df` and `test_df` come out with gaps in their index unless you call
   `.reset_index(drop=True)` on each. Later lessons that iterate or join on position, not on the
   original label, depend on this.
3. **Pick `split_step` from the data you're splitting, not a constant.** The tutorial's own fixture
   documents its boundary in `tests/fixtures/PROVENANCE.md` — step 26, chosen so that every row at
   step 26 lands in train and the first test row is step 27. Your real run should choose a boundary
   the same way: a step value that appears in your sample, near wherever you want the train/test
   proportions to land, not a number picked in the abstract.

## Checks

Ask the learner to answer these in their own words:

- Why does chronological order matter for *this* split specifically, when plenty of other
  train/test splits in machine learning are drawn randomly without anyone worrying about it?
- What would go wrong, concretely, if `temporal_split` used `step < split_step` for train and
  `step >= split_step` for test instead of `<=` and `>`? (Trick question: nothing goes wrong with
  that pair on its own — walk through why the two comparisons still have to be complements of each
  other, and what breaks if they aren't.)
- If two rows share the exact same `step` value, which split do they both belong to, and why is
  "both must agree" non-negotiable rather than a preference?

Then run the baked-in check:

```sh
uv run pytest ../aml-tutor/tests/003_test_time_based_split.py
```

```json validation
[
  {
    "id": "003-time-based-split",
    "label": "Time-based split",
    "command": "uv",
    "args": ["run", "pytest", "../aml-tutor/tests/003_test_time_based_split.py"],
    "cwd": "../aml-triage"
  }
]
```

## Pressure test

Lesson 004 builds features on top of whichever split it's handed, and some of the features it will
be tempted to write — a rolling average of the last few transactions, a per-account running
balance — are computed over a *window* of rows, not a single row in isolation. If that window is
allowed to span the train/test boundary, a feature computed for a training row could quietly
depend on test rows that come after it in time. That feature would pass every test in this lesson
today — `temporal_split` itself is correct — and still leak the future, because the leak lives in
lesson 004's code, not in the split it built on. The fix has to hold at the point features are
computed: every feature must be derivable from its own split alone, train features from train rows
only and test features from test rows only, with no window reaching across the boundary this
lesson drew.
