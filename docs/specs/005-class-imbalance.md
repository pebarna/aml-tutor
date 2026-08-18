# Class imbalance handling

Turn the train split's own fraud rate into a single number XGBoost can use to stop ignoring the
minority class, without turning it into a tool that itself leaks the future.

## Key concept

**`scale_pos_weight` changes what a wrong answer costs, not what the model sees.** XGBoost's loss
function scores every training row and adds up the total error to minimize. With no weighting,
every row counts the same, and when the positive class (`isFraud == 1`) is a small fraction of the
rows, the arithmetic favors a model that mostly ignores it: getting every fraud row wrong but every
legitimate row right is a low total loss when fraud rows are rare, so that is the shortcut gradient
descent finds. `scale_pos_weight` multiplies the loss contribution of every positive-class row by a
fixed factor, so a missed fraud row costs proportionally more than a missed legitimate row — the
model can no longer minimize its loss by simply predicting "not fraud" and being right most of the
time.

The standard formula, and the one this lesson implements, is `count(negative) / count(positive)`
on the training labels. That is not the only choice available, but it is not a free knob either:
push it higher than that ratio and the model starts flagging more transactions than the true fraud
rate justifies — precision drops, because more of what gets flagged is actually legitimate, and
whoever reviews those flags spends more time on false alarms. Push it lower — toward the unweighted
default of 1 — and the model drifts back toward its natural bias against the minority class:
recall drops, because real fraud stops crossing whatever threshold turns a predicted probability into
a flag. Lesson 007's threshold selection is the other lever on this same tradeoff; this lesson sets
the weight the model trains with, not the threshold it's read against, and the two interact rather
than substitute for each other.

Running this against the tutorial's own fixture (`tests/fixtures/paysim_fixture.csv`, split at
`step <= 26` per `PROVENANCE.md`) shows the train split contains 939 rows: 301 with `isFraud == 1`
and 638 with `isFraud == 0`, giving `638 / 301 ≈ 2.12`. Computing the same ratio over the *entire*
fixture instead — 1791 rows, 590 positive, 1201 negative — gives `1201 / 590 ≈ 2.04`. Those two
numbers are close on this particular fixture but not identical, and the fixture itself is far more
fraud-heavy than the real PaySim sample lesson 002 loads (whose fraud rate is roughly 0.1%, not the
roughly 30% these counts imply); on the real sample the gap between a train-only weight and a
whole-dataset weight would not be this small by coincidence. The number always has to come from
*train* labels, computed the same way every other train-only quantity in this tutorial is: using
only what would actually be known at the point a real model is trained, which by lesson 003's own
rule excludes anything about the test split's composition.

## Implementation order

1. **Implement `compute_scale_pos_weight(y: pd.Series) -> float`** in `src/aml_triage/imbalance.py`.
   Count `positive = (y == 1).sum()` and `negative = (y == 0).sum()`, and return `negative / positive`.
2. **Guard the case with no positive examples.** Dividing by zero positives is not a number this
   function should ever silently produce — raise a `ValueError` with a clear message instead, so a
   caller that accidentally hands it an all-negative slice (or the wrong column entirely) gets a
   loud failure instead of an `inf` or a `ZeroDivisionError` traceback with no context.
3. **Call it on the train split's labels only.** The caller (lesson 006, in `train_baseline`) is
   expected to run this on `train_df["isFraud"]` — the output of lesson 003's `temporal_split` — and
   never on the test split or the whole, unsplit DataFrame. This function itself has no way to
   enforce that; it trusts whatever `Series` it's handed, so the discipline lives in the calling
   code, same as lesson 004's leakage rules live in what `add_features` chooses to keep, not in a
   check `add_features` performs on its caller. Computing the weight from the test split's class
   balance instead would leak information into training the same way lesson 003 warned about, just
   applied to a different quantity: the test split's positive/negative counts are only knowable
   because you already hold the labels for data the model hasn't been evaluated on yet, and feeding
   that count into how the model is trained — rather than treating it as something only revealed
   once training and evaluation are both done — is the same category of leak lesson 003 named for
   raw feature values, not a new kind of mistake.

### If you ask the tutor to do this step for you

Like lesson 003's `temporal_split` and lesson 004's `add_features`, `compute_scale_pos_weight` is a
small pure function with a signature already fixed by this spec, and nothing about it requires a
shell command or a file the doer can't see. If asked, the doer writes
`src/aml_triage/imbalance.py` by hand: count positives and negatives, raise `ValueError` when there
are no positives, otherwise return `negative / positive`, matching the signature
`compute_scale_pos_weight(y: pd.Series) -> float` exactly. It does **not** decide which split to
call it on — that choice belongs to lesson 006's `train_baseline` and is outside this function's own
signature — so the baked-in check below verifies only the function's arithmetic and its error
behavior, against the tutorial's own fixture.

## Checks

Ask the learner to answer these in their own words:

- Why does computing this weight from the train split's labels only, rather than the whole fixture
  or the test split, matter here — given that `scale_pos_weight` never touches which rows the model
  is trained on, only how much each row's error counts?
- The train split's own weight (≈2.12) and the whole fixture's weight (≈2.04) are close on this
  fixture. Why doesn't that closeness mean it would be safe to use the whole-fixture number as a
  shortcut on a real, larger sample?
- Concretely, what happens to the *kind* of mistake the model makes if `scale_pos_weight` is set
  much higher than the train split's true `negative / positive` ratio? What if it's left at the
  default of 1 on a dataset this imbalanced?
- Why does an all-negative input deserve a raised `ValueError` instead of whatever number
  `negative / 0` would otherwise produce?

Then run the baked-in check:

```sh
uv run pytest ../aml-tutor/tests/005_test_class_imbalance.py
```

```json validation
[
  {
    "id": "005-class-imbalance",
    "label": "Class imbalance",
    "command": "uv",
    "args": ["run", "pytest", "../aml-tutor/tests/005_test_class_imbalance.py"],
    "cwd": "../aml-triage"
  }
]
```

## Pressure test

Lesson 006's `train_baseline` takes this function's output as a plain `float` argument and passes it
straight into XGBoost's `scale_pos_weight` parameter — it has no way to tell, from the number alone,
whether it was computed on the train split, the test split, or the whole fixture. If lesson 006's
own code (or a learner following it) calls `compute_scale_pos_weight` on the wrong `Series` — the
whole, unsplit DataFrame's labels instead of `train_df["isFraud"]` — this lesson's own test still
passes, because it only checks the function's arithmetic in isolation, and lesson 006's test only
checks that training runs and produces a model. Neither test is positioned to catch a value computed
on the wrong split; the model just trains with a slightly wrong weight and looks fine until its
precision or recall is compared against an expectation that assumed the correct one, which on a
real, much larger sample than this fixture is exactly the kind of quiet miscalibration that erodes
trust in a monitoring model.
