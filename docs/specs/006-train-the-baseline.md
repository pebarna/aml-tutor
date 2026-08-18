# Train the baseline

Fit the first real classifier on the fixture's train split, and hand lesson 007 a model whose
`predict_proba` output means something.

## Key concept

**Gradient-boosted trees are the right first model for this data, not a placeholder for a "real"
one later.** After lesson 004's `add_features`, every row is ten numeric columns — `step`,
`amount`, the four raw balance columns, the two balance deltas, and the two transaction-type
flags — plus the `isFraud` label. There is no image, no free text, no sequence for a neural
network to learn a representation of; the columns already *are* the representation. Representation
learning earns its cost (more data, more compute, much less interpretability) when the raw input
isn't yet in a form a simple model can use directly. A handful of numeric and binary columns from a
tabular transaction log is already in exactly that form, which is why gradient-boosted trees —
XGBoost here — routinely match or beat neural nets on tabular data at this scale, and do it with
less tuning and less data than a network needs to find the same signal.

XGBoost earns its place for two more reasons specific to this problem. First, `scale_pos_weight`
from lesson 005 is a first-class constructor argument, not something bolted on: the library was
built assuming exactly this class-imbalance correction, so lesson 005's output plugs straight in.
Second, a boosted-tree model stays inspectable — feature importances and individual tree splits can
be read back out — which matters for a monitoring model whose flags a human compliance reviewer
has to be able to justify, not just trust. A neural net buys none of that back for data this small
and this already-numeric; it would cost more to train and tune while giving up the one property
(interpretability) that matters most for this use case.

## Implementation order

1. **Implement `train_baseline(X_train, y_train, weight) -> xgb.XGBClassifier`** in
   `src/aml_triage/model.py`. It takes the already-featured train matrix from lesson 004, the
   train labels, and the `float` from lesson 005's `compute_scale_pos_weight` — nothing in this
   function computes that weight itself; it only consumes it.
2. **Construct the classifier with a fixed, small hyperparameter set**: `n_estimators=100`,
   `max_depth=4`, `learning_rate=0.1`, `scale_pos_weight=weight`, `eval_metric="aucpr"`,
   `random_state=42`. `max_depth=4` keeps each tree shallow enough to stay readable and resist
   overfitting on a training set this size; `random_state=42` makes two runs on the same data
   produce the same model, which matters when a learner is trying to reason about *why* a score
   changed between two runs rather than chasing random variation.
3. **Set `eval_metric="aucpr"`, not the default `logloss`.** Log-loss weights every row's error
   equally regardless of class, which is exactly the bias lesson 005 exists to counteract; area
   under the precision-recall curve is the metric that stays informative when the positive class is
   rare, so it is the more honest thing for this specific dataset to optimize against.
4. **Fit and return the model**: `model.fit(X_train, y_train)`, then `return model`. The function
   does nothing with the test split — it only sees `X_train` and `y_train`, plus the weight lesson
   005 already computed from `y_train` alone.

### If you ask the tutor to do this step for you

Like lessons 003–005's functions, `train_baseline` has a signature and a hyperparameter set both
already fixed by this spec, so nothing about *writing* it requires a judgment call the doer can't
make. If asked, the doer writes `src/aml_triage/model.py` by hand: construct `xgb.XGBClassifier`
with exactly the seven arguments listed above, call `.fit(X_train, y_train)`, and return the fitted
model — matching `train_baseline(X_train, y_train, weight) -> xgb.XGBClassifier` exactly. What the
doer cannot do is *run* it: fitting a real model and checking its score requires executing Python
against the fixture, which needs a shell the doer doesn't have. The baked-in check below is what
actually verifies the model fits, predicts valid probabilities, and beats chance — running it
yourself is not optional the way it might feel optional for a pure function you could eyeball.

## Checks

Ask the learner to answer these in their own words:

- Why does `scale_pos_weight` get passed into `train_baseline` as an argument computed by lesson
  005's function, rather than baked into `add_features` as another engineered column?
- Why is `eval_metric` set to `"aucpr"` instead of left at XGBoost's default, given what lesson 005
  established about this dataset's class balance?
- `train_baseline` never looks at `X_test` or `y_test`. What's the one thing this function does see
  that was itself computed under a rule from an earlier lesson, and which lesson set that rule?
- If a learner called `compute_scale_pos_weight(y_test)` instead of `compute_scale_pos_weight(y_train)`
  by mistake and passed *that* weight into `train_baseline`, would either of today's two tests catch
  it? Why or why not?

Then run the baked-in check:

```sh
uv run pytest ../aml-tutor/tests/006_test_train_baseline.py
```

```json validation
[
  {
    "id": "006-train-the-baseline",
    "label": "Train the baseline",
    "command": "uv",
    "args": ["run", "pytest", "../aml-tutor/tests/006_test_train_baseline.py"],
    "cwd": "../aml-triage"
  }
]
```

## Pressure test

Today's two tests check that the model fits without error, that every predicted probability is a
valid number between 0 and 1, and that the model's ROC-AUC on the test split beats 0.5. None of that
guarantees the model is *useful* at any single decision threshold. A model with a subtle training
bug — say, one that has collapsed toward predicting one class almost everywhere, from a weight
computed on the wrong split or a label column dropped in the wrong place — can still emit valid
probabilities and even rank cases correctly enough to clear a bare ROC-AUC-over-0.5 bar, because
ROC-AUC only cares about relative ordering, not about whether any of the predicted probabilities are
concentrated somewhere a real threshold could use. That same model would show its damage the moment
lesson 007 turns these `predict_proba` scores into a precision-recall curve and picks an operating
threshold: a degenerate curve — precision collapsing everywhere, or recall unreachable at any
threshold worth flagging — would surface there, not here. If lesson 007's numbers look wrong, the
bug likely lives in this lesson's training call, not in its own threshold logic.
