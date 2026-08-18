# Evaluation and threshold selection

Turn lesson 006's `predict_proba` scores into the two numbers that decide whether this classifier
is fit to hand to a human reviewer: a precision/recall pair earned at one specific threshold, not
just a curve.

## Key concept

**A fraud score is not a decision until something picks a threshold.** `predict_proba` returns a
number between 0 and 1 for every transaction, and by itself that number commits to nothing: it is
only a decision once some rule turns it into "flag this" or "don't." **Precision** is the fraction
of flagged transactions that are actually fraud — it measures how much of a reviewer's time is
spent on real cases versus false alarms. **Recall** is the fraction of actual fraud that got
flagged at all — it measures how much fraud slips through uncaught. The two move in opposite
directions as the threshold moves: lower the bar for "flag this" and recall climbs while precision
falls, because more of what gets flagged is legitimate; raise the bar and precision climbs while
recall falls, because only the most confident flags survive. **PR-AUC** (the area under the
precision-recall curve, `average_precision_score`) summarizes that whole tradeoff curve in one
number, independent of any single threshold — it is what lesson 006 optimized `eval_metric` toward,
and this lesson is where that choice pays off, because PR-AUC is what tells you the curve itself is
worth picking a point on before you commit to one.

Picking that point is a business decision wearing a math costume, and this tutorial pins it as:
**maximize recall subject to precision ≥ 90%.** Read backwards, that says: a compliance team can
tolerate reviewing some number of false alarms — one wrong flag in every ten is the line drawn here
— but every fraud case that clears the threshold and slips through uncaught is a loss with no floor
under it. Fixing the precision floor first and then asking for the most recall available *under*
that floor is what makes the tradeoff legible to someone who has to defend it later: "we catch as
much fraud as we can while keeping false alarms to one in ten" is a sentence a compliance lead can
say to their own leadership. "We picked the threshold with the best F1 score" is not that sentence —
F1 weights precision and recall equally by construction, and nothing about catching fraud makes a
missed case exactly as costly as a false alarm. A model could have an F1-optimal threshold sitting
at 70% precision, which no compliance program would accept, and F1 alone would never surface that;
the precision floor is the thing that encodes what this specific use case can and cannot tolerate,
and F1 has no floor to encode it with.

## Implementation order

1. **Implement `report(y_true, scores, objective) -> dict`** in `src/aml_triage/evaluate.py`. It is
   a pure function: given true labels, predicted scores, and an objective shaped like
   `{"min_precision": 0.90}`, it returns a `dict` with keys `precision`, `recall`, `pr_auc`, and
   `threshold` — it does not write anything to disk, print anything, or know that `aml-triage` has a
   filesystem at all.
2. **Build the full precision-recall curve first.** Call
   `precision, recall, thresholds = precision_recall_curve(y_true, scores)` from
   `sklearn.metrics`, and compute `pr_auc = average_precision_score(y_true, scores)` — both come
   from every score at once, not from a single threshold. `precision_recall_curve` returns one more
   `precision`/`recall` point than it has `thresholds`: its last point (recall 0, precision 1 by
   convention) has no threshold that produced it, so every index this function considers next must
   stay inside `range(len(thresholds))`, never the full length of `precision`.
3. **Filter to thresholds that clear the objective's precision floor**, then take the one with the
   highest recall among those: `candidates = [i for i in range(len(thresholds)) if precision[i] >=
   objective["min_precision"]]`, then `best = max(candidates, key=lambda i: recall[i])`. This is the
   whole "maximize recall subject to precision ≥ X" rule translated directly into code — filter on
   the constraint, then optimize the objective inside what survives the filter.
4. **Raise, don't silently degrade, when nothing clears the floor.** If `candidates` is empty — no
   threshold on this data reaches the requested precision at all — raise a `ValueError` naming the
   unreachable precision, the same defensive instinct lesson 005 already established for
   `compute_scale_pos_weight`'s zero-positives case: a caller that asked for an objective this model
   cannot meet needs a loud failure, not a `dict` whose numbers quietly mean something other than
   what was asked for.
5. **Return `{"precision": ..., "recall": ..., "pr_auc": ..., "threshold": ...}`**, each cast to a
   plain `float` — `precision_recall_curve` and `average_precision_score` return numpy scalars, and
   a plain `float` is what makes the next step's `json.dump` work without a custom encoder.
6. **As a separate step, call `report` with the real model's scores and save the result.** Lesson
   006's `train_baseline` returns a fitted model; take its test-split scores with
   `model.predict_proba(X_test)[:, 1]` (the second column is the probability of the positive class,
   `isFraud == 1`), call `report(y_test, scores, {"min_precision": 0.90})`, and write the returned
   `dict` to `aml-triage/reports/phase1_report.json`:
   ```python
   import json
   json.dump(result, open("reports/phase1_report.json", "w"), indent=2)
   ```
   This file — not `report`'s return value sitting in a variable that disappears when the script
   ends — is the concrete Phase 1 deliverable: a real precision, recall, PR-AUC, and threshold,
   committed to disk, that someone other than the person who trained the model can open and read.

### If you ask the tutor to do this step for you

Like lessons 003–006's functions, `report` has a signature fixed by this spec and a decision rule
(filter on the precision floor, then maximize recall) that leaves no judgment call for the doer to
make, so if asked, it writes `src/aml_triage/evaluate.py` by hand to match
`report(y_true, scores, objective) -> dict` exactly, including the `ValueError` guard. The doer can
also do step 6 itself: calling the already-implemented `report` with the model's test-split scores
and writing the resulting `dict` to `aml-triage/reports/phase1_report.json` needs no shell access —
it is a Python call and a file write, both inside the doer's write-to-`aml-triage` boundary, the
same boundary that already lets it write `src/aml_triage/evaluate.py` itself. What the doer cannot
do is decide the objective's precision floor for you or judge whether the resulting threshold is
one a real compliance team should actually operate with — those are read from this spec and from
the numbers `report` returns, not invented by the doer.

## Checks

Ask the learner to answer these in their own words:

- Precision and recall move in opposite directions as the threshold moves. In your own words, what
  does the compliance team lose by picking a threshold with higher recall but lower precision, and
  what does it lose the other way around?
- `report` filters to thresholds meeting the precision floor and then maximizes recall among
  *those*. Why is that a different threshold, in general, than the one with the single best F1
  score — and what would have to be true about the fraud-versus-false-alarm cost for F1's threshold
  and this objective's threshold to actually coincide?
- `pr_auc` is computed once, over the whole curve, before any threshold is chosen. What does it tell
  you that a single `(precision, recall, threshold)` triple from `report` cannot, and what does the
  triple tell you that `pr_auc` alone cannot?
- If a learner passed `model.predict_proba(X_test)[:, 0]` — the *negative*-class column — into
  `report` instead of column `1`, would the four-key shape of the returned `dict` reveal the
  mistake? What would actually look wrong?

Then run the baked-in check:

```sh
uv run pytest ../aml-tutor/tests/007_test_evaluation.py
```

```json validation
[
  {
    "id": "007-evaluation-and-threshold",
    "label": "Evaluation and threshold",
    "command": "uv",
    "args": ["run", "pytest", "../aml-tutor/tests/007_test_evaluation.py"],
    "cwd": "../aml-triage"
  }
]
```

## Pressure test

Today's tests check `report`'s own arithmetic against a fixed, hand-supplied `y_true`/`scores`
pair, chosen so the exact precision, recall, and PR-AUC are known in advance — they say nothing
about whether the *live* model's scores from lesson 006 produce a threshold worth operating with.
A model whose scores are only weakly separated between classes can still make `report` run without
error and return a well-formed `dict`; the `ValueError` guard only fires when *no* threshold clears
the precision floor, not when the recall attached to the threshold that does clear it is small
enough to be operationally useless. Reading `reports/phase1_report.json` after step 6 and asking
whether its recall number is one a compliance team could live with is a judgment call this test
suite cannot make for you — the same way lesson 006's pressure test noted that a degenerate model
can still clear a bare ROC-AUC-over-0.5 bar. If the saved report's recall looks too low to be
useful, the fix is almost never in this lesson's threshold-selection code; it is in whether the
upstream model, features, or class-imbalance weight actually gave the scores enough separation to
work with.

## Closing note (last lesson)

This is the last lesson of the tutorial. **Phase 1 is complete and defensible even if nothing
further is built.** Walk back through what actually exists at this point: a time-based split that
never leaks the future into training (lesson 003), engineered features built only from information
available at transaction time (lesson 004), a `scale_pos_weight` computed from the train split's
own class balance so the model isn't ignoring the minority class by default (lesson 005), a fitted
XGBoost classifier trained with that weight and evaluated with `aucpr` (lesson 006), and — as of this
lesson — a precision, a recall, a PR-AUC, and a chosen operating threshold, all backed by a real
precision-recall curve and saved to `aml-triage/reports/phase1_report.json` rather than left as a
number nobody wrote down. That is a real, working fraud classifier with a documented, defensible
operating point, not a proof of concept waiting on more work to become useful. Phases 2 and 3 of the
larger AML project — whatever they turn out to be — would build *on* this Phase 1, not *toward* it;
nothing about stopping here leaves this deliverable half-finished.
