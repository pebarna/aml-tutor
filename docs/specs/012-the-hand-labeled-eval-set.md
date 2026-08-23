# The hand-labeled triage eval set

Load a small helper (`load_eval_set`), then hand-label 16 unlabeled candidate transactions with your
own escalate/monitor/close judgment and a one-sentence rationale. Unlike every other lesson in this
tutorial, the deliverable here isn't code — it's the labels themselves. There is no committed
`aml-tutor` fixture to check this lesson's *content* against, because the whole point is that you
wrote the labels; the baked-in test only checks structure.

## Key concept

SEED.md's Phase 3 asks for hand-annotated cases for a reason: a mechanically generated label can
never produce a genuine borderline `monitor` call, and it's exactly those judgment calls a real
analyst's labeling would surface. A script can decide "score above threshold → escalate," but it
can't decide whether a large CASH_OUT that drains an account to zero is a textbook structuring
pattern or a one-off legitimate withdrawal — that call requires reading the transaction the way an
analyst would, and reasoning about it in your own words.

`tests/fixtures/triage_eval_candidates.jsonl` (Task 1) gives you 16 *unlabeled* candidates — 8
fraud-flagged and 8 not, split by the PaySim simulator's own `isFraud` flag — for you to label.
The `classifier_score` in each row is a documented stand-in (0.95 for the flagged half, 0.05 for the
rest), not a real Phase 1 model inference; treat it as a hint about how the classifier would have
scored the transaction, not as ground truth.

**At a regulated shop:** this is what an SR 11-7-style model-risk review actually requires — a
documented, SME-labeled golden set, periodically refreshed by real humans, not synthetic labels.
Model validation teams specifically look for evidence that a model's outputs were checked against
independent human judgment, not just against its own training distribution. A golden set you
labeled yourself, with a note explaining each call, is a small-scale version of exactly that
artifact.

## Implementation order

1. **Implement `load_eval_set(path) -> list[dict]`** in `src/aml_triage/eval.py` (a new file — this
   is the first of three lessons, 012-014, that add to this same file). It reads a JSON-Lines file
   and returns one dict per line:

   ```python
   import json


   def load_eval_set(path):
       with open(path) as f:
           return [json.loads(line) for line in f if line.strip()]
   ```

2. **Label the 16 candidates.** For each row in `../aml-tutor/tests/fixtures/triage_eval_candidates.jsonl`,
   look at the transaction fields — `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`,
   `oldbalanceDest`, `newbalanceDest` — and decide what you think should happen to it. Optionally
   call your own `triage(...)` from lesson 011 on the transaction first, to see what the agent would
   say, then decide whether you agree or disagree with it. Either way, write your own judgment as
   two new fields added to the row:
   - `label_decision`: one of `"escalate"`, `"monitor"`, or `"close"`
   - `label_note`: a one-sentence explanation of what in the transaction data drove that call

3. **Save all 16 labeled rows** to `aml-triage/data/triage_eval_set.jsonl`, one JSON object per line,
   each with the original `transaction` and `classifier_score` fields plus your new
   `label_decision` and `label_note`.

   Be honest about scope: 16 hand-labeled cases is a floor, not SEED.md's 30-50-case target. Treat
   this file as a starting point and keep labeling more cases on your own time for the real
   deliverable — the more cases you label, and the more consistently you apply your own criteria,
   the more the agreement-rate metric in lesson 015 will actually mean something.

4. **Do not commit scratch code.** The reference implementation of `load_eval_set` lives in
   `/tmp/aml-tutor-plan003-scratch/src/aml_triage/eval.py`. Only the spec and the baked-in test in
   this tutorial's git history. The learner writes their own `eval.py` — and their own labeled
   `data/triage_eval_set.jsonl` — inside their `aml-triage` repo, which is never committed to this
   one.

### If you ask the tutor to do this step for you

There is **no doer fallback here.** Every other lesson in this tutorial has a fallback path where
the tutor can write deterministic code on the learner's behalf once the mechanical prerequisites are
met. This lesson is different: deciding escalate/monitor/close for each of the 16 candidates is
exactly the human judgment this lesson exists to practice, and the doer has no principled way to
make that call for you — it isn't a model-risk analyst, and a set of labels it invented wouldn't be
a golden set, it would just be more synthetic data. `load_eval_set` itself is a three-line function
the tutor could write, but that was never the hard part of this lesson. If asked, the tutor should
say this plainly rather than offering to label the file for you.

## Checks

Rather than generic comprehension questions with a fixed right answer, the tutor asks about your
own labels directly — because there isn't a hidden answer key to check them against, only your own
reasoning:

- Pick one case you marked `escalate` and one you marked `close` (or `monitor`), and explain, in
  your own words, what in the transaction data drove each call.
- The tutor checks that your stated reasoning is actually consistent with the transaction fields in
  front of you — not that it matches some predetermined label. If you said a transaction was
  escalated because balances "look like structuring," can you point at which specific fields (the
  amount, the before/after balances, the transaction type) support that?
- Did you label any two similar-looking transactions differently? If so, what specifically
  distinguished them in your mind? If you can't articulate a distinguishing factor, that's worth
  noticing now — see the Pressure test below.

Then run the baked-in check:

```sh
uv run pytest ../aml-tutor/tests/012_test_eval_set.py -v
```

All five tests must pass:

- `test_eval_set_has_one_row_per_candidate`: confirms your labeled file has exactly as many rows as
  the candidate pool it was built from.
- `test_every_case_has_the_expected_keys`: confirms every row has exactly the four expected keys —
  `transaction`, `classifier_score`, `label_decision`, `label_note` — no more, no less.
- `test_label_decisions_are_within_the_allowed_set`: confirms every `label_decision` is one of
  `escalate`, `monitor`, or `close`.
- `test_more_than_one_decision_type_was_actually_used`: confirms you didn't label everything the
  same way — the pool is a mix of flagged and unflagged transactions on purpose.
- `test_every_label_note_is_a_real_sentence_not_left_blank`: confirms every `label_note` is at least
  10 characters, ruling out an empty string or a placeholder.

```json validation
[
  {
    "id": "012-the-hand-labeled-eval-set",
    "label": "The hand-labeled triage eval set",
    "command": "uv",
    "args": ["run", "pytest", "../aml-tutor/tests/012_test_eval_set.py", "-v"],
    "cwd": "../aml-triage"
  }
]
```

## Pressure test

Lesson 013's `decision_match` compares `triage()` output against these hand-written labels, one
case at a time. That comparison is only as trustworthy as the consistency of the labels behind it.
Labeling two very similar transactions differently with no principled reason will surface downstream
as a confusingly low agreement rate that looks like an agent problem but is actually a
labeling-consistency problem — the agent might be making the same call both times, and it's your own
labels that disagree with each other.

This is a real, well-known issue with human-labeled eval sets — inter-rater and intra-rater
consistency — not unique to this tutorial. It's the same reason a real model-risk review asks for
multiple SMEs to label overlapping subsets and checks their agreement with each other before
trusting the golden set at all. You are, right now, both the SME and the reviewer, so the discipline
has to come from you: apply the same criteria to every case, and write your `label_note` well enough
that a stranger reading it later could tell whether you were being consistent.
