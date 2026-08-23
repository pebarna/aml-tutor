# Deterministic triage checks

Once you have a triage agent (lesson 011) and a hand-labeled eval set (lesson 012), you can start
measuring how often the agent's decisions agree with your own judgment. But before you deploy a full
LLM-as-judge (lesson 014) to score each case, build two fast, deterministic checks first: did the
agent cite anything at all, and did its decision match yours.

These two checks catch the easiest failures (no citation, decision mismatch) before you pay for model
inference on harder questions like whether the cited typology was actually relevant.

## Key concept

SEED.md's Phase 3 calls for deterministic checks on two dimensions: citation *validity* (does the id
exist, and was it shown to the model?) and citation *presence* (did the model cite anything?), plus
decision *agreement*. By design, these checks are split across lessons:

- **Citation *validity*** was already enforced upstream, at generation time, by lesson 010's
  `parse_triage_decision` guard. Any `result` that reaches `deterministic_score` structurally
  cannot cite a non-existent id or an id that wasn't shown to the model — validity is already
  guaranteed by the time the result is in your hands.
- **Citation *presence*** and **decision *agreement*** are what's left to check here: did the model
  cite *something* (anything), and did it pick the same escalate/monitor/close call you did?

**At a regulated shop:** deterministic, code-level checks like these are the cheap, always-on layer
of a validation suite — necessary but not sufficient. Model-risk reviewers require this level of
mechanical checking as table stakes, but they also require a human sign-off and a documented
judgment about whether the patterns you're seeing make sense. A `decision_match: 0.92` might sound
good until you spot the pattern that the agent escalates everything on Fridays, or never cites
typologies from the SANCTIONS family. The numbers alone won't catch those — only reading a sample
of the results, and asking whether the agent's reasoning is actually sound, will.

## Implementation order

Implement `deterministic_score(case, result) -> dict` in `src/aml_triage/eval.py`. It takes:

- `case`: one row from your hand-labeled eval set (lesson 012's shape — has `label_decision`)
- `result`: the return value of `triage(...)` (lesson 011's shape — has `decision` and
  `cited_typology_ids`)

And returns a dict with two boolean keys:

- `decision_match`: `True` if `result["decision"] == case["label_decision"]`
- `citation_present`: `True` if `len(result["cited_typology_ids"]) > 0`

```python
def deterministic_score(case, result):
    return {
        "decision_match": result["decision"] == case["label_decision"],
        "citation_present": len(result["cited_typology_ids"]) > 0,
    }
```

## Checks

Comprehension questions, then the baked-in test:

- Why is citation *validity* (upstream, in lesson 010) a different check from citation *presence*
  (here)? What does the `parse_triage_decision` guard prevent that `deterministic_score` does not?
- Why is it worth checking `citation_present` even though you're also planning an LLM-as-judge in
  lesson 014? What failure mode does `citation_present == False` catch that the LLM-as-judge alone
  would not?

Run the baked-in check:

```sh
uv run pytest ../aml-tutor/tests/013_test_deterministic_checks.py -v
```

All three tests must pass:

- `test_decision_match_true_when_decisions_agree`: confirms `decision_match` is `True` when result
  and case decisions match.
- `test_decision_match_false_when_decisions_disagree`: confirms `decision_match` is `False` when
  they don't.
- `test_citation_present_false_when_nothing_was_cited`: confirms `citation_present` is `False` when
  `cited_typology_ids` is empty.

```json validation
[
  {
    "id": "013-deterministic-triage-checks",
    "label": "Deterministic triage checks",
    "command": "uv",
    "args": ["run", "pytest", "../aml-tutor/tests/013_test_deterministic_checks.py", "-v"],
    "cwd": "../aml-triage"
  }
]
```

## Pressure test

A model that always cites *something* real but never actually engages with the transaction data
would pass `citation_present` every time. This check catches the "cited nothing" failure, not the
"cited something irrelevant" failure — the latter is what lesson 014's LLM-as-judge is for.

Imagine an agent that, when in doubt, just returns `{"decision": "monitor", "cited_typology_ids":
["TY-999"]}` without reasoning. Your `deterministic_score` would flag every case where you labeled
it `escalate` or `close`, catching the decision mismatch. But it wouldn't catch that the agent
never read your transaction — it would pass `citation_present` with flying colors.

That's fine. That's exactly why you need lesson 014.

## Doer fallback note

There is **no shell command fallback here** — `deterministic_score` is pure Python with no external
tool. If asked to do this step for you, the doer can write `deterministic_score` by hand directly
into your `src/aml_triage/eval.py`, using the reference implementation above.
