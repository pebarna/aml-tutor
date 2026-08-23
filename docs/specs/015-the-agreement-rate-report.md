# The agreement-rate report

Wire the three validation layers together: deterministic checks (lesson 013) and LLM-as-judge scores (lesson 014) both feed into a single aggregated report that answers the headline question from SEED.md — how often does your triage agent agree with your own judgment on real, hand-labeled cases?

## Key concept

SEED.md names this report's headline number — the decision-agreement rate — "the single highest-leverage output of this whole project." It is now backed by real hand-labeled ground truth (lesson 012), not a generated scaffold, which is exactly what makes it a defensible number to cite. The report aggregates three rates side by side:

- `decision_agreement_rate`: how often the agent's escalate/monitor/close decision matches your label.
- `citation_present_rate`: how often the agent cited at least one typology.
- `judge_agreement_rate`: how often the LLM-as-judge agrees that the agent's rationale plausibly follows from its cited typologies.

Three separate rates, not one combined score, because they diagnose different upstream failures:
- A low `decision_agreement_rate` says the model's judgment is wrong — a classifier or prompt problem.
- A low `citation_present_rate` says the model is not even trying to explain itself — a tool-use or prompt problem.
- A low `judge_agreement_rate` says the model cites typologies but reasons poorly from them — a reasoning or grounding problem.

**At a regulated shop:** this report is the shape of what a model-validation document's metrics section would show, but the real artifact would be versioned, tied to a specific model/prompt version, and re-run on a refreshed golden set on a schedule — not generated once and forgotten. The three rates themselves are useful diagnostic signals, but the real validation work lives in the human review: reading a sample of your hand-labeled cases alongside the agent's decisions and rationales, and asking whether the patterns make sense or whether there's a systematic failure hiding in the aggregates.

## Implementation order

1. **Implement `report(cases, results, deterministic_scores, judge_scores) -> dict`** in `src/aml_triage/eval.py`. The function takes four parallel lists (all the same length) and returns a dict with these keys:

   - `n_cases`: the number of cases evaluated.
   - `decision_agreement_rate`: the fraction of cases where `deterministic_scores[i]["decision_match"]` is `True`.
   - `citation_present_rate`: the fraction of cases where `deterministic_scores[i]["citation_present"]` is `True`.
   - `judge_agreement_rate`: the fraction of cases where `judge_scores[i]["agrees"]` is `True`.

   If the four lists have different lengths, raise `ValueError` with a message that shows the set of lengths seen. If the lists are all empty (`n == 0`), also raise `ValueError` rather than dividing by zero — an empty eval set has nothing to report on.

2. **Run the full pipeline over your hand-labeled cases (separate from pytest).** This is NOT part of the test suite and requires a real `ANTHROPIC_API_KEY`:

   ```python
   cases = load_eval_set("path/to/hand-labeled/set.jsonl")
   results = [triage(c["transaction"], c["classifier_score"]) for c in cases]
   deterministic_scores = [deterministic_score(c, r) for c, r in zip(cases, results)]
   judge_scores = [llm_judge_score(r) for r in results]
   rep = report(cases, results, deterministic_scores, judge_scores)
   ```

   Save the result to `aml-triage/reports/phase3_triage_eval.json` via `json.dump(rep, open(...), indent=2)`. This pipeline is not asserted against fixed expected values (the judge will give different answers on different days, and on different models); it is a one-time run to populate the final deliverable metric.

```python
def report(cases, results, deterministic_scores, judge_scores):
    lengths = {len(cases), len(results), len(deterministic_scores), len(judge_scores)}
    if len(lengths) != 1:
        raise ValueError(f"mismatched list lengths: {lengths}")

    n = len(cases)
    if n == 0:
        raise ValueError("no cases to report on")
    return {
        "n_cases": n,
        "decision_agreement_rate": sum(d["decision_match"] for d in deterministic_scores) / n,
        "citation_present_rate": sum(d["citation_present"] for d in deterministic_scores) / n,
        "judge_agreement_rate": sum(j["agrees"] for j in judge_scores) / n,
    }
```

## Checks

Ask the learner to answer these in their own words before moving on:

- Why does `report()` compute three separate rates rather than one combined score? What does a low `citation_present_rate` tell you that a low `decision_agreement_rate` alone would not?
- If `judge_agreement_rate` is low (0.3) but `decision_agreement_rate` is high (0.9), what does that pattern suggest is wrong upstream — your model's judgment, its explanation, or its grounding to the typologies?

Then run the baked-in check:

```sh
uv run pytest ../aml-tutor/tests/015_test_agreement_report.py -v
```

Both tests must pass:

- `test_report_computes_the_three_rates`: confirms that `report()` correctly aggregates each rate as a fraction of the eval set.
- `test_report_raises_on_mismatched_list_lengths`: confirms that mismatched list lengths raise `ValueError`.

```json validation
[
  {
    "id": "015-the-agreement-rate-report",
    "label": "The agreement-rate report",
    "command": "uv",
    "args": ["run", "pytest", "../aml-tutor/tests/015_test_agreement_report.py", "-v"],
    "cwd": "../aml-triage"
  }
]
```

## Pressure test

An aggregated report can hide failures. If your `judge_agreement_rate` is 0.7, that's useful, but it could mean the judge agrees on 70% of cases uniformly — or it could mean the judge is confident on easy cases (high agreement) and uncertain on edge cases (low agreement). The real diagnostic work starts after this lesson: read a sample of your results, especially the disagreements and the low-confidence cases, and ask whether the agent's reasoning is sound or whether you're seeing a systematic misunderstanding of a particular typology or a particular transaction pattern.

This report is the *start* of the validation conversation, not the end.

## Closing note — end of Parts 2 and 3

The project described in `aml-triage/SEED.md` is now complete end to end:

- **Phase 1 (Part 1, lessons 001–007):** A classifier with a documented operating point (`reports/phase1_report.json`, lesson 007) — the baseline XGBoost model, threshold-tuned for precision.
- **Phase 3 (Parts 2–3, lessons 008–015):** A triage agent with a documented, hand-labeled agreement rate (`reports/phase3_triage_eval.json`, this lesson) — the end-to-end agent that retrieves typologies, reasons over them, and decides whether each transaction warrants escalation.

Both follow the shape that SEED.md's Shipping section asks the eventual README/write-up to lead with: "problem → method → measured result." Your classifier solves the problem of detecting anomalies; your triage agent solves the problem of explaining them. Each now has a measured result — a number you can cite.

**Note:** 16 hand-labeled cases (or whatever your eval set size) is a *floor*, not a ceiling or a target. SEED.md calls for 30–50 cases in a full production deployment; your eval set here is proof of concept, sufficient to validate the pipeline works end to end. A real model-validation document would expand this to dozens or hundreds of cases, stratified by decision type and transaction family, and reviewed by SMEs who know the typologies.

## Doer fallback note

The `report()` function is pure aggregation — no external tools, no model calls, no shell commands needed. If asked to write this step for you, the doer can write `report()` by hand directly into your `src/aml_triage/eval.py`, using the reference implementation above. Running the full pipeline to populate `phase3_triage_eval.json` requires your own `ANTHROPIC_API_KEY` and network access, same boundary as lesson 011; that step is outside the doer's scope.
