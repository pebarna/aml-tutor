# LLM-as-judge scoring

Once you have a triage agent (lesson 011) and deterministic checks (lesson 013) confirming the agent
cited something and matched your decision, you can ask a harder question: does the agent's *rationale*
actually *follow* from the *text* of the typologies it cited?

A model might cite `TY-001` (Structuring) but claim the transaction "involves a single large
withdrawal" — which doesn't follow from any part of the structuring definition. Deterministic checks
won't catch this. Lesson 014's LLM-as-judge does.

## Key concept

The judge is a separate, lightweight scoring model that reads:

- The agent's decision and rationale (from `result["decision"]` and `result["rationale"]`)
- The typology IDs the agent cited (from `result["cited_typology_ids"]`)
- The *text* of those typologies (from `result["retrieved"]`, the list of retrieved objects that
  `triage()` already carries from lesson 011)

And asks: does the rationale plausibly follow from the text? It returns two fields: `agrees`
(boolean) and `comment` (string).

Like lesson 011's `triage()`, the judge uses client injection for testability — the same pattern,
same reason. The judge takes `result` alone; it does *not* take a separate `case` parameter,
because there is no human-label comparison happening here. Lesson 013's `deterministic_score` reads
the human label to compare the agent's decision against it; lesson 014's judge reads the typology
text to validate the reasoning. These are different checks, hence different signatures.

**At a regulated shop:** LLM-as-judge is used as a fast, scalable *supplementary* signal in ongoing
monitoring — never a replacement for lesson 012's SME-labeled golden set, which is what a SR 11-7
model-risk review actually requires. The judge might agree that a rationale follows from a cited
typology; that does not mean the typology itself applies to the transaction, or that the model's
decision is correct. Supplementary, not foundational.

## Implementation order

Implement `JUDGE_TOOL_SCHEMA`, `_judge_prompt(result)`, and `llm_judge_score(result, *, client=None)
-> dict` in `src/aml_triage/eval.py`. The function takes:

- `result`: a return value from `triage(...)` (lesson 011's shape, including the `retrieved` key)

And returns a dict with two keys:

- `agrees`: a boolean indicating whether the rationale follows from the cited typology text
- `comment`: a one-sentence string explaining the verdict

Do *not* add a separate `retrieved` parameter to the function signature. The `result` already carries
everything this function needs via the `"retrieved"` key that lesson 011's `triage()` always includes.
There is no separate way to reconstruct that list, and there is no human-label comparison in this
function's job.

```python
JUDGE_TOOL_SCHEMA = {
    "name": "submit_judge_verdict",
    "description": "Submit a verdict on whether a triage rationale is supported by its cited typologies.",
    "input_schema": {
        "type": "object",
        "properties": {
            "agrees": {"type": "boolean"},
            "comment": {"type": "string"},
        },
        "required": ["agrees", "comment"],
    },
}


def _judge_prompt(result):
    retrieved = result["retrieved"]
    typology_lines = "\n".join(f"- {t['id']} ({t['title']}): {t['text']}" for t in retrieved)
    return (
        f"A triage agent decided '{result['decision']}' with rationale: {result['rationale']}\n"
        f"It cited: {result['cited_typology_ids']}.\n"
        f"The typologies it had available:\n{typology_lines}\n\n"
        "Does the rationale plausibly follow from the cited typology text? Answer with a verdict "
        "and a one-sentence comment."
    )


def llm_judge_score(result, *, client=None):
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    response = client.messages.create(
        model=os.environ.get("TRIAGE_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=512,
        tools=[JUDGE_TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": JUDGE_TOOL_SCHEMA["name"]},
        messages=[{"role": "user", "content": _judge_prompt(result)}],
    )
    tool_use = next(block for block in response.content if block.type == "tool_use")
    return {"agrees": tool_use.input["agrees"], "comment": tool_use.input["comment"]}
```

## Checks

Comprehension questions, then the baked-in test:

- Why does the judge read `result["retrieved"]` — the list of typologies shown to the model — rather
  than comparing the agent's decision to a human label? What is the judge actually checking?
- If a judge agrees that the rationale follows from the text, does that mean the model's decision is
  correct? Why or why not?
- Why is there no separate `retrieved` parameter in the `llm_judge_score` signature? What makes it
  safe to read it from `result["retrieved"]` instead?

Run the baked-in check:

```sh
uv run pytest ../aml-tutor/tests/014_test_llm_judge.py -v
```

All two tests must pass:

- `test_judge_forces_a_structured_tool_and_returns_its_verdict`: confirms the judge calls the model
  with the tool forced and parses the returned verdict correctly.
- `test_judge_prompt_includes_the_rationale_and_cited_typology_text`: confirms the prompt includes
  both the agent's rationale and the text of the cited typologies.

```json validation
[
  {
    "id": "014-llm-as-judge",
    "label": "LLM-as-judge scoring",
    "command": "uv",
    "args": ["run", "pytest", "../aml-tutor/tests/014_test_llm_judge.py", "-v"],
    "cwd": "../aml-triage"
  }
]
```

## Pressure test

A judge model, like any model, can be wrong. This lesson's test proves wiring and parsing, not judge
accuracy. The judge might say a rationale follows from a typology when it actually doesn't, or vice
versa. Lesson 015 will aggregate the judge's verdicts across your eval set into a `judge_agreement_rate`,
which is a signal — one of several — to read alongside the deterministic numbers and lesson 012's
human labels, not a ground truth that overrides either.

Imagine an agent that systematically misunderstands a particular typology, always citing it when the
transaction is actually a different pattern. The judge might agree that the rationale follows — and
the judge would be wrong. That's when you pull the human labels, compare them to the agent's decisions,
and ask: is the model's judgment sound, or is it gaming the system? That conversation can only happen
with a labeled eval set and a skeptical eye on a sample of results. The judge accelerates the
screening, but it doesn't replace human review.

## Doer fallback note

The `anthropic` dependency is already installed (lesson 011). If asked to write this step for you,
the doer can write `llm_judge_score` by hand directly into your `src/aml_triage/eval.py`, using the
reference implementation above.
