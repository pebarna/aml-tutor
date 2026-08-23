# Structured triage decisions

Build a prompt that presents a transaction and its retrieved typologies to a language model, then
define a schema for the model to respond with a structured triage decision. The key insight: force
structured output at the model boundary using a tool schema, then validate that all citations
refer only to typologies the model was actually shown.

## Key concept

**Forcing structured output at the model boundary — not parsing free text — is what a model-risk
review requires.** Lesson 008 and 009 retrieved typologies that might support a triage decision.
This lesson builds the prompt that shows those typologies to an LLM, but it does not call the LLM
yet (that is lesson 011). Instead, it defines a tool schema — a JSON contract — that tells the
model exactly what fields and value types it must return: a decision (one of `"escalate"`,
`"monitor"`, or `"close"`), a rationale (free text), and a list of typology IDs it cites from the
list shown to it.

**`known_typology_ids` means the IDs actually retrieved and shown for this call, not the whole
corpus.** This is crucial: a model might hallucinate a citation to a real typology that exists in
your database but was never shown to the model on this particular call. That hallucination gets
rejected the same way a completely fabricated ID does — both fail the `parse_triage_decision` guard.
The model sees exactly six typologies (in the tutorial) or perhaps hundreds (in a real system), and
its citations are validated only against what it was shown.

**At a regulated shop:** forced structured output at the model boundary — validated by deterministic
rules in code — is exactly what a model-risk review requires. Rather than trusting an LLM to parse
its own free text (brittle, hard to audit), you give it a JSON schema and tell it to respond with
one. The schema becomes part of the audit trail. The validation code (rejecting unknown citations)
is deterministic and testable. The full decision — decision + rationale + validated citations —
feeds downstream systems (escalation queues, monitoring dashboards) with confidence that the model
cannot bypass the schema or sneak in unvetted typology IDs.

## Implementation order

1. **Implement `TRIAGE_TOOL_SCHEMA`** in `src/aml_triage/triage_schema.py` — a `dict` shaped like an
   Anthropic tool definition, with a `"submit_triage_decision"` tool that has three input fields:
   - `decision`: an enum string, one of `["escalate", "monitor", "close"]`
   - `rationale`: a free-text string explaining the decision
   - `cited_typology_ids`: an array of strings (IDs from the corpus)

   All three fields are required. The schema is a plain dict; lesson 011 will pass it to the Claude
   API.

2. **Implement `build_prompt(transaction, classifier_score, retrieved) -> str`** in the same file —
   a pure function that takes:
   - `transaction`: a dict with keys like `type`, `amount`, `step` (from the transaction being triaged)
   - `classifier_score`: a float between 0 and 1 (the output of lesson 007's baseline classifier)
   - `retrieved`: a list of dicts, each shaped `{"id", "title", "text", "score"}` (from lesson 009's
     hybrid retriever)

   and returns a prompt string that includes all four pieces of data and tells the model to cite only
   typology IDs from the retrieved list. The exact wording is up to you, but the prompt must make
   clear that citations must be drawn from what was shown.

3. **Implement `parse_triage_decision(tool_input, known_typology_ids) -> dict`** in the same file —
   a pure function that takes:
   - `tool_input`: the dict the model returned (with `decision`, `rationale`, `cited_typology_ids`)
   - `known_typology_ids`: a set of strings (the IDs that were actually shown to the model in
     `build_prompt`)

   and validates that the decision is one of the three allowed values and that every cited ID is
   in `known_typology_ids`. If validation passes, return the `tool_input` dict unchanged. If it
   fails, raise a `ValueError` with a clear message:
   - Invalid decision → "invalid decision: ..." with the bad value quoted
   - Unknown citations → "cited typology ids not shown to the model: ..." with the unknown IDs
     listed

   This is the guard that stops a model from citing a real typology it was never shown, or a
   completely fabricated ID, or any typo thereof.

4. **Do not commit scratch code.** The reference implementation lives in
   `/tmp/aml-tutor-plan003-scratch/src/aml_triage/triage_schema.py`. Only the spec and the
   baked-in test in this tutorial's git history. The learner writes their own `triage_schema.py`
   inside their `aml-triage` repo, which is never committed to this one.

### If you ask the tutor to do this step for you

No shell command is needed — this lesson has no new dependencies and makes no network calls. All
three functions are pure Python: dict construction for the tool schema, string formatting for the
prompt, and set intersection for validation. A doer can write this entire file by hand,
deterministically, using only this spec's signatures and logic.

## Checks

Ask the learner to answer these in their own words before moving to lesson 011:

- What would happen if a model cited a typology ID that is real and exists in your corpus, but was
  never included in the `retrieved` list passed to `build_prompt`? Why is that as bad as a
  fabricated ID?
- `build_prompt` constructs a user-facing message that tells the model which typologies exist for
  this transaction. Why must `parse_triage_decision` validate that citations match exactly what was
  shown — rather than trusting the model to follow instructions?
- If you lowered the bar for what counts as a valid decision (e.g., allowing `"ignore"` as well as
  the three current values), what would that mean for downstream systems that expect exactly three
  states?

Then run the baked-in check:

```sh
uv run pytest ../aml-tutor/tests/010_test_structured_decisions.py -v
```

All five tests must pass:

- `test_prompt_includes_transaction_score_and_retrieved_titles`: confirms the prompt contains all
  four key pieces of data (transaction type, amount, classifier score, and retrieved typology titles).
- `test_tool_schema_names_a_forced_tool_with_the_expected_fields`: confirms the schema has the
  right fields with the right enum values for decision.
- `test_parse_returns_a_clean_dict_for_a_valid_tool_call`: confirms that valid input passes through
  unchanged.
- `test_parse_rejects_an_invalid_decision_enum`: confirms that an invalid decision value raises
  `ValueError`.
- `test_parse_rejects_a_citation_the_model_was_not_shown`: confirms that citing an ID not in
  `known_typology_ids` raises `ValueError` — the key guard.

```json validation
[
  {
    "id": "010-structured-triage-decisions",
    "label": "Structured triage decisions",
    "command": "uv",
    "args": ["run", "pytest", "../aml-tutor/tests/010_test_structured_decisions.py", "-v"],
    "cwd": "../aml-triage"
  }
]
```

## Pressure test

Today's tests are pure unit tests: they check `build_prompt` and `parse_triage_decision` in
isolation over fixed test data, not against a real model API call. Lesson 011 is the first to
call an actual LLM with this schema. If a model's response contains a citation to an ID not in
`known_typology_ids` — either a real typology from the corpus or a hallucinated one — lesson 011's
test will catch it via `parse_triage_decision` and fail loudly. That failure is a feature, not a
bug: it proves the guard is working. A model that obeys the tool schema and cites only IDs it was
shown will pass; one that does not will fail, surfacing the model's failure to follow the forced-
output contract at the boundary.
