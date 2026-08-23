# The end-to-end triage agent

Wire the lessons together: retrieval (008/009) → prompt + schema (010) → the forced tool call → parsing
(010). This is the first lesson where the `aml_triage` code calls a real LLM. The baked-in test injects
a fake client and never calls the real Anthropic API — though with the default `alpha=0.5`, it still
runs through lesson 009's `top_k_typologies_hybrid`, so the embedding model lookup can still touch the
network on a cold cache (the same one-time exception lesson 009 documents). Production callers can inject a custom client, or omit the
parameter and get a real `anthropic.Anthropic()` — the pattern that answers the interview question
"how do you test code that calls an LLM?"

## Key concept

**The LLM client as an injected, testable seam.** `triage(...)` takes a `client=None` parameter and only
constructs a real `anthropic.Anthropic()` when nothing was supplied. This is the concrete pattern for
testing LLM features: dependency injection at the boundary. The test passes a fake client that returns
predetermined outputs; production code passes nothing and gets a live client. Same interface, different
behavior — no mocking framework needed, no test-doubles hidden behind a global registry. The client
becomes a seam that you can swap without changing the code.

**At a regulated shop:** this call would go through Bedrock or Azure OpenAI rather than a public API
directly, for the enterprise DPA and VPC boundary — and its output would be a draft recommendation
routed to a human analyst queue, never an auto-executed decision. Human-in-the-loop is close to a hard
requirement at the compliance and risk level: the model makes a suggestion (escalate/monitor/close), but
a human analyst reviews it before any action is taken. That review flow — the queue, the UI, the
audit trail of approvals — lives outside this lesson, but the triage decision itself is built to feed
into that queue.

## Implementation order

1. **Add the `anthropic` dependency.** Run `uv add anthropic` — this is the Anthropic SDK; it provides
   the `anthropic.Anthropic()` client that `triage()` will construct when no client is injected.

2. **Implement `triage(transaction, classifier_score, *, client=None, k=3, corpus_path=None, alpha=0.5)
   -> dict`** in `src/aml_triage/triage.py` (a new file). The function wires retrieval → prompt → tool call
   → parsing:

   - If `client is None`, construct a real `anthropic.Anthropic()`.
   - Call `top_k_typologies_hybrid(...)` to retrieve relevant typologies. Use the transaction's `type`
     and `amount` to build the query: `f"{transaction['type']} transaction of amount {transaction['amount']}"`.
     Pass `k`, `corpus_path`, and `alpha` straight through to the retriever.
   - Call `build_prompt(...)` to construct the system message.
   - Call `client.messages.create(...)` with:
     - `model`: read from a `TRIAGE_MODEL` env var (mirroring the `aml-tutor` convention), defaulting
       to a small, fast model: `"claude-haiku-4-5-20251001"`.
     - `max_tokens`: 1024.
     - `tools`: `[TRIAGE_TOOL_SCHEMA]`.
     - `tool_choice`: force the tool: `{"type": "tool", "name": TRIAGE_TOOL_SCHEMA["name"]}`.
     - `messages`: a single user message with the prompt string.
   - Extract the tool-use block from the response and pass its `input` to `parse_triage_decision(...)`,
     along with the set of IDs from the retrieved typologies.
   - **Add a `"retrieved"` key to the result dict before returning it.** Make a shallow copy of what
     `parse_triage_decision` returned before adding the key — `result = dict(parse_triage_decision(...))`
     — rather than mutating that dict in place. This carries forward the list of typologies the model
     was shown — lesson 014 needs it later to build its own prompt, and lesson 015's pipeline needs it
     to reproduce the exact conditions a given result was judged under.

### If you ask the tutor to do this step for you

The tutor cannot run `uv add anthropic` (no shell access). Once `anthropic` is installed in your
environment, the tutor can write `src/aml_triage/triage.py` by hand using this spec's signatures and
the wiring order described above.

## Checks

Ask the learner to answer these in their own words before moving on:

- Why is `client=None` with conditional construction (`if client is None: import anthropic; client =
  anthropic.Anthropic()`) better than constructing the client unconditionally at the top of the function?
- The test injects a fake client that returns predetermined outputs. Why does that test not need
  `ANTHROPIC_API_KEY` set, and why does it never call the real Anthropic API?
- What would change if the model hallucinated a citation to a typology ID that exists in your corpus but
  was never included in the `retrieved` list passed to `build_prompt`? How does `parse_triage_decision`
  catch that?

Then run the baked-in check:

```sh
uv run pytest ../aml-tutor/tests/011_test_triage_agent.py -v
```

Both tests must pass:

- `test_triage_forces_the_structured_tool_and_returns_the_parsed_decision`: confirms that `triage()` calls
  `client.messages.create()` with the correct tool schema and tool_choice, constructs the right prompt
  from the transaction and retrieved typologies, and returns a result dict that includes the `"retrieved"`
  key alongside `decision`, `rationale`, and `cited_typology_ids`.
- `test_triage_raises_when_the_model_cites_something_it_was_not_shown`: confirms that `parse_triage_decision`
  rejects citations to IDs not in the retrieved list.

```json validation
[
  {
    "id": "011-the-triage-agent",
    "label": "The end-to-end triage agent",
    "command": "uv",
    "args": ["run", "pytest", "../aml-tutor/tests/011_test_triage_agent.py", "-v"],
    "cwd": "../aml-triage"
  }
]
```

## Pressure test

Today's test is a baked-in unit test with a fake client — it proves the wiring and the citation guard,
not that a real Claude call produces sensible triage decisions. As a separate, manual (non-pytest)
step, you should:

1. Export `ANTHROPIC_API_KEY` with your actual key.
2. Pick one transaction the Phase 1 classifier actually flagged (from `paysim_fixture.csv`).
3. Call `triage(...)` for real against that transaction, omitting `client=` so you get a live client.
4. Read the result and see what the model decided: escalate, monitor, or close? Does the rationale
   make sense? Do the citations match typologies that actually exist in your corpus?

This manual run is not graded, but it builds intuition: you see what a real decision looks like, and
you calibrate your understanding of whether the model is being useful or hallucinating.
