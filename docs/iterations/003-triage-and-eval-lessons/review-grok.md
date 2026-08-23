# Consistency & flow review — iteration 003 plan

Reviewer: grok (cold read). Scope: plan-document consistency only — signatures, numbering, intra-task alignment, front-matter vs body, forward references, and cwd/path resolution. Not a review of technical merit.

---

## 1. Signature consistency

1. **[High] Task 10 Step 2 / ARCHITECTURE.md table abbreviates lesson 010 parameter names.**  
   Global Constraints, Task 4 Interfaces, Task 4 code, and Task 4/5 call sites all use:
   - `build_prompt(transaction, classifier_score, retrieved) -> str`
   - `parse_triage_decision(tool_input, known_typology_ids) -> dict`  
   Task 10’s table snippet instead writes:
   - `(transaction, score, retrieved) -> str`
   - `(tool_input, known_ids) -> dict`  
   `score` ≠ `classifier_score`, `known_ids` ≠ `known_typology_ids`. An implementer copying the final ARCHITECTURE row would ship a different public API than every earlier task pinned.

2. **[Medium] Task 8 — `case` is in every `llm_judge_score` signature appearance but never used.**  
   Global Constraints, Task 8 Interfaces, the def, the tests, the spec Implementation order, and the Task 10 ARCHITECTURE row all include `case` as the first parameter. The scratch body and `_judge_prompt(case, result, retrieved)` ignore `case` entirely; the tests never assert that label fields appear in the judge prompt; the Key concept says the judge scores rationale-vs-cited-typology-text, not agreement with the human label. Either the implementation/spec should consume `case` (e.g. include `label_decision` / `label_note` in the judge prompt) or `case` should be dropped from the public signature everywhere. As written, the plan consistently advertises a dead parameter.

3. **[Low] Task 10 ARCHITECTURE row for 010 omits `TRIAGE_TOOL_SCHEMA`.**  
   Task 4 Interfaces explicitly produce three exports (`TRIAGE_TOOL_SCHEMA`, `build_prompt`, `parse_triage_decision`), and Task 5’s test asserts on `TRIAGE_TOOL_SCHEMA` by name. The §6 table only lists the two functions. Precedent in §6 is “one primary function per lesson,” so this may be intentional compression, but it is incomplete relative to Task 4’s own Interfaces block.

4. **[Low] Task 7 Interfaces annotates `deterministic_score(case: dict, result: dict)` while Global Constraints, code, and ARCHITECTURE use bare `(case, result)`.**  
   Cosmetic only; names and arity match.

**Otherwise signatures match** across Global Constraints, per-task Interfaces, scratch defs, later callers, and (except finding 1) the Task 10 table for:
`top_k_typologies`, `top_k_typologies_hybrid`, `triage`, `load_eval_set`, `deterministic_score`, `llm_judge_score`, `report`.

---

## 2. Numbering consistency

**No issues found.**

- Task headers 008–015 line up with `docs/specs/NNN-*.md` and `tests/NNN_test_*.py` filenames.
- Validation-block `id` values match the spec slugs (`008-typology-retrieval` … `015-the-agreement-rate-report`).
- Task 10 Part grouping (008–011 → Part 2, 012–015 → Part 3) matches the Goal / Why-these-choices phase split.
- Ledger “Add row NNN as Done” steps use the correct lesson numbers throughout.

---

## 3. Internal consistency within each Task

5. **[High] Task 4 Step 4 expects “6 passed” but the test file defines 5 tests.**  
   Listed tests:  
   `test_prompt_includes_transaction_score_and_retrieved_titles`,  
   `test_tool_schema_names_a_forced_tool_with_the_expected_fields`,  
   `test_parse_returns_a_clean_dict_for_a_valid_tool_call`,  
   `test_parse_rejects_an_invalid_decision_enum`,  
   `test_parse_rejects_a_citation_the_model_was_not_shown`.  
   Step 4: “Expected: 6 passed.” An implementer who sees 5 passed will think the scratch impl is incomplete. Fix the expected count (or add the missing sixth test if one was drafted away).

6. **[Medium] Task 8 — spec bullets vs code/tests disagree on what the judge is given.**  
   Implementation order lists `llm_judge_score(case, result, retrieved, *, client=None)`.  
   Key concept / Checks focus on cited typology text supporting the rationale.  
   Code and `_judge_prompt` never read `case`; the second test only asserts rationale text and typology title appear in the prompt. The spec-writing step does not tell the author to put label fields into the judge prompt, so a faithful spec will describe a `case` parameter the reference code does not use (same root cause as finding 2, surfaced inside the task).

7. **[Medium] Task 2 / Task 5 vs Global Constraints — “aml-triage’s own root” vs cwd-relative open.**  
   Global Constraints and Task 2 Implementation order say `corpus_path=None` resolves to `"data/typologies.json"` **relative to aml-triage’s own root**. Scratch `_load_corpus` does `open(corpus_path or DEFAULT_CORPUS_PATH)` — i.e. **process cwd**, not package/repo root via `__file__`. That is fine when validation always uses `cwd=../aml-triage`, but the plan’s wording over-claims root resolution. Either change the wording to “relative to process cwd (expected to be aml-triage root)” or make the reference impl resolve against a fixed root.

8. **[Low] Task 8 scratch appends `import os` mid-module** after `load_eval_set` / `deterministic_score` were already written in Tasks 6–7. Valid Python, but unlike Task 5’s `triage.py` (imports at top) and unlike a clean single-module authoring story the eventual student spec will want. Spec Implementation order should say to keep imports at the top of `eval.py`.

9. **[Low] Task 4 `test_parse_returns_a_clean_dict_for_a_valid_tool_call` asserts `result == tool_input`.**  
   Implementation always returns a freshly built three-key dict. Equal only because the test’s `tool_input` has exactly those keys; a fourth key would make a correct parser fail the assertion. Minor test/impl brittleness inside the same task.

**Tasks 1, 2 (except wording in 7), 3, 5, 6, 7, 9: intra-task spec bullets ↔ tests ↔ scratch code align** on shapes, keys, and behavior described.

---

## 4. Consistency between front-matter and body

10. **[High] Architecture blurb incorrectly groups lesson 010 with LLM-calling, client-injected lessons.**  
    Opening Architecture §1:  
    > “Lessons 010 (structured decisions) and 011 (the agent) and 014 (LLM-as-judge) call a real LLM in production use; every one of them takes its LLM client as an injectable parameter…”  
    Task 4’s own body contradicts this: 010 is pure functions, no SDK import, no `client=` parameter, no network. Why-these-choices correctly limits DI to “lessons 011/014.” The Architecture sentence reads like a leftover from an earlier draft that folded schema+agent together. Should be 011 and 014 only (010 prepares the schema those lessons call).

11. **[Medium] Definition of done omits lesson 012 from the “At a regulated shop” checklist while Global Constraints and Task 6 require it.**  
    Global Constraints: every new spec gets an “At a regulated shop” paragraph.  
    Task 6 Step 6 includes that paragraph for 012.  
    Definition of done: “Every relevant spec (008, 009, 010, 011, 013, 014, 015)” — skips 012. DoD should either include 012 or explain why 012 is exempt; right now front-matter and body disagree.

12. **[Medium] Definition of done over-claims that this iteration “produces” the hand-labeled eval set.**  
    > “Lesson 012 produces a real, hand-labeled `aml-triage/data/triage_eval_set.jsonl`…”  
    Task 6 explicitly does **not** commit labels; it only builds `load_eval_set`, a structure-only test, and a throwaway stand-in for plan verification. The labeled file is a **student** deliverable when they take the lesson. Wording should match Task 6 / Why-these-choices (“structure only; judgment in conversation”), e.g. “Lesson 012’s spec and test require the student to produce…”.

13. **[Low] ARCHITECTURE.md title/header still framed as “Iteration 001”.**  
    Task 0 deletes “Known gap”; Task 10 extends §6. Neither step updates the doc title (`# Architecture contract — Iteration 001`) or the “Pinned before Phase A…” line. Stale framing after this iteration lands — small leftover relative to the plan’s own “close the gap / extend the table” story.

14. **[Low] Why-these-choices / Open risks scope notes are consistent with the eight-lesson body.**  
    No further front-matter leftovers found beyond 10–13. Tech stack (`sentence-transformers` @ 009, `anthropic` @ 011), scratch path `/tmp/aml-tutor-plan003-scratch/`, and “no new engine capability” all match the tasks.

---

## 5. Narrative flow / forward references

15. **[Medium] Scratch package never creates `aml_triage/__init__.py`.**  
    Iteration 002 Task 2 explicitly created `/tmp/aml-tutor-plan002-scratch/src/aml_triage/__init__.py`. Iteration 003 never does. Tasks 2–4 may limp along as a namespace package, but Task 5’s `triage.py` uses **relative** imports (`from .retrieval import …`, `from .triage_schema import …`), which are fragile without a real package. First failure may appear only at Task 5 Step 4, not when retrieval alone is tested. Add an early step (Task 1 or Task 2) that creates the empty `__init__.py`, matching 002.

16. **[Low] Task 1 Step 4/5 bash commands do not state cwd.**  
    `python3 scripts/build_eval_candidates.py` and the spot-check assume `aml-tutor` root (script paths are relative). Iteration 002 Task 1 said “from aml-tutor’s root” explicitly. Minor footgun for an agentic worker that drifted directories.

**Forward-reference check otherwise clean:**  
each task only consumes fixtures/functions earlier tasks produce (typologies + candidates → 008/009 → schema 010 → triage 011 → eval set shape 012 → deterministic 013 → judge 014 → report 015). Pressure-test mentions of later lessons (008→010, 010→011, 012→013, 013→014, 014→015) are forward-looking narrative, not code dependencies. Task 5’s test calling `top_k_typologies_hybrid` is valid (Task 3). Task 6’s optional “call your own `triage(...)` while labeling” is valid (Task 5).

---

## 6. cwd / path-resolution consistency (esp. lesson 012)

17. **[Medium] Global Constraints over-promise “no test result depends on the process’s cwd” then lesson 012 depends entirely on cwd.**  
    Constraints: every baked-in test passes `corpus_path` explicitly, “so no test result depends on the process’s cwd.”  
    Task 6 hardcodes `load_eval_set("data/triage_eval_set.jsonl")` with no path injection; the plan correctly calls 012 the first lesson whose test depends on cwd and verifies via `cd /tmp/.../aml-triage`. The corpus_path sentence is true for retrieval tests only; as a Global Constraint it is overstated and will confuse anyone applying it to 012. Narrow the constraint to retrieval/`corpus_path`, and name 012 as the deliberate cwd-dependent exception next to it.

18. **[Low] Task 10 Step 4 batch run is path-consistent for the lessons it includes, and correctly excludes 012.**  
    Run from `aml-tutor` root with `PYTHONPATH` to scratch; 008–011 pass `corpus_path=str(typologies_path)`; 013–015 use in-memory fixtures only. 012 excluded with an explicit reason. Good.

19. **[Low] Task 6 verification paths are internally consistent.**  
    Stand-in file written to `/tmp/aml-tutor-plan003-scratch/aml-triage/data/triage_eval_set.jsonl`, pytest invoked after `cd` to that aml-triage directory, test path absolute to `aml-tutor/tests/012_test_eval_set.py` so `conftest.py` / `eval_candidates_path` still resolve via `Path(__file__)`. Matches what real validation (`cwd=../aml-triage`, test path `../aml-tutor/tests/012_…`) will do for the relative `"data/triage_eval_set.jsonl"` open.

20. **[Low] Real validation `cwd=../aml-triage` vs scratch verification using `/tmp/.../aml-triage` is intentional and documented.**  
    No command in the plan runs 012’s test from `aml-tutor` root without a prior `cd`, and no command runs retrieval tests without either `corpus_path=` or an implied aml-triage cwd. OK.

**Path/cwd summary:** lesson 012’s own steps are careful; the main defect is Global Constraints claiming cwd-independence globally (finding 17), plus the “repo root” vs cwd wording for default `corpus_path` (finding 7).

---

## Category totals

| Category | High | Medium | Low | Clean? |
|---|---|---|---|---|
| 1. Signature consistency | 1 | 1 | 2 | No |
| 2. Numbering consistency | 0 | 0 | 0 | **Yes — zero issues** |
| 3. Internal consistency within each Task | 1 | 2 | 2 | No |
| 4. Front-matter vs body | 1 | 2 | 2 | No |
| 5. Narrative flow / forward references | 0 | 1 | 1 | Mostly clean |
| 6. cwd / path-resolution | 0 | 1 | 3 | Mostly clean |

---

## Highest-priority fixes (if only a few land before execution)

1. Task 10 ARCHITECTURE row: use `classifier_score` and `known_typology_ids` (finding 1).  
2. Task 4 Step 4: “Expected: 5 passed” (finding 5).  
3. Architecture intro: drop lesson 010 from the “calls a real LLM / injectable client” list (finding 10).  
4. Decide fate of `case` on `llm_judge_score` and make signature, impl, tests, and spec agree (findings 2 & 6).  
5. Create scratch `__init__.py` early (finding 15).  
6. Tighten Global Constraints cwd/`corpus_path` wording and DoD’s regulated-shop + hand-label bullets (findings 7, 11, 12, 17).
