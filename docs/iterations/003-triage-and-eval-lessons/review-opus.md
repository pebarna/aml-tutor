# Ensemble review — iteration 003 plan (reviewer: opus)

Scope: consistency and flow review of `docs/iterations/003-triage-and-eval-lessons/plan.md` only —
not a judgement on the technical/architectural choices. Precedent read: iteration 002's plan,
`docs/ARCHITECTURE.md`, `docs/specs/007-evaluation-and-threshold.md`. Where a finding was checkable
against the repo, I checked it (noted inline).

**Totals: 25 findings — 2 High, 9 Medium, 14 Low.**

---

## 1. Signature consistency

Clean for the majority: `top_k_typologies`, `top_k_typologies_hybrid`, `triage`, `load_eval_set`,
`deterministic_score` and `report` are byte-identical (names, order, defaults, keyword-only markers)
across Global Constraints, each Task's Interfaces block, the reference implementations, later tasks
that call them, and Task 10 Step 2's ARCHITECTURE.md table snippet. Findings below are the
exceptions.

1. **ARCHITECTURE.md table renames lesson 010's parameters — and the baked-in test uses keyword
   args.** Task 10 Step 2 writes
   `| 010 | ... | (transaction, score, retrieved) -> str | (tool_input, known_ids) -> dict |`,
   but every other appearance says `build_prompt(transaction, classifier_score, retrieved)` and
   `parse_triage_decision(tool_input, known_typology_ids)` (Global Constraints lines 42–43, Task 4
   Interfaces, Task 4 Step 3 code, Task 4 Step 5 spec bullets). This is not cosmetic: Task 4's test
   calls `build_prompt(TRANSACTION, classifier_score=0.87, retrieved=RETRIEVED)` and
   `parse_triage_decision(tool_input, known_typology_ids={...})` **by keyword**, so a student who
   implements to the names in the §6 table — the table the plan itself calls "the lesson-to-code
   API" — gets `TypeError: got an unexpected keyword argument`. **Medium.**

2. **`llm_judge_score`'s `case` parameter is dead in the reference implementation.** The signature
   `llm_judge_score(case, result, retrieved, *, client=None)` appears in Global Constraints, Task 8
   Interfaces, the ARCH table row 014 and the test — but Task 8 Step 3's `_judge_prompt(case, result,
   retrieved)` never references `case`, and the return value is built entirely from the tool call.
   Nothing in Task 8's spec bullets explains the (defensible) design that the judge deliberately does
   not see `label_decision`, so a spec author working from these bullets can't tell whether the
   omission is intentional or a bug. **Medium.**
   *Same family, Low:* Task 9's `report(cases, results, deterministic_scores, judge_scores)` uses
   `cases` only for `n` and `results` only for the length check — the four-argument signature is
   wider than the computation needs, and no bullet says why.

3. **Task 8's Interfaces block omits the constants its own spec bullet requires.** Task 4 lists
   `TRIAGE_TOOL_SCHEMA` in Interfaces ("lesson 011 imports all three by name"); Task 8's Interfaces
   list only `llm_judge_score`, even though Step 5's Implementation order says to implement
   `JUDGE_TOOL_SCHEMA` and `_judge_prompt`, and the test asserts the literal tool name
   `submit_judge_verdict`. Neither tool-schema constant appears in Global Constraints' API list.
   **Low.**

4. **Model-id / env-var handling diverges between lessons 011 and 014 without comment.** Task 5 uses
   `DEFAULT_MODEL = "claude-haiku-4-5-20251001"` plus `os.environ.get("TRIAGE_MODEL", DEFAULT_MODEL)`;
   Task 8 inlines the same literal (`os.environ.get("TRIAGE_MODEL", "claude-haiku-4-5-20251001")`)
   with no constant. Lesson 011's spec bullets document the `TRIAGE_MODEL` convention; lesson 014's
   never mention that the judge also reads it (or that a judge sharing the *triage* model var is
   deliberate). The "Model naming drift" open risk names only lesson 011's `DEFAULT_MODEL`. **Low.**

5. **Two same-named `report` functions in near-identically named modules.** Lesson 007 ships
   `evaluate.report(y_true, scores, objective)`; lessons 012–015 build `eval.py`, and lesson 015 ships
   `eval.report(cases, results, deterministic_scores, judge_scores)`. After Task 10 Step 2 the §6
   table will list both, adjacent, with unrelated signatures. Nothing in the plan flags the collision
   or the `evaluate.py` vs `eval.py` split. **Low.**

## 2. Numbering consistency

**No off-by-one, skipped or duplicated lesson number.** Task headers, spec filenames, test filenames,
`json validation` `id` fields and the ARCH table rows agree exactly for all eight lessons
(`008-typology-retrieval` / `008_test_retrieval.py` … `015-the-agreement-rate-report` /
`015_test_agreement_report.py`), and Task 10 Step 4's batch run lists the right seven files with 012's
exclusion explained. Three counting/scope items:

6. **Task 9's Files list describes a ledger edit that doesn't exist.** It says
   `Modify: aml-tutor/docs/specs/README.md` *(last remaining `Todo` becomes `Done`)*, and Step 6 says
   "After this step, `docs/specs/README.md` has no remaining `Todo` rows". Verified against the repo:
   `docs/specs/README.md` currently has rows 001–007, **all `Done`, zero `Todo`**. Every task in this
   plan *adds* a new row; none flips one. The parenthetical is a leftover from iteration 002's Task 7,
   and could send an executor hunting for a `Todo` row to flip. **Low.**

7. **"all fifteen lessons" overcounts the ARCH table.** Definition of done: "`docs/ARCHITECTURE.md`
   §6's lesson-to-code API table covers all fifteen lessons." §6 has no row for lesson 001 (project
   setup produces no function), and Task 10 Step 2 adds only 008–015, so the finished table covers
   002–015 = fourteen rows. **Low.**

8. **Test-count off-by-one in Task 4 Step 4.** "Expected: 6 passed" for
   `tests/010_test_structured_decisions.py`, which defines **5** test functions
   (`test_prompt_includes_transaction_score_and_retrieved_titles`,
   `test_tool_schema_names_a_forced_tool_with_the_expected_fields`,
   `test_parse_returns_a_clean_dict_for_a_valid_tool_call`,
   `test_parse_rejects_an_invalid_decision_enum`,
   `test_parse_rejects_a_citation_the_model_was_not_shown`). Every other task's expected count is
   correct (008: 4/4, 009: 4/4, 011: 2/2, 012: 5/5, 013: 3/3, 014: 2/2, 015: 2/2), which makes this
   one look like a real missing test rather than a typo. **Medium.**

## 3. Internal consistency within each Task

9. **Lesson 011's happy-path test derives its "valid" citation from a different retrieval query than
   `triage()` actually issues — and on the plan's own corpus that makes the test fail. HIGH.**
   Task 5 Step 1's `test_triage_forces_the_structured_tool_and_returns_the_parsed_decision` computes
   `retrieved = top_k_typologies_hybrid("many small transfers just under a reporting threshold", k=2,
   ...)` and takes `valid_id = retrieved[0]["id"]`. But Task 5 Step 3's `triage()` builds its own
   query internally:
   `f"{transaction['type']} transaction of amount {transaction['amount']}"` → `"TRANSFER transaction
   of amount 181.0"`, and retrieves `k=2` on *that*. `known_ids` therefore comes from a different
   ranking than `valid_id` did.
   I ran the plan's own `_tfidf_scores` code over the plan's own six-entry corpus (Task 1 Step 1) for
   both queries:
   - `"many small transfers just under a reporting threshold"` → TY-001 (0.463), then all others 0.0
     → `valid_id = "TY-001"`.
   - `"TRANSFER transaction of amount 181.0"` → **TY-003 (0.145), TY-005 (0.129), TY-001 (0.123)** →
     with `k=2`, `known_ids = {"TY-003", "TY-005"}`.
   So `parse_triage_decision` raises `ValueError("cited typology ids not shown to the model: ['TY-001']")`
   and the test fails — with the *citation guard*, i.e. exactly the failure mode the second test is
   supposed to be the only one exercising. The `alpha=0.5` embedding half might or might not pull
   TY-001 into the top two; nothing in the plan makes that deterministic, and the plan's own Open risk
   about embedding-version drift says such things shouldn't be relied on.
   Compounding this, the Note under Task 5 Step 3 asserts the fix is already in place — "the baked-in
   test always passes `corpus_path` explicitly, pinning both `triage`'s own retrieval call and the
   test's separate direct `top_k_typologies_hybrid(...)` call to the same fixture corpus" — which is
   true about the *corpus* and irrelevant to the property the test depends on (the same *query*, hence
   the same top-k). The test needs to either query `top_k_typologies_hybrid` with the exact string
   `triage()` uses, or have `triage()` return/expose the `retrieved` list it used.

10. **The retrieval query `triage()` builds is load-bearing but is specified nowhere in lesson 011's
    spec.** Task 5 Step 5's Implementation order says only "wiring retrieval (008/009) → prompt +
    schema (010) → the forced tool call → parsing (010)". Given finding 9, whether the baked-in test
    passes depends on that exact query string, so a student (or the doer, whose fallback note says it
    "can write `src/aml_triage/triage.py` by hand") will invent a different one and get a citation
    `ValueError` with no way to tell from the spec what was expected. **Medium.**

11. **Lesson 015's spec bullets don't describe the guard its own test requires.** Task 9's test has
    `test_report_raises_on_mismatched_list_lengths`, and the reference implementation raises
    `ValueError(f"mismatched list lengths: {lengths}")`, but Step 5's Implementation order is just
    "implement `report(cases, results, deterministic_scores, judge_scores) -> dict`". Compare the
    finished shape this is supposed to produce: `docs/specs/007-evaluation-and-threshold.md` step 4
    documents its `ValueError` guard explicitly ("Raise, don't silently degrade…"). The `n == 0` case
    (`ZeroDivisionError`) is also unmentioned anywhere. **Low.**

12. **Task 1 Step 4's failure guidance points at data that isn't in `PROVENANCE.md`.** "If it raises
    `SystemExit`, stop and check `tests/fixtures/PROVENANCE.md`'s fraud-row count". Verified: the
    committed `tests/fixtures/PROVENANCE.md` records source, creator, seed, step range, total rows
    (1791), the split-boundary step and the CC BY-SA licence — **no fraud-row count**. (For the record
    the guard cannot fire: the fixture has 590 fraud / 1201 non-fraud rows, so `n = 8` and every "16
    candidates" claim in the plan holds.) **Low.**

13. **Task 6's Pressure test describes `decision_match` as if it were a function.** "lesson 013's
    `decision_match` compares `triage()` output against these hand-written labels" — `decision_match`
    is a *key* in `deterministic_score`'s returned dict, not a comparator. **Low.**

Otherwise the per-task spec-bullet ↔ test ↔ reference-implementation match is good, and in several
places notably careful: lesson 009's `alpha=1.0`-is-exact rationale is arithmetically correct
(`1.0 * tfidf + 0.0 * embeddings` with a stable sort really does reproduce lesson 008's ranking, and I
confirmed the TF-IDF ranking for its query); lesson 013's spec correctly explains why only citation
*presence* is checked given lesson 010's upstream guard; lesson 012's five structural assertions all
hold against the stand-in file Task 6 Step 4 generates (16 rows, exact key set, two decision types,
notes ≥ 10 chars). I also checked the plan's quotations of `aml-triage/SEED.md` against the supplied
file at `/Users/pebarna/projects/temp/aml-triage/SEED.md`: "hand-annotate 30-50 cases", "did it cite
an actual typology; did its escalate/close decision match the label", "the single highest-leverage
output of this whole project", "Weekend 2: RAG + triage agent + triage eval", "problem → method →
measured result", and Task 0 Step 2's expected closing heading all appear verbatim. No findings there.

## 4. Front-matter vs body

14. **The Architecture preamble says lesson 010 calls a real LLM and takes an injected client. It does
    neither.** Departure #1 (lines 23–26): "Lessons 010 (structured decisions) and 011 (the agent) and
    014 (LLM-as-judge) call a real LLM in production use; every one of them takes its LLM client as an
    injectable parameter". Task 4 contradicts this three times: its test docstring ("No LLM call here:
    `build_prompt` and `parse_triage_decision` are pure functions over plain data"), its spec bullet
    ("all pure functions with no SDK import and no network call — deliberate, and why this lesson adds
    no dependency"), and the signatures themselves (no `client` parameter). The "Why these choices"
    bullet gets it right ("lessons 011/014"), so the Architecture sentence reads as an earlier draft
    that survived a scope change. **Medium.**

15. **"No baked-in test requires network access" is stated absolutely and disproved in the plan's own
    steps.** Global Constraints: "no baked-in test may require `ANTHROPIC_API_KEY` or network access
    to pass." Definition of done: "No baked-in test in `aml-tutor/tests/` requires network access or
    `ANTHROPIC_API_KEY` to pass." But `009_test_hybrid_retrieval.py` (all four tests) and
    `011_test_triage_agent.py` (both tests, via `top_k_typologies_hybrid` at default `alpha=0.5`) go
    through `_get_embedding_model()`, and Task 3 Step 4 says so plainly: "First run downloads the
    `all-MiniLM-L6-v2` model weights (~90MB) — expected, one time, **requires network**." The intended
    claim is presumably "no test calls a hosted LLM API"; as written it's a promise a fresh student
    machine breaks on lesson 009. **Medium.**

16. **"No test result depends on the process's cwd" is contradicted by lesson 012.** Global
    Constraints bullet 1 ends "every baked-in test passes it explicitly instead, so no test result
    depends on the process's cwd," and the last bullet says "Fixture path convention unchanged".
    Task 6's test calls `load_eval_set("data/triage_eval_set.jsonl")` — a bare relative path — and
    Task 6 Step 2 says so outright: "this is the first lesson in the tutorial where the test itself,
    not just `import aml_triage`, depends on `cwd`". The Architecture preamble's "Two departures" list
    names lesson 012's human-judgment departure but not this path-resolution departure, and Global
    Constraints carve out no exception. Given the brief flags 012 as the plan's unusual case, this is
    the front-matter place it should be named. **Medium.**

17. **The "At a regulated shop" requirement and the DoD checklist disagree about lesson 012.** Global
    Constraints require the callout in "**Every** new spec"; Definition of done narrows it to "Every
    relevant spec (008, 009, 010, 011, 013, 014, 015)" — omitting 012. But Task 6's spec bullets *do*
    contain one ("**At a regulated shop:** this is what an SR 11-7-style model-risk review actually
    requires — a documented, SME-labeled golden set…"), so the DoD list is the stale one and would
    under-verify the actual deliverable. **Medium.**

18. **Task 10's README.md edit is invisible to the DoD.** Task 10's Files list and Step 3 modify
    `aml-tutor/README.md` (dropping the stale "lessons 002–007 are paused…" sentence), but no
    Definition-of-done bullet covers it. (The quoted sentence matches `README.md` verbatim —
    verified — so the edit itself is well-specified.) **Low.**

19. **The scratch-workspace constraint doesn't cover the fake student repo Task 6 creates.** Global
    Constraints: "every task's throwaway reference implementation lives under
    `/tmp/aml-tutor-plan003-scratch/src/aml_triage/`". Task 6 Step 2 additionally creates
    `/tmp/aml-tutor-plan003-scratch/aml-triage/data/` and Step 4 writes a stand-in labeled file there.
    Task 10 Step 7 deletes the whole parent, so cleanup is fine — but the convention as stated doesn't
    describe what the plan actually does. **Low.**

20. **Ledger row edits are less specified than the precedent.** Iteration 002 spelled out the exact
    before/after markdown for row 002 and then said "same edit shape as Task 2 Step 6"; this plan says
    only "Add row 008 as `Done`" eight times, leaving the row wording, the Goal column text and the
    link format unspecified for all eight new rows. **Low.**

## 5. Narrative flow / forward references

**No genuine forward reference.** Every consumer is preceded by its producer in the order presented:
Task 1's `typologies.json` / `triage_eval_candidates.jsonl` / conftest fixtures precede all users;
008 → 009 (`_load_corpus` / `_documents` / `_tfidf_scores`); 009 → 011 (`top_k_typologies_hybrid`);
010 → 011 (`TRIAGE_TOOL_SCHEMA` / `build_prompt` / `parse_triage_decision`); 012 → 013/014/015 (the
eval-set row shape and `eval.py` itself). The forward-*looking* pressure tests (008 → 010, 012 → 013,
014 → 015) are iteration 002's established convention, not defects. Two smaller items:

21. **Task 5's Interfaces mis-state who calls `triage()`.** "lessons 013–015 call this (or use its
    already-produced `results`) by name." The first lesson to call it is **012** (Task 6's spec:
    "optionally call your own `triage(...)` on it to see what the agent would say"), and lesson **013**
    never calls it — its test hand-writes `result` dicts. Only 015's manual pipeline actually invokes
    it. **Low.**

22. **No task creates the scratch package's `__init__.py`, unlike the precedent.** Iteration 002's
    Task 2 Step 3 said explicitly "Also create `/tmp/aml-tutor-plan002-scratch/src/aml_triage/__init__.py`
    (empty file)"; nothing in plan 003 does, and the scratch directory tree is never explicitly created
    either (tasks just say "Create `/tmp/.../src/aml_triage/retrieval.py`"). Task 5's `triage.py` uses
    relative imports (`from .retrieval import …`, `from .triage_schema import …`), which do work under
    a PEP-420 namespace package, so this is unlikely to break — but it's an unstated divergence from
    the pattern the plan says it matches, and the real `aml-triage` package (which does have
    `__init__.py` and `py.typed`) is what the tests ultimately run against. **Low.**

## 6. cwd / path-resolution consistency of the verification steps

The lesson-012 case is handled correctly, and better than I expected: Task 6 Step 2 `mkdir -p`s the
fake `aml-triage/data`, `cd`s into `/tmp/aml-tutor-plan003-scratch/aml-triage`, and passes the test
file by **absolute** path — so `load_eval_set("data/triage_eval_set.jsonl")` resolves against the fake
repo root while `tests/conftest.py` (and its `eval_candidates_path` fixture) still loads, because
pytest collects conftest files along the test file's own directory chain rather than from the cwd.
Step 4 writes the stand-in file with absolute paths (no cwd dependency), and Step 5 re-`cd`s before
re-running. Task 10 Step 4 correctly excludes 012 from the `cwd=aml-tutor` batch run and says why.
Tasks 2/3/4/5/7/8/9 are genuinely cwd-independent from `aml-tutor`'s root (absolute `typologies_path`
via the session fixture, or pure in-file data), and 012's relative `data/…` path is consistent with the
real validation contract (`cwd: "../aml-triage"`, test passed as `../aml-tutor/tests/…`). Findings:

23. **Task 0 Step 4's commit command cannot succeed — `aml-triage/.gitignore` ignores `SEED.md`. HIGH.**
    The step is
    `cd /Users/pebarna/projects/aml-triage && git add SEED.md && git commit -m "docs: add project SEED (phases 1-3)"`.
    Verified in the sibling repo:
    ```
    $ git -C /Users/pebarna/projects/aml-triage check-ignore -v SEED.md
    .gitignore:2:SEED.md	SEED.md
    ```
    `git add` on an ignored path prints "The following paths are ignored by one of your .gitignore
    files" and exits non-zero, so the `&&`-chained `git commit` never runs. The file would sit
    untracked in `aml-triage` and Definition of done's first bullet ("`aml-triage/SEED.md` exists…")
    would read as satisfied on disk while nothing is committed — and Task 10 Step 5's `git status`
    check ("clean, or only `SEED.md` newly committed") would also silently pass, since the file is
    ignored. Task 0 needs either `git add -f SEED.md` or an explicit step removing the `SEED.md` line
    from `aml-triage/.gitignore`; the plan never mentions that file. (Everything else about Task 0
    checks out: the source file exists at `/Users/pebarna/projects/temp/aml-triage/SEED.md`, it does
    cover all three phases, and it ends with the heading Step 2 expects.)

24. **Task 1 Step 6's collect-only check now sweeps iteration 002's tests, and its expectation is
    environment-dependent.** `PYTHONPATH=/tmp/aml-tutor-plan003-scratch/src python3 -m pytest tests/ -v
    --collect-only`, "Expected: collects with no errors." Unlike iteration 002 (where `tests/` was
    empty at the equivalent step), `tests/` now holds seven committed test files, two of which
    (`002_test_data_loading.py`, `004_test_feature_engineering.py`) `import pandas` at module scope —
    so collection needs pandas in whatever bare `python3` resolves to (on this machine it isn't there;
    `python3 -c "import pandas"` fails, and the plan's other scratch runs assume the same ambient
    interpreter). The check's stated purpose is "`conftest.py` must import cleanly", which a narrower
    target (a single new-fixture-using file, or `--fixtures`) would verify without depending on the
    older lessons' imports. **Low.**

25. **Task 10 Step 5 weakens iteration 002's leak check in exactly the direction Task 6 opened.**
    Iteration 002 Task 8 Step 2 used `find /Users/pebarna/projects/aml-triage -mindepth 1 -not -path
    '*/.git/*'`; this plan replaces it with `git -C /Users/pebarna/projects/aml-triage status`.
    Verified: `aml-triage/.gitignore` line 1 is `data/` — and `data/` is precisely where Task 6's
    throwaway labeled file lives (`.../aml-triage/data/triage_eval_set.jsonl`). If that write ever
    landed in the real sibling repo instead of the `/tmp` stand-in, `git status` would report clean and
    the step would pass. Since Task 6 is the only task in this plan that creates an `aml-triage/data/`
    at all, the check is now blind to the single new leak vector it exists to catch. **Medium.**

## 6b. Also checked, no finding

`uv --directory /tmp/aml-tutor-plan003-scratch pip install sentence-transformers 2>/dev/null || pip install sentence-transformers`
(Task 3 Step 4) is odd — there is no project or venv at that path, so the first arm almost certainly
fails and the fallback bare `pip install` is what runs. It is written with a fallback precisely so it
still works, so I'm not counting it. Task 1's script serialising pandas values with `json.dumps`
also looked like a numpy-scalar hazard; I ran it and pandas' object-dtype upcast in `iterrows()`
boxes to plain `int`/`float`, so Step 4's expected output ("Wrote 16 candidates (8 fraud-flagged, 8
not)") is reachable as written. Fixture arithmetic backs it up: 590 fraud / 1201 non-fraud rows in the
committed fixture, so `n = min(8, 590, 1201) = 8`.
