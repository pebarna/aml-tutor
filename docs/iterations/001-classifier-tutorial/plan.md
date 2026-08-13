# Iteration 001 — Phase 1 classifier tutorial

## Goal

Build an interactive, step-by-step tutorial that teaches the user (strong SWE
background, new to Python data tooling and applied ML) to build Phase 1 of the
AML/fraud triage project — the PaySim-based XGBoost classifier
described in `~/projects/aml-triage/SEED.md` — from scratch, in their own
`aml-triage` repo. It reuses the teaching pattern (not the content) of
`~/projects/software-factory-tutorial`: a browser-led tutor that infers
lessons from `README.md` + `docs/specs/*.md`, asks comprehension questions
and validates the answers, and checks implementation work against baked-in,
pre-written tests rather than judging on the fly.

Scope is Phase 1 only (classifier + its eval). Phases 2/3 from
`aml-triage/SEED.md` (RAG triage agent, agreement-rate eval) are out of
scope for this iteration.

## Why these choices (decisions made during brainstorming)

- **Fork `tutorial-engine`, don't rebuild it.** It already implements exactly
  the pattern SEED.md asks for — directory-inferred lessons, a ledger, a
  validation-command protocol, a browser UI — generically enough that only
  the model backend and the cross-repo workspace resolution need to change.
- **Replace the Pi SDK with the Claude Agent SDK.** Keeps the whole
  toolchain (tutor + optional doer) in the ecosystem already authenticated
  and used day to day.
- **Choose XGBoost as the canonical baseline library.** LightGBM is
  out of scope for this iteration; keeping the stack to one gradient-boosting
  library removes API ambiguity from Lessons 005–006 and the dependency list.
- **Sibling repos, not a merged one.** `aml-tutor` stays pure tutorial
  content + engine; `aml-triage` is the real deliverable. Lessons write into
  `../aml-triage`, mirroring how the original's lessons write into
  `calculator/`, just across a repo boundary instead of a subfolder.
- **One lesson per Phase-1 skill (7 lessons)**, including a dedicated data
  literacy lesson (002) before feature engineering, because the learner is
  new to pandas/data tooling, not just ML.
- **Tests are visible before implementation (TDD-style).** Each lesson's
  spec includes the pytest file the implementation must satisfy, so
  "correct" is unambiguous before the student starts, and checking is a
  deterministic command, not an on-the-fly judgment call.
- **A doer option stays available per step**, matching the original's
  offer-to-do-it-for-you pattern, scoped to `../aml-triage` only, no shell
  access.
- **uv** for Python tooling (fast, single lockfile, good fit for a
  from-scratch weekend project).
- **Tutorial-owned tests, triage-targeted execution.** Baked-in pytest suites
  stay in `aml-tutor/tests/`; the validation runner executes them with
  `cwd=../aml-triage` so `import aml_triage` resolves to the student's
  package. The student does not author or modify the tests.

## Target repo layout (end state)

```
aml-tutor/                          (this repo — engine + content)
  README.md                         orientation, links the ledger
  docs/specs/README.md              lesson ledger (7 rows)
  docs/specs/001-project-setup.md
  docs/specs/002-load-and-explore-the-data.md
  docs/specs/003-time-based-split.md
  docs/specs/004-feature-engineering.md
  docs/specs/005-class-imbalance.md
  docs/specs/006-train-the-baseline.md
  docs/specs/007-evaluation-and-threshold.md
  tests/                            pytest suites, one per lesson, run against ../aml-triage
    001_test_project_setup.py
    002_test_data_loading.py
    ... etc.
  tests/fixtures/                   deterministic, small slices carved from PaySim CSV
  tutorial-engine/                  forked from software-factory-tutorial/tutorial-engine
    src/agent/                      claude-agent-adapter.ts replaces pi-adapter.ts
    src/lesson/                     load.ts generalized for cross-repo workspace
    src/validation/                 runner.ts: validation commands run with cwd=../aml-triage
    web/                            React UI, carried over largely as-is
  scripts/setup.mjs                 Claude Agent SDK auth + permission readiness check
  scripts/tutorial.mjs              launches the engine pointed at this directory
  package.json                      workspaces: tutorial-engine; no calculator workspace

aml-triage/                         (sibling repo — the real Phase 1 deliverable)
  data/PS_20174392719_...csv        already present
  pyproject.toml, uv.lock           written by lesson 001
  src/aml_triage/                   student's implementation, grows lesson by lesson
  tests/                            optional local test target; no baked-in tests live here
                                    (the tutorial engine runs tests from aml-tutor/tests/)
```

## Architecture contract (write before Phase A)

Before touching the engine, pin the following in a short `docs/ARCHITECTURE.md` or
inline note so every later phase has a single source of truth:

1. **Test ownership.** Baked-in pytest suites are authoritative in
   `aml-tutor/tests/` and are not authored or modified by the student.
   Validation commands run them from `aml-triage` by passing the absolute or
   `../aml-tutor/tests/...` path.
2. **Library stack.** XGBoost is the canonical gradient-boosting library for
   this iteration; LightGBM is out of scope.
3. **SDK auth.** The exact Claude Agent SDK credential mode (e.g. API key env
   var) and the exact readiness check used by `scripts/setup.mjs` are pinned.
4. **Doer tool grant.** Read access to `aml-tutor` and `aml-triage`; write
   access only inside `aml-triage`; no bash; no escape outside the two sibling
   roots.
5. **`cwd` resolution.** `validationCommands.cwd` is resolved relative to the
   discovered tutorial root, not the shell's process cwd.
6. **Lesson-to-code API.** Each lesson produces a small, named public function
   in `src/aml_triage/`. Example mapping:
   - 002: `data.load_transactions(path)` → DataFrame
   - 003: `split.temporal_split(df, ...)` → train_df, test_df
   - 004: `features.add_features(df)` → DataFrame
   - 005: `imbalance.compute_scale_pos_weight(y)` → float
   - 006: `model.train_baseline(X_train, y_train, weight)` → fitted XGBoost
   - 007: `evaluate.report(y_true, scores, objective)` → metrics + threshold

## Phase A — Fork and adapt the engine

1. Copy `software-factory-tutorial/tutorial-engine` into `aml-tutor/tutorial-engine`.
   Drop the `calculator`-specific bits from the root (`factory/`, `evals/`
   calculator fixtures) — they don't apply here.
2. Prune the root `package.json` and `check` script: remove the `calculator`
   workspace, `evals` fixtures, and any calculator-specific check steps so the
   root `npm run check` exercises only the engine and the tutorial content.
3. Replace `@earendil-works/pi-coding-agent` with `@anthropic-ai/claude-agent-sdk`
   in `tutorial-engine/package.json` and the root `package.json`.
4. Rewrite `src/agent/pi-adapter.ts` → `src/agent/claude-agent-adapter.ts`:
   tutor session (read-only tools, both repos) and doer session (read/edit/write
   in `../aml-triage` only, no bash), same boundary discipline as the original.
   The doer must be able to read files in `aml-triage` and the test files in
   `aml-tutor/tests/` to understand what to fix, but all writes stay inside
   `aml-triage`.
5. Generalize `src/agent/workspace-boundary.ts` to accept two allowed roots
   (`aml-tutor` for reading lesson content, `aml-triage` for writing code)
   instead of one.
6. Update `src/lesson/load.ts` / `src/validation/runner.ts` so a lesson's
   `validationCommands` can specify a `cwd` distinct from the tutorial
   directory. The `cwd` is resolved relative to the discovered tutorial root
   (the directory containing `README.md` and `docs/specs/`), not the shell's
   process cwd. This is required so `uv run pytest ...` executes inside
   `../aml-triage` while the test file path remains in `aml-tutor/tests/`.
7. Rewrite `src/lesson/system-prompt.ts` (or equivalent) to replace
   calculator-specific framing (Kent Beck rules, `factory/refactor/success.md`,
   `pi` command references) with AML/`uv`/pytest framing and the target learner
   profile.
8. Rewrite `scripts/setup.mjs` to check the Claude Agent SDK's supported
   credential mode and tool-permission readiness instead of Pi's `/login` state.
   Pin the SDK version and the exact auth mechanism (e.g. API key env var or
   Claude Code session). Drop `TUTOR_MODEL`/Pi-specific env vars from the README.
9. Add explicit test coverage for the two genuinely new behaviours before moving
   on:
   - Unit tests for two-root `WorkspaceBoundary` (accept both roots, reject
     `..` escapes and third paths).
   - Unit tests for per-command `cwd` in `ValidationRunner`.
   - A smoke test that verifies the doer can write a file in `../aml-triage` and
     cannot invoke Bash or write outside `aml-triage`.
10. Run the engine's test suite (`npm run check` inside `tutorial-engine/` and
    the root `check` script) after each rewrite step; fix or rewrite tests that
    assumed the Pi adapter or the single-root/single-cwd engine.

## Phase B — Lesson 001: project setup

1. In `aml-triage`, nothing exists yet but the CSV — lesson 001 is where the
   student runs `uv init`, adds `pandas`/`xgboost`/`scikit-learn`/`pytest`
   as dependencies, and creates the `src/aml_triage/` + `tests/` layout.
   The spec also documents the doer fallback path: because the doer has no bash
   access, it must scaffold the same files directly (write `pyproject.toml`,
   `uv.lock`, `src/aml_triage/__init__.py`, and an empty `tests/` marker) when
   the student asks it to take the step.
2. Write `aml-tutor/docs/specs/001-project-setup.md` (Key concept /
   Implementation order / Checks / Pressure test, same shape as the
   original's lessons).
3. Write `aml-tutor/tests/001_test_project_setup.py` — a baked-in check that
   `pyproject.toml`, `uv.lock`, and the expected package layout exist and
   `uv run python -c "import aml_triage"` succeeds.
4. Add the lesson's `validationCommand`:
   - `uv run pytest ../aml-tutor/tests/001_test_project_setup.py` (or the
     absolute path to the test file), with `cwd=../aml-triage` resolved relative
     to the tutorial root.

## Phase C — Lessons 002–007: content authoring

0. **Fixture foundation.** Before authoring any lesson, carve a small,
   deterministic fixture slice from the real PaySim CSV and commit it under
   `aml-tutor/tests/fixtures/`. Include a documented provenance note (PaySim
   license permits redistribution of a small slice) and the carving script. All
   Lessons 002–007 assert against this shared fixture, not the full CSV.

For each lesson, in order, produce three things together — the spec, the
baked-in pytest file, and (where relevant) any small fixture data the test
needs from the shared fixture set:

| # | Spec focus | Test asserts |
|---|---|---|
| 002 | Load PaySim CSV with pandas, validate expected columns/dtypes, compute fraud rate, explain why accuracy is meaningless here | a loader function returns the right row count / fraud rate on a small fixture slice |
| 003 | Time-based split by the `step` column (no overlapping steps between splits, tie handling documented), why random split leaks | split function respects chronology and produces non-overlapping, correctly-ordered train/test sets |
| 004 | Derive features from raw transaction fields (e.g. balance deltas, transaction-type flags) with explicit leakage policy | feature function output matches expected values on known fixture rows; spec documents which columns are decision-time available and which identifiers/labels must be excluded |
| 005 | Class imbalance handling (`scale_pos_weight` for XGBoost) — explain the tradeoff, not just apply a knob | a helper computes the correct imbalance weight / config consumed by the trainer |
| 006 | Train an XGBoost baseline; why gradient boosting over a neural net for tabular fraud data | model trains, predicts, and produces scores in valid probability range on fixture data |
| 007 | Precision/recall/PR-AUC, picking and justifying an operating threshold against a concrete objective (e.g. recall ≥ X at precision ≥ Y, or cost-matrix minimization) | eval function computes correct precision/recall/PR-AUC on a known fixture and selects a threshold according to the stated objective |

Each spec keeps the four-part shape and, per the TDD-style decision, embeds
or links its own test file inline so the student sees the target before
implementing. Each spec's "Checks" section pairs the deterministic pytest
command with 2-3 comprehension questions the tutor asks and validates in
conversation (e.g. "why does PR-AUC matter more than accuracy here — answer
in your own words").

Update `docs/specs/README.md` ledger after each spec is written, matching the
original's `LEDGER_STATUSES` convention the engine already parses. The ledger
tracks *authoring completeness* for the tutorial maintainers, not per-student
progress. Each new lesson starts as `Todo` and is advanced to `Done` once its
spec, test, and fixture are written and the test passes against a fresh
`aml-triage`.

## Phase D — Wiring and orientation

1. Write `aml-tutor/README.md`: title, one-paragraph orientation, setup
   instructions (Claude Agent SDK auth + permission readiness, `uv` install,
   sibling-repo expectation).
2. Confirm `scripts/tutorial.mjs` launches the engine pointed at `aml-tutor`
   while lessons resolve their validation `cwd` relative to the discovered
   tutorial root into `../aml-triage` (requires the two repos to be sibling
   directories — document that assumption explicitly in the README's setup
   section).
3. Decide and document the doer's tool grant precisely: read access to
   `aml-tutor` (lesson content and test files) and `aml-triage` (student code);
   edit/write access only inside `../aml-triage`; no bash; no access outside
   the two sibling repos. Consistent with Phase A step 4.

## Phase E — Dry run

1. **Vertical slice after Phase A/B.** Before authoring later lessons, run a
   minimal end-to-end path: launcher → lesson discovery → ledger → one
   comprehension question → one doer write → one cross-repo validation command
   (`uv run pytest`) → displayed result. Confirm `uv` resolves inside the
   runner's scrubbed spawn environment (only `PATH`, `HOME`, `CI`, `NO_COLOR`
   passed). Fix any boundary, import, or command failures before writing
   Lessons 002–007.
2. **Milestone regression after each lesson.** After each Phase C lesson is
   written, run the cumulative baked-in test suite against a fresh
   `aml-triage` to catch spec/test mismatches early.
3. **Full manual dry run.** Walk the complete tutorial once, start to finish,
   as the student would: fresh `aml-triage` with just the CSV, run every lesson
   in order, answer the comprehension questions, let the doer take at least one
   step to confirm that path still works, run every baked-in test.
4. Fix any spec/test mismatches found during the dry run before calling the
   iteration done.

*Note on comprehension questions.* Code correctness is validated deterministically
by pytest. The 2–3 comprehension questions per lesson are validated by the
tutor model in conversation; this is the deliberate separation between
“does it work?” and “does the student understand why?”.

## Definition of done

- `npm run check` passes in the forked `tutorial-engine` and the root check
  script (pruned to cover only the engine + tutorial content).
- All 7 lesson specs exist, are listed in the ledger, and each has a passing
  baked-in test runnable against a freshly scaffolded `aml-triage`.
- The deterministic fixture tests in `aml-tutor/tests/` run reproducibly and
  do not depend on the full PaySim CSV.
- A documented command exists that, from a clean seeded `aml-triage`, installs
  dependencies and runs the full tutorial-owned test suite.
- A full manual dry run (Phase E) completes end to end and produces a real,
  working Phase 1 classifier in `aml-triage` with precision/recall/PR-AUC and a
  chosen operating threshold reported.
- Lesson 007 produces or saves a report artifact (e.g. threshold, metrics, model
  path) in `aml-triage` so the Phase 1 stopping point is a concrete, defensible
  deliverable.
- `aml-triage/SEED.md`'s Phase 1 "stopping point" is satisfied: a real,
  defensible classifier exists even if nothing further is built.

## Open risks / watch items

- The Claude Agent SDK's exact credential and tool-permission model must be
  pinned before `scripts/setup.mjs` is implemented — verify the no-bash,
  path-scoped boundary is actually enforced (not just requested) before trusting
  the doer path. This is now explicitly covered in Phase A step 9.
- Cross-repo `cwd` resolution in `validation/runner.ts` is new relative to
  the original engine (which never left its own directory) — it now has its own
  unit-test coverage in Phase A step 9 and must resolve relative to the
  discovered tutorial root, not the shell's cwd.
- Fixture data for lessons 002-007 needs to be small and deterministic
  (not the full PaySim CSV) so tests run fast and give reproducible
  assertions — carve these out from the real CSV in Phase C step 0, not by
  inventing synthetic rows.
- PaySim license/redistribution: confirm the committed fixture slice is small
  enough and licensed for redistribution before shipping it in a public repo.
- `uv` must resolve inside the runner's stripped spawn environment. If it does
  not, the engine's `PATH` passing logic may need to be widened or the README
  must instruct users to set a specific env var.
