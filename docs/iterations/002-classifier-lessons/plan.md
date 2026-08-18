# Iteration 002 — Finish the classifier curriculum (lessons 002–007) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Carve the PaySim fixture iteration 001 deferred, then write the six remaining lesson
specs and their baked-in pytest tests (`docs/specs/002-*` through `007-*`) so the ledger reads
fully `Done` and a fresh `aml-triage` can be walked end-to-end to a real Phase 1 classifier.

**Architecture:** Pure content on top of iteration 001's unchanged engine. Each lesson task
produces a spec (`docs/specs/NNN-*.md`), a baked-in test (`aml-tutor/tests/NNN_test_*.py`) that
asserts against one shared committed fixture, and a ledger row. "Implementation" for each task
means writing a throwaway *reference* solution to that lesson's pinned function — scratch-only,
never committed — solely to prove the baked-in test is actually satisfiable and to compute any
values the test needs to check.

**Tech Stack:** Python 3 + pandas + scikit-learn + xgboost + pytest (all already pinned as
dependencies by lesson 001); the scratch reference implementation needs the same stack available
locally to run.

## Global Constraints

- Lesson-to-code API (from `docs/ARCHITECTURE.md` §6, unchanged by this iteration):
  `data.load_transactions(path) -> DataFrame`,
  `split.temporal_split(df, split_step) -> (train_df, test_df)`,
  `features.add_features(df) -> DataFrame`,
  `imbalance.compute_scale_pos_weight(y) -> float`,
  `model.train_baseline(X_train, y_train, weight) -> fitted XGBoost`,
  `evaluate.report(y_true, scores, objective) -> dict` (see Task 7 for the exact `dict` shape).
- Every baked-in test resolves the fixture path relative to the test file itself
  (`Path(__file__).parent / "fixtures" / "paysim_fixture.csv"`), never via the working directory —
  each test actually *runs* with `cwd=../aml-triage` (`ARCHITECTURE.md` §1/§5).
- Every new spec follows `docs/specs/001-project-setup.md`'s four-part shape (Key concept /
  Implementation order / Checks / Pressure test), its `json validation` fenced-block convention for
  wiring the baked-in test, and its "If you ask the tutor to do this step for you" doer-fallback
  note where the lesson has the student run a command the doer (no shell access) cannot.
- **Scratch workspace convention:** every task's throwaway reference implementation lives under
  `/tmp/aml-tutor-plan002-scratch/src/aml_triage/`, growing one module per task. This directory is
  never committed to `aml-tutor` or `aml-triage`, and is explicitly discarded in Task 8 — the real
  `aml-triage` stays empty except for `.gitignore` throughout this plan, exactly as a real student
  would find it.
- **Blocked prerequisite:** Task 1 cannot run until the full PaySim CSV (~470MB, 6.3M rows,
  Kaggle's "Synthetic Financial Datasets For Fraud Detection", `ealaxi/paysim1`) has been downloaded
  and its local path is known. Nothing else in this plan needs the full CSV — only Task 1 touches
  it, once.

---

## Background (from brainstorming, ensemble-reviewed)

**Why a stratified + hand-picked fixture, not pure random sampling.** PaySim's fraud rate is
~0.13%; a small pure-random draw could land zero or one fraud row, too thin to test imbalance
handling (Task 5) or PR-AUC/threshold logic (Task 7). The fixture instead keeps every fraud row
within a bounded `step` window wide enough to contain both fraud types (`TRANSFER`, `CASH_OUT` —
the only two PaySim ever marks fraudulent), plus a random fixed-seed draw of non-fraud rows, plus
at least one zero-balance row (a common real pattern a naive ratio feature can divide by zero on —
see Task 4).

**Why the student downloads the full CSV but never trains on it directly.** The student does
download the full file once, in lesson 002 (Task 2) — unavoidable, since their working sample is
drawn from it — but a documented, deterministic (fixed-seed) sampling step immediately produces
the much smaller file (`aml-triage/data/paysim_sample.csv`, ~100k–300k rows, gitignored, never
committed) that every later lesson actually works with. Nobody trains on the full 6.3M-row file.
The committed pytest fixture is a separate, much smaller file carved once by whoever runs Task 1 —
not by the student, and not derived from the student's sample.

**Why the evaluation objective is pinned.** Lesson 007 (Task 7) teaches "maximize recall subject to
precision ≥ 90%" as a single, concrete, explainable fraud-detection framing, rather than presenting
several unresolved options.

**Why the leakage policy excludes `nameOrig`/`nameDest`/`isFlaggedFraud`.** `nameOrig`/`nameDest`
are high-cardinality transaction-party identifiers. `isFlaggedFraud` is the PaySim simulator's own
naive rule (flags transfers over 200,000) — close enough to label-adjacent that using it as a
feature would be closer to cheating than modeling. Balance fields and derived deltas are
decision-time-available and kept.

---

## Task 0: Commit iteration 001's outstanding work

Nothing from iteration 001 is committed yet — only the repo's original `SEED.md`/plan.md commit
exists; `docs/ARCHITECTURE.md`, `README.md`, `docs/specs/`, `tests/`, `tutorial-engine/`,
`package.json`, and `scripts/` are all still untracked. Not required by this iteration's design,
but doing it first gives iteration 001 and iteration 002 a clean boundary in git history before any
of this plan's commits land on top.

**Files:**
- Modify: none (staging + committing existing untracked/modified files only)

**Interfaces:** None — this task has no code interface, nothing later depends on its internals
beyond "the working tree is clean before Task 1 starts."

- [ ] **Step 1: Review what's about to be committed**

Run: `git status` in `aml-tutor`. Confirm the untracked/modified list matches: `SEED.md`,
`docs/iterations/001-classifier-tutorial/plan.md` (both modified — path-typo fixes), plus
untracked `.gitignore`, `README.md`, `docs/ARCHITECTURE.md`, `docs/specs/`, `package-lock.json`,
`package.json`, `scripts/`, `tests/`, `tutorial-engine/`, `docs/iterations/002-classifier-lessons/`.

- [ ] **Step 2: Stage and commit**

```bash
git add SEED.md docs/iterations/001-classifier-tutorial/plan.md .gitignore README.md \
  docs/ARCHITECTURE.md docs/specs package.json package-lock.json scripts tests tutorial-engine
git commit -m "feat: implement iteration 001 — tutorial engine fork and lesson 001"
```

- [ ] **Step 3: Confirm the tree is clean**

Run: `git status`. Expected: only `docs/iterations/002-classifier-lessons/plan.md` remains
untracked (this plan itself) — commit that separately once it's finalized, or fold it into Task 8's
final commit.

---

## Task 1: Carve the fixture

**Files:**
- Create: `aml-tutor/scripts/carve_fixture.py`
- Create (generated by running the script, then committed): `aml-tutor/tests/fixtures/paysim_fixture.csv`
- Create (generated by running the script, then committed): `aml-tutor/tests/fixtures/PROVENANCE.md`

**Interfaces:**
- Consumes: `<FULL_CSV_PATH>` — the full PaySim CSV's local path, supplied at execution time (this
  task is blocked without it; see Global Constraints).
- Produces: `tests/fixtures/paysim_fixture.csv` (committed fixture, ~1,500–2,000 rows) and
  `tests/fixtures/PROVENANCE.md`, which records the exact **split-boundary `step` value** Task 3
  depends on.

- [ ] **Step 1: Write the carving script**

Create `aml-tutor/scripts/carve_fixture.py`:

```python
#!/usr/bin/env python3
"""One-time script: carve a small, deterministic PaySim fixture for aml-tutor's baked-in tests.

Usage: python scripts/carve_fixture.py /path/to/full/paysim.csv

Never run against a re-hosted copy of the dataset — only against your own download from Kaggle's
"Synthetic Financial Datasets For Fraud Detection" (ealaxi/paysim1).
"""
import sys
from pathlib import Path

import pandas as pd

SEED = 20260818
NON_FRAUD_SAMPLE_SIZE = 1200
STEP_RANGE = (1, 50)  # bounded window; widened if it doesn't satisfy the checks below
FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "paysim_fixture.csv"
PROVENANCE_PATH = FIXTURE_PATH.parent / "PROVENANCE.md"


def carve(full_csv_path: str) -> None:
    df = pd.read_csv(full_csv_path)
    windowed = df[(df["step"] >= STEP_RANGE[0]) & (df["step"] <= STEP_RANGE[1])]

    fraud = windowed[windowed["isFraud"] == 1]
    fraud_types = set(fraud["type"].unique())
    if not {"TRANSFER", "CASH_OUT"}.issubset(fraud_types):
        raise SystemExit(
            f"step range {STEP_RANGE} does not contain both fraud types (found {fraud_types}); "
            "widen STEP_RANGE in this script and re-run."
        )

    zero_balance = windowed[(windowed["oldbalanceOrg"] == 0) & (windowed["newbalanceOrig"] == 0)]
    if zero_balance.empty:
        raise SystemExit(
            f"step range {STEP_RANGE} has no zero-balance row; widen STEP_RANGE and re-run."
        )

    non_fraud_pool = windowed[windowed["isFraud"] == 0]
    non_fraud = non_fraud_pool.sample(
        n=min(NON_FRAUD_SAMPLE_SIZE, len(non_fraud_pool)), random_state=SEED
    )

    steps_present = sorted(windowed["step"].unique())
    if len(steps_present) < 2:
        raise SystemExit(f"step range {STEP_RANGE} has fewer than 2 distinct steps; widen it.")
    split_boundary_step = steps_present[len(steps_present) // 2]
    step_after_boundary = next(s for s in steps_present if s > split_boundary_step)

    fixture = (
        pd.concat([fraud, zero_balance.head(1), non_fraud])
        .drop_duplicates()
        .sort_values(["step"])
        .reset_index(drop=True)
    )

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fixture.to_csv(FIXTURE_PATH, index=False)

    PROVENANCE_PATH.write_text(
        "# paysim_fixture.csv provenance\n\n"
        '- Source: Kaggle, "Synthetic Financial Datasets For Fraud Detection" (ealaxi/paysim1).\n'
        f"- Carved by: scripts/carve_fixture.py, seed={SEED}, step range {STEP_RANGE}.\n"
        f"- Rows: {len(fixture)}.\n"
        f"- Split-boundary step (lesson 003): {split_boundary_step} "
        f"(train: step <= {split_boundary_step}; test starts at step {step_after_boundary}).\n"
        "- License permits redistribution of a small slice; this fixture is not the full "
        "dataset and was never committed alongside it.\n"
    )
    print(f"Wrote {len(fixture)} rows to {FIXTURE_PATH}")
    print(f"Split-boundary step: {split_boundary_step} (next present step: {step_after_boundary})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/carve_fixture.py /path/to/full/paysim.csv")
    carve(sys.argv[1])
```

- [ ] **Step 2: Run it against the full CSV**

Run: `python3 scripts/carve_fixture.py <FULL_CSV_PATH>` from `aml-tutor`'s root.
Expected: prints the row count and split-boundary step; `tests/fixtures/paysim_fixture.csv` and
`tests/fixtures/PROVENANCE.md` now exist. If it raises `SystemExit` about the step range, widen
`STEP_RANGE` in the script (e.g. to `(1, 100)`) and re-run — this is expected tuning against real
data, not a bug.

- [ ] **Step 3: Spot-check the fixture**

Run:

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('tests/fixtures/paysim_fixture.csv')
print('rows:', len(df))
print('fraud rows:', int(df['isFraud'].sum()))
print('fraud types:', sorted(df[df['isFraud'] == 1]['type'].unique()))
print('zero-balance rows:', int(((df['oldbalanceOrg'] == 0) & (df['newbalanceOrig'] == 0)).sum()))
"
```

Expected: `rows` between 1,000 and 2,500; `fraud rows` > 0; `fraud types` includes both
`CASH_OUT` and `TRANSFER`; `zero-balance rows` >= 1.

- [ ] **Step 4: Write the shared pytest fixtures**

Tasks 2-7's tests all need the fixture's path, and Tasks 3/5/6 additionally need the exact
split-boundary `step` value from `PROVENANCE.md`. Rather than each test file hardcoding its own
copy of these (duplication a task reviewer would flag independently in three different tasks),
create `aml-tutor/tests/conftest.py` once, here, so every later task's tests request them as
ordinary pytest fixtures:

```python
"""Shared pytest fixtures for aml-tutor's baked-in lesson tests."""
import re
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_path():
    return FIXTURES_DIR / "paysim_fixture.csv"


@pytest.fixture(scope="session")
def split_step():
    text = (FIXTURES_DIR / "PROVENANCE.md").read_text()
    match = re.search(r"Split-boundary step \(lesson 003\): (\d+)", text)
    if not match:
        raise RuntimeError("Could not find the split-boundary step in PROVENANCE.md")
    return int(match.group(1))
```

Run: `PYTHONPATH=/tmp/aml-tutor-plan002-scratch/src python3 -m pytest tests/ -v --collect-only`
Expected: pytest collects with no errors (there are no test functions yet, but `conftest.py` must
import cleanly).

- [ ] **Step 5: Commit**

```bash
git add scripts/carve_fixture.py tests/fixtures/paysim_fixture.csv tests/fixtures/PROVENANCE.md tests/conftest.py
git commit -m "feat: carve PaySim fixture for lessons 002-007"
```

---

## Task 2: Lesson 002 — load and explore the data

**Files:**
- Create: `aml-tutor/docs/specs/002-load-and-explore-the-data.md`
- Create: `aml-tutor/tests/002_test_data_loading.py`
- Modify: `aml-tutor/docs/specs/README.md` (row 002: `Todo — blocked...` → `Done`)
- Create (scratch, not committed): `/tmp/aml-tutor-plan002-scratch/src/aml_triage/data.py`

**Interfaces:**
- Consumes: `tests/fixtures/paysim_fixture.csv` (Task 1).
- Produces: reference `load_transactions(path: str) -> pd.DataFrame`, columns
  `step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud`
  — Task 3 imports this signature by name.

- [ ] **Step 1: Write the failing test**

Create `aml-tutor/tests/002_test_data_loading.py`:

```python
"""Baked-in check for lesson 002 — load and explore the data.

Run via `uv run pytest ../aml-tutor/tests/002_test_data_loading.py` with cwd=aml-triage, so
`import aml_triage` resolves to the student's package. fixture_path comes from tests/conftest.py
(Task 1).
"""
import pandas as pd

EXPECTED_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]


def test_columns_and_row_count_match_the_fixture(fixture_path):
    from aml_triage.data import load_transactions

    df = load_transactions(str(fixture_path))
    raw_row_count = sum(1 for _ in open(fixture_path)) - 1  # minus header
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == raw_row_count


def test_dtypes_are_numeric_where_expected(fixture_path):
    from aml_triage.data import load_transactions

    df = load_transactions(str(fixture_path))
    for column in ["step", "amount", "oldbalanceOrg", "newbalanceOrig",
                    "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]:
        assert pd.api.types.is_numeric_dtype(df[column]), f"{column} is not numeric"


def test_fraud_rate_matches_an_independent_recomputation(fixture_path):
    from aml_triage.data import load_transactions

    df = load_transactions(str(fixture_path))
    raw = pd.read_csv(fixture_path)
    assert df["isFraud"].mean() == raw["isFraud"].mean()
    assert 0 < df["isFraud"].mean() < 0.5, "fixture should be fraud-heavy but still a minority class"
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `PYTHONPATH=/tmp/aml-tutor-plan002-scratch/src python3 -m pytest tests/002_test_data_loading.py -v`
Expected: `ModuleNotFoundError: No module named 'aml_triage'`.

- [ ] **Step 3: Write the scratch reference implementation**

Create `/tmp/aml-tutor-plan002-scratch/src/aml_triage/data.py`:

```python
import pandas as pd

EXPECTED_DTYPES = {
    "step": "int64", "amount": "float64",
    "oldbalanceOrg": "float64", "newbalanceOrig": "float64",
    "oldbalanceDest": "float64", "newbalanceDest": "float64",
    "isFraud": "int64", "isFlaggedFraud": "int64",
}


def load_transactions(path):
    df = pd.read_csv(path)
    missing = set(EXPECTED_DTYPES) - set(df.columns)
    if missing:
        raise ValueError(f"missing expected columns: {sorted(missing)}")
    for column, dtype in EXPECTED_DTYPES.items():
        df[column] = df[column].astype(dtype)
    return df
```

Also create `/tmp/aml-tutor-plan002-scratch/src/aml_triage/__init__.py` (empty file).

- [ ] **Step 4: Run it to confirm it passes**

Run: `PYTHONPATH=/tmp/aml-tutor-plan002-scratch/src python3 -m pytest tests/002_test_data_loading.py -v`
Expected: 3 passed.

- [ ] **Step 5: Write the spec**

Create `aml-tutor/docs/specs/002-load-and-explore-the-data.md` following `001-project-setup.md`'s
four-part shape. It must cover, concretely:

- **Key concept:** why accuracy is meaningless at ~0.1% fraud prevalence (a model predicting "never
  fraud" scores >99.8% accuracy while catching zero fraud) — motivating precision/recall over
  accuracy for the rest of the tutorial.
- **Implementation order**, in this order: (1) download the full PaySim CSV from Kaggle
  (`ealaxi/paysim1`) and place it outside the repo; (2) run a documented, fixed-seed sampling
  command — the spec must show the exact command, e.g.
  `uv run python -c "import pandas as pd; df = pd.read_csv('<full path>'); df.sample(n=200000, random_state=20260818).to_csv('data/paysim_sample.csv', index=False)"`
  — producing `aml-triage/data/paysim_sample.csv` (add `data/` to `aml-triage/.gitignore`); (3)
  implement `load_transactions(path) -> DataFrame` in `src/aml_triage/data.py`; (4) load the sample,
  validate columns/dtypes, print the fraud rate.
- **Checks:** 2-3 comprehension questions (e.g. "why would 99.8% accuracy be a meaningless claim
  about this model?"), then the baked-in `uv run pytest ../aml-tutor/tests/002_test_data_loading.py`
  command and its `json validation` block (`id: "002-load-and-explore-the-data"`,
  `cwd: "../aml-triage"`).
- **Pressure test:** the next lesson (003) assumes `load_transactions` returns a DataFrame sorted
  by nothing in particular — an unstated ordering assumption there would surface as a confusing
  split bug, not a loading bug.
- **Doer fallback note:** the doer cannot run the download/sampling command (no shell access); if
  asked to do this step, it can only write `src/aml_triage/data.py` by hand — the student must run
  the download/sample themselves regardless, same pattern as lesson 001's `uv init`.

- [ ] **Step 6: Update the ledger**

In `aml-tutor/docs/specs/README.md`, change row 002 from
`| [002](002-load-and-explore-the-data.md) | Load and explore the data | Todo — blocked on the PaySim CSV fixture, see \`docs/ARCHITECTURE.md\` |`
to
`| [002](002-load-and-explore-the-data.md) | Load and explore the data | Done |`.

- [ ] **Step 7: Commit**

```bash
git add docs/specs/002-load-and-explore-the-data.md tests/002_test_data_loading.py docs/specs/README.md
git commit -m "feat: lesson 002 — load and explore the data"
```

---

## Task 3: Lesson 003 — time-based split

**Files:**
- Create: `aml-tutor/docs/specs/003-time-based-split.md`
- Create: `aml-tutor/tests/003_test_time_based_split.py`
- Modify: `aml-tutor/docs/specs/README.md` (row 003)
- Create (scratch): `/tmp/aml-tutor-plan002-scratch/src/aml_triage/split.py`

**Interfaces:**
- Consumes: Task 2's `load_transactions`; the split-boundary `step` value from Task 1's
  `tests/fixtures/PROVENANCE.md`.
- Produces: reference `temporal_split(df: pd.DataFrame, split_step: int) -> tuple[pd.DataFrame, pd.DataFrame]`
  — Task 5's test imports this to build its train split.

- [ ] **Step 1: Write the failing test**

Create `aml-tutor/tests/003_test_time_based_split.py`. `fixture_path` and `split_step` come from
`tests/conftest.py` (Task 1) — no need to open `PROVENANCE.md` by hand:

```python
"""Baked-in check for lesson 003 — time-based split."""


def test_split_has_no_overlapping_steps(fixture_path, split_step):
    from aml_triage.data import load_transactions
    from aml_triage.split import temporal_split

    df = load_transactions(str(fixture_path))
    train_df, test_df = temporal_split(df, split_step)
    assert set(train_df["step"]).isdisjoint(set(test_df["step"]))


def test_split_is_chronologically_ordered_and_complete(fixture_path, split_step):
    from aml_triage.data import load_transactions
    from aml_triage.split import temporal_split

    df = load_transactions(str(fixture_path))
    train_df, test_df = temporal_split(df, split_step)
    assert train_df["step"].max() <= split_step
    assert test_df["step"].min() > split_step
    assert len(train_df) + len(test_df) == len(df)


def test_boundary_step_lands_entirely_in_train(fixture_path, split_step):
    from aml_triage.data import load_transactions
    from aml_triage.split import temporal_split

    df = load_transactions(str(fixture_path))
    train_df, test_df = temporal_split(df, split_step)
    boundary_rows_in_raw = (df["step"] == split_step).sum()
    boundary_rows_in_train = (train_df["step"] == split_step).sum()
    assert boundary_rows_in_raw > 0, "split_step does not match any row in the fixture"
    assert boundary_rows_in_train == boundary_rows_in_raw
    assert (test_df["step"] == split_step).sum() == 0
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `PYTHONPATH=/tmp/aml-tutor-plan002-scratch/src python3 -m pytest tests/003_test_time_based_split.py -v`
Expected: `ModuleNotFoundError: No module named 'aml_triage.split'`.

- [ ] **Step 3: Write the scratch reference implementation**

Create `/tmp/aml-tutor-plan002-scratch/src/aml_triage/split.py`:

```python
def temporal_split(df, split_step):
    train_df = df[df["step"] <= split_step].reset_index(drop=True)
    test_df = df[df["step"] > split_step].reset_index(drop=True)
    return train_df, test_df
```

- [ ] **Step 4: Run it to confirm it passes**

Run: `PYTHONPATH=/tmp/aml-tutor-plan002-scratch/src python3 -m pytest tests/003_test_time_based_split.py -v`
Expected: 3 passed. If `test_boundary_step_lands_entirely_in_train` fails with "does not match any
row," `conftest.py`'s `split_step` fixture didn't parse `PROVENANCE.md` correctly — check Task 1's
regex against the exact line it wrote.

- [ ] **Step 5: Write the spec**

Create `aml-tutor/docs/specs/003-time-based-split.md`, four-part shape, covering:

- **Key concept:** why a random split leaks the future into training (a model trained on rows from
  after the ones it's tested on has implicitly seen outcomes that hadn't happened yet at
  decision time); the `step` column's meaning (hours since simulation start — confirm the exact
  max value against the real CSV, don't assume a cited figure); the tie rule — a `step` value is
  never divided between train and test.
- **Implementation order:** implement `temporal_split(df, split_step) -> (train_df, test_df)` in
  `src/aml_triage/split.py`; use the fixture's own boundary step (from `PROVENANCE.md`) as the
  spec's worked example.
- **Checks:** comprehension questions on why chronological order matters here specifically; the
  baked-in `uv run pytest ../aml-tutor/tests/003_test_time_based_split.py` command + validation
  block (`id: "003-time-based-split"`).
- **Pressure test:** lesson 004's features must be computable from either split independently — a
  feature that peeks across the split boundary (e.g. a rolling average spanning both) would pass
  today's tests and quietly leak tomorrow's.

- [ ] **Step 6: Update the ledger**

Row 003 → `Done` in `docs/specs/README.md`, same edit shape as Task 2 Step 6.

- [ ] **Step 7: Commit**

```bash
git add docs/specs/003-time-based-split.md tests/003_test_time_based_split.py docs/specs/README.md
git commit -m "feat: lesson 003 — time-based split"
```

---

## Task 4: Lesson 004 — feature engineering

**Files:**
- Create: `aml-tutor/docs/specs/004-feature-engineering.md`
- Create: `aml-tutor/tests/004_test_feature_engineering.py`
- Modify: `aml-tutor/docs/specs/README.md` (row 004)
- Create (scratch): `/tmp/aml-tutor-plan002-scratch/src/aml_triage/features.py`

**Interfaces:**
- Consumes: Task 2's `load_transactions` output shape.
- Produces: reference `add_features(df: pd.DataFrame) -> pd.DataFrame` — Task 6 feeds its output
  into `train_baseline`.

- [ ] **Step 1: Write the failing test**

Create `aml-tutor/tests/004_test_feature_engineering.py`:

```python
"""Baked-in check for lesson 004 — feature engineering."""
import pandas as pd

LEAKY_COLUMNS = ["nameOrig", "nameDest", "isFlaggedFraud", "type"]


def test_leaky_and_raw_type_columns_are_dropped(fixture_path):
    from aml_triage.data import load_transactions
    from aml_triage.features import add_features

    df = load_transactions(str(fixture_path))
    featured = add_features(df)
    for column in LEAKY_COLUMNS:
        assert column not in featured.columns, f"{column} should have been dropped"


def test_derived_columns_exist_and_are_numeric(fixture_path):
    from aml_triage.data import load_transactions
    from aml_triage.features import add_features

    df = load_transactions(str(fixture_path))
    featured = add_features(df)
    for column in ["orig_balance_delta", "dest_balance_delta", "is_transfer", "is_cash_out"]:
        assert column in featured.columns
        assert pd.api.types.is_numeric_dtype(featured[column])


def test_balance_deltas_are_correct_on_known_rows(fixture_path):
    from aml_triage.data import load_transactions
    from aml_triage.features import add_features

    df = load_transactions(str(fixture_path))
    featured = add_features(df)
    expected_orig_delta = df["newbalanceOrig"] - df["oldbalanceOrg"]
    expected_dest_delta = df["newbalanceDest"] - df["oldbalanceDest"]
    assert (featured["orig_balance_delta"] == expected_orig_delta).all()
    assert (featured["dest_balance_delta"] == expected_dest_delta).all()


def test_zero_balance_row_does_not_produce_a_crash_or_nan(fixture_path):
    from aml_triage.data import load_transactions
    from aml_triage.features import add_features

    df = load_transactions(str(fixture_path))
    zero_balance_mask = (df["oldbalanceOrg"] == 0) & (df["newbalanceOrig"] == 0)
    assert zero_balance_mask.sum() >= 1, "fixture should contain a zero-balance row (Task 1)"
    featured = add_features(df)
    assert not featured.loc[zero_balance_mask, "orig_balance_delta"].isna().any()
    assert (featured.loc[zero_balance_mask, "orig_balance_delta"] == 0).all()


def test_type_flags_match_the_raw_type_column(fixture_path):
    from aml_triage.data import load_transactions
    from aml_triage.features import add_features

    df = load_transactions(str(fixture_path))
    featured = add_features(df)
    assert (featured["is_transfer"] == (df["type"] == "TRANSFER").astype(int)).all()
    assert (featured["is_cash_out"] == (df["type"] == "CASH_OUT").astype(int)).all()
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `PYTHONPATH=/tmp/aml-tutor-plan002-scratch/src python3 -m pytest tests/004_test_feature_engineering.py -v`
Expected: `ModuleNotFoundError: No module named 'aml_triage.features'`.

- [ ] **Step 3: Write the scratch reference implementation**

Create `/tmp/aml-tutor-plan002-scratch/src/aml_triage/features.py`:

```python
LEAKY_COLUMNS = ["nameOrig", "nameDest", "isFlaggedFraud"]


def add_features(df):
    result = df.drop(columns=LEAKY_COLUMNS)
    result["orig_balance_delta"] = result["newbalanceOrig"] - result["oldbalanceOrg"]
    result["dest_balance_delta"] = result["newbalanceDest"] - result["oldbalanceDest"]
    result["is_transfer"] = (result["type"] == "TRANSFER").astype(int)
    result["is_cash_out"] = (result["type"] == "CASH_OUT").astype(int)
    result = result.drop(columns=["type"])
    return result
```

- [ ] **Step 4: Run it to confirm it passes**

Run: `PYTHONPATH=/tmp/aml-tutor-plan002-scratch/src python3 -m pytest tests/004_test_feature_engineering.py -v`
Expected: 5 passed.

- [ ] **Step 5: Write the spec**

Create `aml-tutor/docs/specs/004-feature-engineering.md`, four-part shape, covering:

- **Key concept:** the explicit leakage policy — `nameOrig`/`nameDest` (identifiers) and
  `isFlaggedFraud` (label-adjacent) are excluded; balance deltas are decision-time-safe. State
  plainly *why* deltas (`newbalance - oldbalance`) were chosen over a ratio
  (`newbalance / oldbalance`): a ratio divides by zero on the zero-balance rows the fixture
  deliberately contains, while a delta doesn't — name this as the reason the fixture has that row.
- **Implementation order:** implement `add_features(df) -> DataFrame` in `src/aml_triage/features.py`,
  dropping the three leaky columns plus the raw `type` column once its two flags are derived.
- **Checks:** comprehension questions on why each excluded column is excluded specifically (not
  just "some columns are excluded"); baked-in test command + validation block
  (`id: "004-feature-engineering"`).
- **Pressure test:** lesson 005's imbalance weight and lesson 006's training both consume this
  function's output directly — a feature that's `NaN` or non-numeric here fails two lessons later
  as a confusing training error, not a features error.

- [ ] **Step 6: Update the ledger**

Row 004 → `Done`.

- [ ] **Step 7: Commit**

```bash
git add docs/specs/004-feature-engineering.md tests/004_test_feature_engineering.py docs/specs/README.md
git commit -m "feat: lesson 004 — feature engineering"
```

---

## Task 5: Lesson 005 — class imbalance

**Files:**
- Create: `aml-tutor/docs/specs/005-class-imbalance.md`
- Create: `aml-tutor/tests/005_test_class_imbalance.py`
- Modify: `aml-tutor/docs/specs/README.md` (row 005)
- Create (scratch): `/tmp/aml-tutor-plan002-scratch/src/aml_triage/imbalance.py`

**Interfaces:**
- Consumes: Task 2's `load_transactions`, Task 3's `temporal_split` (for the train-split labels).
- Produces: reference `compute_scale_pos_weight(y: pd.Series) -> float` — Task 6 passes its result
  into `train_baseline`.

- [ ] **Step 1: Write the failing test**

Create `aml-tutor/tests/005_test_class_imbalance.py`. `fixture_path`/`split_step` come from
`tests/conftest.py` (Task 1):

```python
"""Baked-in check for lesson 005 — class imbalance."""
import pytest


def test_weight_equals_negative_over_positive_on_the_train_split(fixture_path, split_step):
    from aml_triage.data import load_transactions
    from aml_triage.split import temporal_split
    from aml_triage.imbalance import compute_scale_pos_weight

    df = load_transactions(str(fixture_path))
    train_df, _ = temporal_split(df, split_step)
    y_train = train_df["isFraud"]
    weight = compute_scale_pos_weight(y_train)
    expected = (y_train == 0).sum() / (y_train == 1).sum()
    assert weight == pytest.approx(expected)
    assert weight > 1, "fraud should be the minority class in the train split"


def test_raises_a_clear_error_with_no_positive_examples():
    from aml_triage.imbalance import compute_scale_pos_weight
    import pandas as pd

    all_negative = pd.Series([0, 0, 0])
    with pytest.raises(ValueError):
        compute_scale_pos_weight(all_negative)
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `PYTHONPATH=/tmp/aml-tutor-plan002-scratch/src python3 -m pytest tests/005_test_class_imbalance.py -v`
Expected: `ModuleNotFoundError: No module named 'aml_triage.imbalance'`.

- [ ] **Step 3: Write the scratch reference implementation**

Create `/tmp/aml-tutor-plan002-scratch/src/aml_triage/imbalance.py`:

```python
def compute_scale_pos_weight(y):
    positive = int((y == 1).sum())
    negative = int((y == 0).sum())
    if positive == 0:
        raise ValueError("no positive examples; cannot compute scale_pos_weight")
    return negative / positive
```

- [ ] **Step 4: Run it to confirm it passes**

Run: `PYTHONPATH=/tmp/aml-tutor-plan002-scratch/src python3 -m pytest tests/005_test_class_imbalance.py -v`
Expected: 2 passed.

- [ ] **Step 5: Write the spec**

Create `aml-tutor/docs/specs/005-class-imbalance.md`, four-part shape, covering:

- **Key concept:** what `scale_pos_weight` does inside XGBoost (up-weights the minority class's
  contribution to the loss), and the tradeoff — too aggressive a weight over-predicts fraud
  (too many false alarms), too weak under-predicts it (fraud slips through) — not just "apply this
  knob."
- **Implementation order:** implement `compute_scale_pos_weight(y) -> float` in
  `src/aml_triage/imbalance.py`, computed on the *train* split's labels only (never the test
  split — using test-split class balance to weight training would itself be a leak).
- **Checks:** comprehension questions on why this is computed from the train split specifically;
  baked-in test command + validation block (`id: "005-class-imbalance"`).
- **Pressure test:** lesson 006 passes this exact weight into `train_baseline` — a weight computed
  on the wrong split (e.g. the whole fixture instead of just train) would silently bias lesson
  006's model without either lesson's own test catching it, since each only tests its own function
  in isolation.

- [ ] **Step 6: Update the ledger**

Row 005 → `Done`.

- [ ] **Step 7: Commit**

```bash
git add docs/specs/005-class-imbalance.md tests/005_test_class_imbalance.py docs/specs/README.md
git commit -m "feat: lesson 005 — class imbalance"
```

---

## Task 6: Lesson 006 — train the baseline

**Files:**
- Create: `aml-tutor/docs/specs/006-train-the-baseline.md`
- Create: `aml-tutor/tests/006_test_train_baseline.py`
- Modify: `aml-tutor/docs/specs/README.md` (row 006)
- Create (scratch): `/tmp/aml-tutor-plan002-scratch/src/aml_triage/model.py`

**Interfaces:**
- Consumes: Task 3's `temporal_split`, Task 4's `add_features`, Task 5's `compute_scale_pos_weight`.
- Produces: reference `train_baseline(X_train, y_train, weight) -> xgb.XGBClassifier` — Task 7
  scores its output on the test split.

- [ ] **Step 1: Write the failing test**

Create `aml-tutor/tests/006_test_train_baseline.py`. `fixture_path`/`split_step` come from
`tests/conftest.py` (Task 1):

```python
"""Baked-in check for lesson 006 — train the baseline."""


def _prepare_splits(fixture_path, split_step):
    from aml_triage.data import load_transactions
    from aml_triage.split import temporal_split
    from aml_triage.features import add_features

    df = load_transactions(str(fixture_path))
    train_df, test_df = temporal_split(df, split_step)
    train_featured = add_features(train_df)
    test_featured = add_features(test_df)
    y_train = train_featured.pop("isFraud")
    y_test = test_featured.pop("isFraud")
    return train_featured, y_train, test_featured, y_test


def test_model_fits_and_predicts_valid_probabilities(fixture_path, split_step):
    from aml_triage.imbalance import compute_scale_pos_weight
    from aml_triage.model import train_baseline

    X_train, y_train, X_test, y_test = _prepare_splits(fixture_path, split_step)
    weight = compute_scale_pos_weight(y_train)
    model = train_baseline(X_train, y_train, weight)
    scores = model.predict_proba(X_test)[:, 1]
    assert len(scores) == len(X_test)
    assert (scores >= 0).all() and (scores <= 1).all()


def test_model_is_at_least_somewhat_better_than_chance(fixture_path, split_step):
    from sklearn.metrics import roc_auc_score

    from aml_triage.imbalance import compute_scale_pos_weight
    from aml_triage.model import train_baseline

    X_train, y_train, X_test, y_test = _prepare_splits(fixture_path, split_step)
    weight = compute_scale_pos_weight(y_train)
    model = train_baseline(X_train, y_train, weight)
    scores = model.predict_proba(X_test)[:, 1]
    assert roc_auc_score(y_test, scores) > 0.5
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `PYTHONPATH=/tmp/aml-tutor-plan002-scratch/src python3 -m pytest tests/006_test_train_baseline.py -v`
Expected: `ModuleNotFoundError: No module named 'aml_triage.model'`.

- [ ] **Step 3: Write the scratch reference implementation**

Create `/tmp/aml-tutor-plan002-scratch/src/aml_triage/model.py`:

```python
import xgboost as xgb


def train_baseline(X_train, y_train, weight):
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=weight,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model
```

- [ ] **Step 4: Run it to confirm it passes**

Run: `PYTHONPATH=/tmp/aml-tutor-plan002-scratch/src python3 -m pytest tests/006_test_train_baseline.py -v`
Expected: 2 passed. If `test_model_is_at_least_somewhat_better_than_chance` fails, the fixture's
fraud/non-fraud mix from Task 1 may be too thin for any signal — revisit `STEP_RANGE`/
`NON_FRAUD_SAMPLE_SIZE` in `carve_fixture.py` and re-run Task 1.

- [ ] **Step 5: Write the spec**

Create `aml-tutor/docs/specs/006-train-the-baseline.md`, four-part shape, covering:

- **Key concept:** why gradient boosting over a neural net for tabular, mixed-type transaction
  data at this size — no need for representation learning on a handful of numeric/categorical
  columns, and boosted trees handle the class imbalance + interpretability tradeoff well for a
  first baseline.
- **Implementation order:** implement `train_baseline(X_train, y_train, weight) -> fitted XGBoost`
  in `src/aml_triage/model.py`, using `scale_pos_weight=weight` from lesson 005's function.
- **Checks:** comprehension questions on why `scale_pos_weight` goes here rather than in the
  features step; baked-in test command + validation block (`id: "006-train-the-baseline"`).
- **Pressure test:** lesson 007 scores this exact model's `predict_proba` output — a model that
  only predicts one class (e.g. from a training bug) would still "fit" today but produce a
  degenerate PR curve next lesson.

- [ ] **Step 6: Update the ledger**

Row 006 → `Done`.

- [ ] **Step 7: Commit**

```bash
git add docs/specs/006-train-the-baseline.md tests/006_test_train_baseline.py docs/specs/README.md
git commit -m "feat: lesson 006 — train the baseline"
```

---

## Task 7: Lesson 007 — evaluation and threshold

**Files:**
- Create: `aml-tutor/docs/specs/007-evaluation-and-threshold.md`
- Create: `aml-tutor/tests/007_test_evaluation.py`
- Modify: `aml-tutor/docs/specs/README.md` (row 007, last remaining `Todo`)
- Create (scratch): `/tmp/aml-tutor-plan002-scratch/src/aml_triage/evaluate.py`

**Interfaces:**
- Consumes: fixed, hand-supplied `y_true`/`scores` arrays (not Task 6's live model — see rationale
  below).
- Produces: reference `report(y_true, scores, objective: dict) -> dict` with keys `precision`,
  `recall`, `pr_auc`, `threshold`. `objective` shape: `{"min_precision": 0.90}`.

**Why this test uses fixed arrays, not a freshly-trained model.** Exact precision/recall/PR-AUC
values are only meaningful to assert against a known, static input — a live XGBoost model's scores
can shift slightly across library versions or seeds (Task 6's test deliberately avoids this by
asserting only that scores fall in `[0, 1]` and beat chance, never exact values). This test instead
checks `report`'s own math independent of training reproducibility.

- [ ] **Step 1: Write the failing test**

Create `aml-tutor/tests/007_test_evaluation.py`:

```python
"""Baked-in check for lesson 007 — evaluation and threshold.

Uses fixed y_true/scores arrays, not a freshly-trained model — see the plan's Task 7 rationale:
exact metric values are only meaningful against a known, static input.
"""
import pytest

Y_TRUE = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
SCORES = [0.05, 0.10, 0.12, 0.20, 0.30, 0.55, 0.60, 0.65, 0.80, 0.95]
OBJECTIVE = {"min_precision": 0.90}


def test_report_returns_the_expected_keys():
    from aml_triage.evaluate import report

    result = report(Y_TRUE, SCORES, OBJECTIVE)
    assert set(result.keys()) == {"precision", "recall", "pr_auc", "threshold"}


def test_threshold_satisfies_the_min_precision_objective():
    from aml_triage.evaluate import report

    result = report(Y_TRUE, SCORES, OBJECTIVE)
    assert result["precision"] >= OBJECTIVE["min_precision"]


def test_recall_is_maximized_among_thresholds_meeting_the_objective():
    from sklearn.metrics import precision_recall_curve

    from aml_triage.evaluate import report

    result = report(Y_TRUE, SCORES, OBJECTIVE)
    precision, recall, _ = precision_recall_curve(Y_TRUE, SCORES)
    best_possible_recall = max(
        (r for p, r in zip(precision, recall) if p >= OBJECTIVE["min_precision"]),
        default=None,
    )
    assert best_possible_recall is not None, "test fixture data doesn't reach min_precision at all"
    assert result["recall"] == pytest.approx(best_possible_recall)


def test_raises_when_no_threshold_reaches_the_objective():
    from aml_triage.evaluate import report

    impossible_objective = {"min_precision": 0.9999}
    with pytest.raises(ValueError):
        report(Y_TRUE, SCORES, impossible_objective)
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `PYTHONPATH=/tmp/aml-tutor-plan002-scratch/src python3 -m pytest tests/007_test_evaluation.py -v`
Expected: `ModuleNotFoundError: No module named 'aml_triage.evaluate'`.

- [ ] **Step 3: Write the scratch reference implementation**

Create `/tmp/aml-tutor-plan002-scratch/src/aml_triage/evaluate.py`:

```python
from sklearn.metrics import average_precision_score, precision_recall_curve


def report(y_true, scores, objective):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    min_precision = objective["min_precision"]
    # precision_recall_curve returns one more precision/recall point than thresholds
    # (the last point has no corresponding threshold); only index where a threshold exists.
    candidates = [i for i in range(len(thresholds)) if precision[i] >= min_precision]
    if not candidates:
        raise ValueError(f"no threshold reaches precision >= {min_precision}")
    best = max(candidates, key=lambda i: recall[i])
    return {
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "pr_auc": float(pr_auc),
        "threshold": float(thresholds[best]),
    }
```

- [ ] **Step 4: Run it to confirm it passes**

Run: `PYTHONPATH=/tmp/aml-tutor-plan002-scratch/src python3 -m pytest tests/007_test_evaluation.py -v`
Expected: 4 passed.

- [ ] **Step 5: Write the spec**

Create `aml-tutor/docs/specs/007-evaluation-and-threshold.md`, four-part shape, covering:

- **Key concept:** precision/recall/PR-AUC explained for fraud detection specifically; the pinned
  objective — "maximize recall subject to precision ≥ 90%" — and why that framing suits catching
  fraud while keeping false alarms operationally tolerable.
- **Implementation order:** implement `report(y_true, scores, objective) -> dict` in
  `src/aml_triage/evaluate.py` (a pure function — it returns values, it does not write files);
  then, as a separate step, call it with the lesson 006 model's test-split scores and write the
  returned dict to `aml-triage/reports/phase1_report.json` (e.g. via
  `json.dump(result, open("reports/phase1_report.json", "w"), indent=2)`) — this file is the
  concrete Phase 1 deliverable, not the function's return value alone.
- **Checks:** comprehension questions on why maximizing recall *subject to* a precision floor
  differs from just picking the threshold with the best F1; baked-in test command + validation
  block (`id: "007-evaluation-and-threshold"`).
- **Closing note (last lesson):** state plainly that Phase 1 is complete and defensible even if
  nothing further is built, mirroring the tutor's own closing-lesson behavior from iteration 001's
  system prompt.
- **Doer fallback note:** if asked to do this step, the doer can call the already-implemented
  `report` function and write the JSON file (both are within its write-to-`aml-triage` boundary;
  neither needs shell access).

- [ ] **Step 6: Update the ledger**

Row 007 → `Done`. After this step, `docs/specs/README.md` has no remaining `Todo` rows.

- [ ] **Step 7: Commit**

```bash
git add docs/specs/007-evaluation-and-threshold.md tests/007_test_evaluation.py docs/specs/README.md
git commit -m "feat: lesson 007 — evaluation and threshold"
```

---

## Task 8: Full dry run and scratch cleanup

**Files:** none created or modified in `aml-tutor`; this task verifies Tasks 1-7's output together
and removes the scratch workspace.

**Interfaces:**
- Consumes: every module under `/tmp/aml-tutor-plan002-scratch/src/aml_triage/` built by Tasks 2-7.
- Produces: nothing persisted — this is a verification-only task.

- [ ] **Step 1: Run every baked-in test together against the scratch package**

Run:

```bash
PYTHONPATH=/tmp/aml-tutor-plan002-scratch/src python3 -m pytest tests/002_test_data_loading.py \
  tests/003_test_time_based_split.py tests/004_test_feature_engineering.py \
  tests/005_test_class_imbalance.py tests/006_test_train_baseline.py \
  tests/007_test_evaluation.py -v
```

Expected: all pass together, not just individually — this catches a signature mismatch between
tasks (e.g. Task 4 renaming a column Task 6 expects) that per-task runs in isolation would miss.

- [ ] **Step 2: Confirm the real `aml-triage` is untouched**

Run: `find /Users/pebarna/projects/aml-triage -mindepth 1 -not -path '*/.git/*'`
Expected: only `.gitignore` — nothing from the scratch package leaked into the real sibling repo.
This confirms the tutorial still starts from scratch for an actual student.

- [ ] **Step 3: Confirm `npm run check` still passes**

Run (from `aml-tutor`'s root): `npm run check`
Expected: same clean result as iteration 001 (119+ tests, `tsc --noEmit` clean) — this iteration
touched no engine code, so this is a regression check, not new coverage.

- [ ] **Step 4: Discard the scratch workspace**

Run: confirm with the user before deleting anything outside the repo, then remove
`/tmp/aml-tutor-plan002-scratch/`. It was never a deliverable — a real student's `aml-triage`
should never have received these files, and Step 2 already confirmed it didn't.

- [ ] **Step 5: Final commit**

```bash
git add docs/iterations/002-classifier-lessons/plan.md
git commit -m "docs: iteration 002 plan"
git log --oneline -10
```

Expected: a clean, readable commit sequence — Task 0's iteration-001 catch-up, then one commit per
lesson task, ending with this plan document itself.

---

## Definition of done

- The fixture exists, is committed with its provenance note (including the split-boundary `step`
  value), and Task 1 Step 3's spot-check passed.
- All six remaining lesson specs exist, are listed in the ledger as `Done`, and each has a passing
  baked-in test — verified both individually (Tasks 2-7) and together (Task 8 Step 1).
- `docs/specs/README.md`'s ledger has no remaining `Todo` rows and no stale "blocked on..." text.
- Lesson 002's data-acquisition/sampling command is written out exactly, not just described.
- Task 8's dry run confirms the real `aml-triage` sibling repo is untouched by anything this plan
  built, and that `npm run check` still passes.

## Open risks / watch items (carried from brainstorming, still relevant during execution)

- **Fraud scarcity in the fixture** — if Task 6's "better than chance" check or Task 1's
  fraud-type check fails, widen `STEP_RANGE`/`NON_FRAUD_SAMPLE_SIZE` in `carve_fixture.py`.
- **PaySim license/redistribution** — confirm the committed fixture's size is within what the
  dataset's license permits before Task 1's commit.
- **Sampling reproducibility** — Task 2's documented sampling command must produce the same row
  count on repeat runs with the same seed; this isn't asserted by a baked-in test (it's an
  environment-prep step, per lesson 001's precedent), so verify it manually once during Task 2.
- **Training reproducibility** — Task 6 pins `random_state=42`; if XGBoost's version isn't also
  pinned in a real `aml-triage/pyproject.toml` during Task 8's dry run, note the installed version
  in the spec rather than assuming it.
- **`aml-triage/SEED.md`** still doesn't exist; not a blocker for this plan (all needed detail is
  in `ARCHITECTURE.md` §6 and this plan), but remains an open documentation gap from iteration 001.
