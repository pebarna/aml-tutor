# Iteration 003 — Triage-agent and eval-harness lessons (Parts 2 & 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.
>
> **Plan location note:** this repo files iteration plans under `docs/iterations/NNN-name/plan.md`
> (see iterations 001–002), not `docs/superpowers/plans/` — that established convention overrides
> `writing-plans`' generic default.

**Goal:** Author lessons 008–015 — Phase 2 (the RAG triage agent) and Phase 3 (the eval harness) of
`aml-triage/SEED.md` — as eight new lesson specs + baked-in pytest tests, in one iteration (per
SEED.md's own "Weekend 2: RAG + triage agent + triage eval" grouping), choosing at each design point
the option closest to what a well-established, regulated fintech would actually run in production —
so the ledger reads fully `Done` for all three phases and the tutorial doubles as a transferable
"here's the equivalent at a real ML/compliance team" story, not just a working demo.

**Architecture:** Pure content on top of iterations 001–002's unchanged engine — no new
`tutorial-engine` capability is required. Each lesson task produces a spec (`docs/specs/NNN-*.md`)
and a baked-in test (`aml-tutor/tests/NNN_test_*.py`). Two departures from iterations 001–002's
pattern, both deliberate:

1. Lessons 011 (the agent) and 014 (LLM-as-judge) call a real LLM in production use; both take
   their LLM client as an injectable parameter so the baked-in test never makes a network call —
   the reproducibility principle lesson 007 already established for exact-value assertions,
   extended to external-service boundaries. Lesson 010 (structured decisions) is a prerequisite for
   011, but is itself pure functions — no SDK import, no client, no network — see its own Task for
   why that split exists.
2. Lesson 012 (the eval set) is the first lesson in this tutorial whose artifact is a genuine human
   judgment call, not a derivable function output — the student hand-labels real cases themselves.
   Its baked-in test checks structure only (schema, count, allowed values); the judgment itself is
   checked by the tutor in conversation, the same split lesson 007's comprehension questions already
   draw between "does it work" and "do you understand it," now applied to "is this label defensible."

**Tech Stack:** Everything already in `aml-triage` (pandas, scikit-learn, xgboost, pytest) plus two
new additions, each introduced exactly where first needed: `sentence-transformers` (lesson 009, the
embeddings half of hybrid retrieval) and `anthropic` (lesson 011, the first real LLM call).

## Global Constraints

- Lesson-to-code API (extends `docs/ARCHITECTURE.md` §6):
  `retrieval.top_k_typologies(query, k=3, corpus_path=None) -> list[dict]`,
  `retrieval.top_k_typologies_hybrid(query, k=3, corpus_path=None, alpha=0.5) -> list[dict]`,
  `triage_schema.build_prompt(transaction, classifier_score, retrieved) -> str`,
  `triage_schema.parse_triage_decision(tool_input, known_typology_ids) -> dict`,
  `triage.triage(transaction, classifier_score, *, client=None, k=3, corpus_path=None, alpha=0.5) -> dict`,
  `eval.load_eval_set(path) -> list[dict]`,
  `eval.deterministic_score(case, result) -> dict`,
  `eval.llm_judge_score(result, *, client=None) -> dict` (reads `result["retrieved"]` internally —
  see lesson 011's return shape and lesson 014's redesign note),
  `eval.report(cases, results, deterministic_scores, judge_scores) -> dict`.
  `corpus_path=None` resolves to `"data/typologies.json"`, relative to the process's working
  directory — which is `aml-triage`'s own root under every real validation command and under this
  plan's own scratch-verification steps, but is *not* pinned to the repo root independent of cwd.
  Every retrieval-touching baked-in test passes `corpus_path` explicitly instead of relying on that
  default, so those tests' results don't depend on cwd. Lesson 012 is the one deliberate exception
  to cwd-independence in this whole iteration: its test reads a bare relative path
  (`"data/triage_eval_set.jsonl"`) on purpose, because the artifact it checks is meant to live in the
  student's real `aml-triage/data/`, not a committed `aml-tutor` fixture — see Task 6.
- Every baked-in test that touches an LLM call injects a fake client through the `client=`
  parameter — no baked-in test may require `ANTHROPIC_API_KEY` or a call to a hosted LLM API to
  pass. This does not cover lesson 009's local embedding model: its first run downloads
  `sentence-transformers` weights and needs network for that one-time download, which is a
  different thing from calling a hosted LLM API and is documented as such in lesson 009's own spec.
- New dependencies land in `aml-triage/pyproject.toml` in the lesson that first needs them, never
  earlier — same incremental-dependency discipline as lessons 001–007.
- Every new spec follows `docs/specs/001-project-setup.md`'s four-part shape (Key concept /
  Implementation order / Checks / Pressure test), the `json validation` fenced-block convention, a
  doer-fallback note wherever relevant, and — new in this iteration — a short **"At a regulated
  shop"** paragraph naming the production-scale equivalent of whatever the lesson just built.
- Scratch workspace: every task's throwaway reference implementation lives under
  `/tmp/aml-tutor-plan003-scratch/src/aml_triage/`, discarded in the final task.
- Fixture path convention unchanged: every fixture resolves relative to the test file
  (`Path(__file__).parent / "fixtures" / ...`), and every real validation command runs with
  `cwd=../aml-triage`.

---

## Why these choices (decisions made during brainstorming — read this before Task 1)

This iteration's design point, agreed with the user before writing began: at every fork, choose
whatever is closest to what a well-established, *regulated* fintech (a bank, card network, or AML
RegTech vendor) actually runs in production — not just whatever is simplest to teach — because the
explicit goal is a transfer story ("here's the equivalent at a real ML team"), not a demo.

- **Retrieval is hybrid, built in two lessons (008 then 009), not embeddings-only.** Pure keyword
  search misses paraphrase; pure dense-embedding search is known to underperform on regulatory text
  specifically (exact dollar thresholds, defined terms, citations) — which is why mature compliance
  retrieval systems run **hybrid** search: BM25/keyword plus dense vectors, blended or reranked. This
  tutorial builds that same shape in miniature: lesson 008 is the keyword half (TF-IDF + cosine
  similarity via scikit-learn, already a dependency, zero model download, exact reproducible
  ranking), lesson 009 adds a local embeddings model and blends the two scores. **At a regulated
  shop**, the keyword half runs on OpenSearch/Elasticsearch or Postgres+pgvector — reusing
  already-audited infrastructure — and the embedding half comes from a model already inside an
  approved enterprise contract (Azure OpenAI/Bedrock/Vertex embeddings) or self-hosted inside the
  bank's own VPC once the text is AML/SAR-adjacent and can't leave the compliance boundary to a new
  subprocessor at all. A dedicated vector-DB SaaS (Pinecone, Weaviate) is comparatively rare at a
  bank specifically because onboarding a brand-new vendor is a slow risk-review process — this
  tutorial's own in-memory approach is architecturally closer to the "reuse what's already approved"
  instinct than a vector-DB integration would have been.
- **Generation is the raw `anthropic` SDK + forced tool-use, not a framework.** Regulated shops lean
  *away* from heavy agent frameworks for this kind of decision precisely because a framework's
  internals are harder to audit line-by-line than a thin, direct SDK call a compliance/model-risk
  reviewer can read start to finish. Forcing structured output at the model boundary
  (`tool_choice={"type": "tool", ...}`) instead of parsing free text is the single most transferable
  idea in Part 2: any production agentic feature that returns a decision needs a validated, typed
  contract at the LLM boundary. **At a regulated shop**, the actual call would go through Bedrock or
  Azure OpenAI rather than a public API directly (for the enterprise DPA and VPC boundary), and its
  output would be a draft recommendation routed to a human analyst queue, never an auto-executed
  decision — human-in-the-loop is close to a hard requirement here, not a nice-to-have. This
  tutorial keeps calling the Anthropic API directly, since standing up a Bedrock/Azure account is out
  of scope for a personal weekend project — that gap is named explicitly in lesson 011's spec rather
  than left implicit.
- **The LLM client is dependency-injected (`client=None`) everywhere it's used.** This is what makes
  lessons 011/014 unit-testable without network access, and it is the concrete pattern that answers
  "how do you test an LLM feature" in an interview — a general software-engineering seam (external
  service as an injected dependency), not something specific to LLMs.
- **The eval set is genuinely hand-labeled by the student (lesson 012), not generated from a rule.**
  This is the one place "closest to production practice" is the *harder* choice, not the lighter
  one: regulated institutions operate under model-risk-management regimes (SR 11-7-style in US
  banking) that require a documented, SME-labeled golden set, periodically refreshed by actual
  humans — a mechanically generated label set could never substitute for that, and more importantly
  can't produce genuine borderline/`monitor`-worthy judgment calls, which is exactly the kind of case
  a real analyst's labeling would surface. A script (Task 1) still deterministically selects *which*
  16 transactions become labeling candidates — reproducible candidate selection, human judgment on
  top — but does not assign them a label. Lesson 012's baked-in test checks the resulting file's
  shape only; the tutor's comprehension questions probe the reasoning behind specific labels, the
  same "code checked by pytest, understanding checked in conversation" split the tutorial has used
  since lesson 001.
- **LLM-as-judge (lesson 014) is a supplementary signal, not a replacement for lesson 012's human
  labels.** Say this explicitly in both specs: in a regulated setting, LLM-as-judge is used as a
  fast, scalable signal for ongoing monitoring between human-labeled reviews, never as the
  validation of record.
- **The eval harness stays hand-rolled, saved as a JSON report — not a platform (Braintrust,
  LangSmith).** A few dozen lines of pure Python computing the agreement rate is more legible and
  more defensible in an interview than a platform integration, and it continues the exact pattern
  lesson 007 already set with `evaluate.report(...) -> dict` saved to `reports/phase1_report.json`.
  **At a regulated shop**, the equivalent is usually a bespoke internal harness or something bolted
  onto existing MLOps tooling (e.g. MLflow) rather than a purpose-built eval SaaS — same
  new-vendor-risk logic as the retrieval layer's preference for already-approved infrastructure.
- **No new `tutorial-engine` capability is required.** The doer's boundary (read both repos, write
  only `aml-triage`, no Bash) is unchanged; validation commands are still `uv run pytest <path>`
  with `cwd=../aml-triage`; no lesson needs the validation runner's spawn environment widened,
  because no baked-in test makes a network call. Lesson 012 is the one lesson where the doer
  genuinely cannot substitute for the student at all (see its spec) — that is a content fact about
  what the lesson teaches, not an engine change.
- **Scope is bigger than a literal one-weekend reading of SEED.md's time budget.** Eight lessons,
  including a real hand-labeling exercise, is more than "RAG + triage agent + triage eval" reads as
  a single weekend. That's an accepted tradeoff for hitting the "closest to prod" choices the user
  asked for over a lighter, faster-to-author scope — flagged here and again in "Open risks" below.
- **Typology corpus is original teaching text; eval-set candidates are script-generated but
  unlabeled.** Unlike the PaySim CSV (a licensed, redistribution-permitted slice), there is no
  upstream dataset to carve from for typologies. Task 1 authors six short, original paraphrases of
  well-known AML red-flag patterns (not verbatim FATF text) and a script that deterministically
  *selects* 16 real PaySim rows as labeling candidates — it does not label them.

---

## Task 0: Place `aml-triage/SEED.md`

`aml-tutor/docs/ARCHITECTURE.md`'s "Known gap" section and iteration 002's open risks both flag that
`aml-triage/SEED.md` — the file `aml-tutor/README.md` links to as "Phase 1 of the project described
in `aml-triage/SEED.md`" — has never actually existed in the sibling repo. It now does, supplied by
the user at `/Users/pebarna/projects/temp/aml-triage/SEED.md` (all three phases). Placing it closes
that standing gap before any new lesson content links to it.

**Files:**
- Create: `/Users/pebarna/projects/aml-triage/SEED.md` (copy of the supplied file)
- Modify: `/Users/pebarna/projects/aml-triage/.gitignore` (remove the `SEED.md` line so the file
  Task 0 adds is actually trackable)
- Modify: `aml-tutor/docs/ARCHITECTURE.md` (delete the "Known gap" section — both its listed gaps are
  closed: the fixture by iteration 002, the SEED.md by this task)

**Interfaces:** None — this task touches no code.

- [ ] **Step 1: Copy the SEED.md into the sibling repo**

```bash
cp /Users/pebarna/projects/temp/aml-triage/SEED.md /Users/pebarna/projects/aml-triage/SEED.md
```

- [ ] **Step 2: Confirm it reads correctly from `aml-tutor`'s own relative link**

Run: `cat ../aml-triage/SEED.md` from `aml-tutor`'s root. Expected: the three-phase document, ending
with "## Next action when picked back up."

- [ ] **Step 3: Remove the now-closed "Known gap" section from `ARCHITECTURE.md`**

Delete the `## Known gap (as of this writing)` section.

- [ ] **Step 4: `aml-triage/.gitignore` ignores `SEED.md` — remove that line before committing**

Verify first: `git -C /Users/pebarna/projects/aml-triage check-ignore -v SEED.md`. If it prints a
match (it will — `aml-triage/.gitignore` line 2 is literally `SEED.md`, left over from before this
file existed), a plain `git add SEED.md` fails and the `&&`-chained commit silently never runs. Fix
the `.gitignore`, not the `git add`:

```bash
cd /Users/pebarna/projects/aml-triage
sed -i '' '/^SEED\.md$/d' .gitignore
git add .gitignore SEED.md
git commit -m "docs: add project SEED (phases 1-3); un-ignore SEED.md"
```

- [ ] **Step 5: Commit the `aml-tutor` side**

```bash
cd /Users/pebarna/projects/aml-tutor && git add docs/ARCHITECTURE.md && git commit -m "docs: close the SEED.md known-gap now that aml-triage/SEED.md exists"
```

---

## Task 1: Fixture foundation — typology corpus and eval-set candidates

**Files:**
- Create: `aml-tutor/tests/fixtures/typologies.json`
- Create: `aml-tutor/tests/fixtures/PROVENANCE_TYPOLOGIES.md`
- Create: `aml-tutor/scripts/build_eval_candidates.py`
- Create (generated by running the script, then committed): `aml-tutor/tests/fixtures/triage_eval_candidates.jsonl`
- Modify: `aml-tutor/tests/conftest.py` (add `typologies_path` and `eval_candidates_path` fixtures)

**Interfaces:**
- Produces: `tests/fixtures/typologies.json` (6 entries, keys `id`/`title`/`text`) and
  `tests/fixtures/triage_eval_candidates.jsonl` (16 lines, keys `transaction`/`classifier_score`,
  deliberately **no label** — lesson 012 adds that) — every later task consumes one or both.

- [ ] **Step 1: Write the typology corpus**

Create `aml-tutor/tests/fixtures/typologies.json`:

```json
[
  {
    "id": "TY-001",
    "title": "Structuring / smurfing",
    "text": "A large sum is broken into several smaller transfers, often just under a reporting or scrutiny threshold, to avoid drawing attention to the total amount actually moved."
  },
  {
    "id": "TY-002",
    "title": "Mule account layering",
    "text": "Funds are routed through an intermediary account controlled by a third party shortly before being cashed out or moved onward, adding a layer between the original source and the eventual destination of the money."
  },
  {
    "id": "TY-003",
    "title": "Rapid pass-through",
    "text": "An account receives funds and moves them onward, by transfer or cash-out, within a very short window, leaving little or no residual balance and little time for the transaction to be reviewed before the money is gone."
  },
  {
    "id": "TY-004",
    "title": "Round-tripping",
    "text": "Funds are sent out through one or more intermediary accounts and eventually return to an account associated with the original sender, giving the appearance of unrelated activity while the money never really left the sender's control."
  },
  {
    "id": "TY-005",
    "title": "Cash-out concentration",
    "text": "A disproportionate share of an account's outbound activity is cash withdrawal rather than transfer or payment, consistent with converting proceeds into a form that is hard to trace once it leaves the financial system."
  },
  {
    "id": "TY-006",
    "title": "Dormant account reactivation",
    "text": "An account with little or no prior activity suddenly originates or receives a large transaction, consistent with an account that sat unused until it was needed for a single laundering event."
  }
]
```

- [ ] **Step 2: Document the corpus's provenance**

Create `aml-tutor/tests/fixtures/PROVENANCE_TYPOLOGIES.md`:

```markdown
# typologies.json provenance

- Original teaching text, written for this tutorial — not a verbatim reproduction of any single
  source. The six patterns paraphrase widely-documented, publicly discussed AML red-flag categories
  (structuring, layering through intermediary/"mule" accounts, rapid pass-through of funds,
  round-tripping, cash-out concentration, and dormant-account reactivation) that appear, described in
  their own words, across public regulatory guidance (e.g. FATF red-flag indicator publications) and
  public compliance-training material.
- Deliberately short and general: this is a teaching-scale corpus for retrieval mechanics (lessons
  008-009), not a comprehensive AML typology reference. A real triage-agent deployment would draw on
  a much larger, sourced, and versioned typology library.
```

- [ ] **Step 3: Write the eval-candidate generation script**

Create `aml-tutor/scripts/build_eval_candidates.py`:

```python
#!/usr/bin/env python3
"""One-time script: select unlabeled triage eval-set candidates from the committed PaySim fixture.

Usage: python scripts/build_eval_candidates.py

This produces CANDIDATES, not labels -- lesson 012 has the student hand-label each one themselves.
classifier_score is a documented stand-in (0.95/0.05 by the source row's isFraud flag), not a real
Phase 1 model inference -- there is no persisted Phase 1 model artifact to reload here.
"""
import json
from pathlib import Path

import pandas as pd

SEED = 20260823
N_PER_CLASS = 8
FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "paysim_fixture.csv"
OUTPUT_PATH = FIXTURE_PATH.parent / "triage_eval_candidates.jsonl"

TRANSACTION_COLUMNS = [
    "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
]


def build() -> None:
    df = pd.read_csv(FIXTURE_PATH)
    fraud = df[df["isFraud"] == 1]
    non_fraud = df[df["isFraud"] == 0]

    if len(fraud) < N_PER_CLASS or len(non_fraud) < N_PER_CLASS:
        raise SystemExit(
            f"need >= {N_PER_CLASS} rows in each class, found {len(fraud)} fraud / "
            f"{len(non_fraud)} non-fraud; lower N_PER_CLASS or widen the source fixture, don't let "
            "this silently produce fewer than 16 candidates"
        )

    fraud_sample = fraud.sample(n=N_PER_CLASS, random_state=SEED)
    non_fraud_sample = non_fraud.sample(n=N_PER_CLASS, random_state=SEED)

    rows = []
    for _, row in pd.concat([fraud_sample, non_fraud_sample]).iterrows():
        is_fraud = bool(row["isFraud"])
        rows.append({
            "transaction": {col: row[col] for col in TRANSACTION_COLUMNS},
            "classifier_score": 0.95 if is_fraud else 0.05,
        })

    with OUTPUT_PATH.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(rows)} candidates ({n} fraud-flagged, {n} not) to {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
```

- [ ] **Step 4: Run it**

From `aml-tutor`'s own root (the script's paths are relative to it):

```bash
python3 scripts/build_eval_candidates.py
```

Expected: prints `Wrote 16 candidates (8 fraud-flagged, 8 not) to .../triage_eval_candidates.jsonl`.
If it raises `SystemExit`, stop and check `tests/fixtures/PROVENANCE.md`'s fraud-row count rather
than loosening the `n = min(...)` guard.

- [ ] **Step 5: Spot-check the output**

```bash
python3 -c "
import json
rows = [json.loads(l) for l in open('tests/fixtures/triage_eval_candidates.jsonl')]
print('rows:', len(rows))
print('all have exactly transaction+classifier_score, no label:',
      all(set(r) == {'transaction', 'classifier_score'} for r in rows))
"
```

Expected: `rows: 16`, `True`.

- [ ] **Step 6: Extend the shared pytest fixtures**

Modify `aml-tutor/tests/conftest.py`, adding two fixtures:

```python
@pytest.fixture(scope="session")
def typologies_path():
    return FIXTURES_DIR / "typologies.json"


@pytest.fixture(scope="session")
def eval_candidates_path():
    return FIXTURES_DIR / "triage_eval_candidates.jsonl"
```

Run: `PYTHONPATH=/tmp/aml-tutor-plan003-scratch/src python3 -m pytest tests/ -v --collect-only`
Expected: collects with no errors.

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures/typologies.json tests/fixtures/PROVENANCE_TYPOLOGIES.md \
  scripts/build_eval_candidates.py tests/fixtures/triage_eval_candidates.jsonl tests/conftest.py
git commit -m "feat: typology corpus and triage eval-set candidates for lessons 008-015"
```

---

## Task 2: Lesson 008 — typology retrieval (keyword half)

**Files:**
- Create: `aml-tutor/docs/specs/008-typology-retrieval.md`
- Create: `aml-tutor/tests/008_test_retrieval.py`
- Modify: `aml-tutor/docs/specs/README.md`
- Create (scratch): `/tmp/aml-tutor-plan003-scratch/src/aml_triage/retrieval.py`

**Interfaces:**
- Consumes: `tests/fixtures/typologies.json` (Task 1) for the baked-in test; in production use,
  consumes the student's own copy at `aml-triage/data/typologies.json` (placed in this lesson).
- Produces: `top_k_typologies(query, k=3, corpus_path=None) -> list[dict]`, each result shaped
  `{"id", "title", "text", "score"}`, sorted descending by `score`; also the private helpers
  `_load_corpus`, `_documents`, `_tfidf_scores` — lesson 009 imports these three by name to build the
  embedding half without duplicating the TF-IDF logic.

- [ ] **Step 1: Write the failing test**

Create `aml-tutor/tests/008_test_retrieval.py`:

```python
"""Baked-in check for lesson 008 — typology retrieval (the keyword/TF-IDF half of hybrid retrieval).

TF-IDF + cosine similarity over a fixed, six-document corpus is exact linear algebra, not a
downloaded model — so this test can assert exact structural and ordering properties.
"""


def test_returns_k_results_sorted_by_descending_score(typologies_path):
    from aml_triage.retrieval import top_k_typologies

    results = top_k_typologies("large cash withdrawal pattern", k=3, corpus_path=str(typologies_path))
    assert len(results) == 3
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_result_shape_matches_the_corpus_entries(typologies_path):
    from aml_triage.retrieval import top_k_typologies

    results = top_k_typologies("structuring below a threshold", k=1, corpus_path=str(typologies_path))
    assert set(results[0].keys()) == {"id", "title", "text", "score"}
    assert results[0]["id"].startswith("TY-")


def test_structuring_query_ranks_the_structuring_typology_first(typologies_path):
    from aml_triage.retrieval import top_k_typologies

    results = top_k_typologies(
        "many small transfers just under a reporting threshold", k=1, corpus_path=str(typologies_path)
    )
    assert results[0]["id"] == "TY-001"


def test_k_larger_than_corpus_returns_the_whole_corpus(typologies_path):
    from aml_triage.retrieval import top_k_typologies

    results = top_k_typologies("anything", k=100, corpus_path=str(typologies_path))
    assert len(results) == 6
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `PYTHONPATH=/tmp/aml-tutor-plan003-scratch/src python3 -m pytest tests/008_test_retrieval.py -v`
Expected: `ModuleNotFoundError: No module named 'aml_triage.retrieval'`.

- [ ] **Step 3: Write the scratch reference implementation**

Create `/tmp/aml-tutor-plan003-scratch/src/aml_triage/retrieval.py`:

```python
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_CORPUS_PATH = "data/typologies.json"


def _load_corpus(corpus_path):
    with open(corpus_path or DEFAULT_CORPUS_PATH) as f:
        return json.load(f)


def _documents(corpus):
    return [f"{entry['title']}. {entry['text']}" for entry in corpus]


def _tfidf_scores(query, documents):
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])
    return cosine_similarity(query_vector, doc_vectors)[0]


def top_k_typologies(query, k=3, corpus_path=None):
    corpus = _load_corpus(corpus_path)
    documents = _documents(corpus)
    scores = _tfidf_scores(query, documents)
    ranked = sorted(zip(corpus, scores), key=lambda pair: pair[1], reverse=True)
    top_k = ranked[:k]
    return [
        {"id": entry["id"], "title": entry["title"], "text": entry["text"], "score": float(score)}
        for entry, score in top_k
    ]
```

Also create `/tmp/aml-tutor-plan003-scratch/src/aml_triage/__init__.py` (empty file) — this is the
first scratch module this plan creates, and, per iteration 002's precedent (its Task 2 Step 3), the
scratch package needs a real `__init__.py` from its first file onward. Lesson 011's `triage.py`
later uses relative imports (`from .retrieval import ...`) that depend on this being a real package,
not an implicit namespace package.

- [ ] **Step 4: Run it to confirm it passes**

Run: `PYTHONPATH=/tmp/aml-tutor-plan003-scratch/src python3 -m pytest tests/008_test_retrieval.py -v`
Expected: 4 passed. If `test_structuring_query_ranks_the_structuring_typology_first` fails, adjust
the test's query wording (not the corpus) until it's unambiguous — the same kind of fixture tuning
`carve_fixture.py`'s `STEP_RANGE` needed in iteration 002.

- [ ] **Step 5: Write the spec**

Create `aml-tutor/docs/specs/008-typology-retrieval.md`, four-part shape, covering:

- **Key concept:** RAG's retrieval half starts here; TF-IDF + cosine similarity is the right tool for
  a six-document corpus — no model download, exact reproducible ranking, zero new dependencies.
  **At a regulated shop:** this is the keyword/BM25 half of a hybrid retrieval pipeline typically run
  on OpenSearch/Elasticsearch or Postgres+pgvector — reusing already-audited infrastructure rather
  than onboarding a new vector-DB vendor. Lesson 009 adds the embedding half this lesson deliberately
  leaves out.
- **Implementation order:** (1) copy `aml-tutor/tests/fixtures/typologies.json` into
  `aml-triage/data/typologies.json` (original content, a plain file copy, no download/license
  friction unlike the PaySim CSV); (2) implement `top_k_typologies(query, k=3, corpus_path=None) ->
  list[dict]` in `src/aml_triage/retrieval.py`, with `corpus_path=None` defaulting to
  `"data/typologies.json"` relative to the process's cwd (which is `aml-triage`'s own root under
  every real validation command), and factoring out `_load_corpus`,
  `_documents`, and `_tfidf_scores` as separate functions — lesson 009 reuses all three.
- **Checks:** comprehension questions (e.g. "why would this exact approach stop working well once
  the typology library grows to thousands of documents, and what's the first thing you'd swap in?");
  baked-in `uv run pytest ../aml-tutor/tests/008_test_retrieval.py` command + validation block
  (`id: "008-typology-retrieval"`).
- **Pressure test:** lesson 010's `parse_triage_decision` validates cited typology ids against only
  the ids actually retrieved and shown for a given call — a `k` too small to include the typology
  that actually matches a transaction means a well-formed decision gets its citation rejected for
  the right reason: the model can only cite what it was shown.
- **Doer fallback note:** no shell command required (no `uv add` — scikit-learn is already a
  dependency); unlike lesson 002's CSV, `typologies.json` has nothing to download, so the doer can
  place `aml-triage/data/typologies.json` itself in addition to writing `retrieval.py` by hand.

- [ ] **Step 6: Update the ledger**

Add row 008 as `Done` (see Task 10 for the ledger's Part 1/2/3 regrouping, done once at the end).

- [ ] **Step 7: Commit**

```bash
git add docs/specs/008-typology-retrieval.md tests/008_test_retrieval.py docs/specs/README.md
git commit -m "feat: lesson 008 — typology retrieval"
```

---

## Task 3: Lesson 009 — hybrid retrieval (embeddings half)

**Files:**
- Create: `aml-tutor/docs/specs/009-hybrid-retrieval.md`
- Create: `aml-tutor/tests/009_test_hybrid_retrieval.py`
- Modify: `aml-tutor/docs/specs/README.md`
- Modify (scratch): `/tmp/aml-tutor-plan003-scratch/src/aml_triage/retrieval.py`

**Interfaces:**
- Consumes: Task 2's `_load_corpus`, `_documents`, `_tfidf_scores`, `top_k_typologies`.
- Produces: `top_k_typologies_hybrid(query, k=3, corpus_path=None, alpha=0.5) -> list[dict]`, same
  result shape as lesson 008 — lesson 011's `triage()` calls this, not the pure-TF-IDF function.

- [ ] **Step 1: Write the failing test**

Create `aml-tutor/tests/009_test_hybrid_retrieval.py`:

```python
"""Baked-in check for lesson 009 — hybrid retrieval.

alpha=1.0 zeroes out the embedding term entirely, so that one case stays exactly deterministic
regardless of which sentence-transformers version or model weights are installed — the other
assertions are intentionally soft (structure and "in the top two"), since embedding scores
themselves can shift slightly across model versions.
"""


def test_alpha_1_matches_pure_tfidf_ranking_exactly(typologies_path):
    from aml_triage.retrieval import top_k_typologies, top_k_typologies_hybrid

    query = "many small transfers just under a reporting threshold"
    tfidf_only = top_k_typologies(query, k=3, corpus_path=str(typologies_path))
    hybrid_alpha_1 = top_k_typologies_hybrid(query, k=3, corpus_path=str(typologies_path), alpha=1.0)
    assert [r["id"] for r in tfidf_only] == [r["id"] for r in hybrid_alpha_1]


def test_returns_k_results_with_expected_shape(typologies_path):
    from aml_triage.retrieval import top_k_typologies_hybrid

    results = top_k_typologies_hybrid("cash withdrawal pattern", k=3, corpus_path=str(typologies_path))
    assert len(results) == 3
    assert set(results[0].keys()) == {"id", "title", "text", "score"}


def test_structuring_query_ranks_the_structuring_typology_in_the_top_two(typologies_path):
    from aml_triage.retrieval import top_k_typologies_hybrid

    results = top_k_typologies_hybrid(
        "many small transfers just under a reporting threshold", k=2, corpus_path=str(typologies_path)
    )
    assert "TY-001" in [r["id"] for r in results]


def test_alpha_0_still_returns_a_valid_shape(typologies_path):
    from aml_triage.retrieval import top_k_typologies_hybrid

    results = top_k_typologies_hybrid(
        "cash withdrawal pattern", k=6, corpus_path=str(typologies_path), alpha=0.0
    )
    assert len(results) == 6
```

- [ ] **Step 2: Run it to confirm it fails**

Run:
`PYTHONPATH=/tmp/aml-tutor-plan003-scratch/src python3 -m pytest tests/009_test_hybrid_retrieval.py -v`
Expected: `ImportError: cannot import name 'top_k_typologies_hybrid'`.

- [ ] **Step 3: Write the scratch reference implementation**

Append to `/tmp/aml-tutor-plan003-scratch/src/aml_triage/retrieval.py`:

```python
_EMBEDDING_MODEL = None


def _get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL


def _embedding_scores(query, documents):
    model = _get_embedding_model()
    doc_embeddings = model.encode(documents)
    query_embedding = model.encode([query])
    return cosine_similarity(query_embedding, doc_embeddings)[0]


def top_k_typologies_hybrid(query, k=3, corpus_path=None, alpha=0.5):
    corpus = _load_corpus(corpus_path)
    documents = _documents(corpus)

    tfidf_scores = _tfidf_scores(query, documents)
    embedding_scores = _embedding_scores(query, documents)
    blended = alpha * tfidf_scores + (1 - alpha) * embedding_scores

    ranked = sorted(zip(corpus, blended), key=lambda pair: pair[1], reverse=True)
    top_k = ranked[:k]
    return [
        {"id": entry["id"], "title": entry["title"], "text": entry["text"], "score": float(score)}
        for entry, score in top_k
    ]
```

- [ ] **Step 4: Run it to confirm it passes**

```bash
uv --directory /tmp/aml-tutor-plan003-scratch pip install sentence-transformers 2>/dev/null || \
  pip install sentence-transformers
PYTHONPATH=/tmp/aml-tutor-plan003-scratch/src python3 -m pytest tests/009_test_hybrid_retrieval.py -v
```

Expected: 4 passed. First run downloads the `all-MiniLM-L6-v2` model weights (~90MB) — expected, one
time, requires network.

- [ ] **Step 5: Write the spec**

Create `aml-tutor/docs/specs/009-hybrid-retrieval.md`, four-part shape, covering:

- **Key concept:** why hybrid retrieval (keyword + embeddings, blended) beats either alone for
  compliance-flavored text — exact terms and thresholds favor keyword matching, paraphrase and
  conceptual similarity favor embeddings. `alpha` is the blend weight: `alpha=1.0` is pure TF-IDF,
  `alpha=0.0` is pure embeddings. **At a regulated shop:** the embedding half would come from a model
  already inside an approved enterprise contract (Azure OpenAI/Bedrock/Vertex embeddings) or
  self-hosted inside the bank's own VPC when the text is AML/SAR-adjacent and can't leave the
  compliance boundary to a new subprocessor at all — this lesson's local, open-weights model
  (`sentence-transformers`) is the version of that same idea that needs no vendor relationship at all.
- **Implementation order:** add the `sentence-transformers` dependency (`uv add sentence-transformers`);
  implement `top_k_typologies_hybrid(query, k=3, corpus_path=None, alpha=0.5) -> list[dict]` in
  `src/aml_triage/retrieval.py`, reusing `_load_corpus`/`_documents`/`_tfidf_scores` from lesson 008
  rather than duplicating them.
- **Checks:** comprehension questions on when you'd raise `alpha` toward 1.0 versus lower it toward
  0.0 for this specific corpus; baked-in test command + validation block
  (`id: "009-hybrid-retrieval"`).
- **Pressure test:** today's test proves the blending arithmetic is correct (via the `alpha=1.0`
  exact-match case), not that 0.5 is the right weight for this corpus — a real deployment would tune
  `alpha` against labeled retrieval-quality data, which this six-document teaching corpus is too
  small to support meaningfully.
- **Doer fallback note:** the doer cannot run `uv add sentence-transformers` or trigger the first-run
  model download (both need shell/network); once installed, it can write the code in
  `src/aml_triage/retrieval.py` by hand.

- [ ] **Step 6: Update the ledger**

Add row 009 as `Done`.

- [ ] **Step 7: Commit**

```bash
git add docs/specs/009-hybrid-retrieval.md tests/009_test_hybrid_retrieval.py docs/specs/README.md
git commit -m "feat: lesson 009 — hybrid retrieval"
```

---

## Task 4: Lesson 010 — structured triage decisions

**Files:**
- Create: `aml-tutor/docs/specs/010-structured-triage-decisions.md`
- Create: `aml-tutor/tests/010_test_structured_decisions.py`
- Modify: `aml-tutor/docs/specs/README.md`
- Create (scratch): `/tmp/aml-tutor-plan003-scratch/src/aml_triage/triage_schema.py`

**Interfaces:**
- Consumes: Tasks 2/3's retrieval output shape (list of `{"id", "title", "text", "score"}` dicts) as
  the `retrieved` parameter — does not call retrieval directly.
- Produces: `TRIAGE_TOOL_SCHEMA` (a dict), `build_prompt(transaction, classifier_score, retrieved) ->
  str`, and `parse_triage_decision(tool_input, known_typology_ids) -> dict` — lesson 011 imports all
  three by name.

- [ ] **Step 1: Write the failing test**

Create `aml-tutor/tests/010_test_structured_decisions.py`:

```python
"""Baked-in check for lesson 010 — structured triage decisions.

No LLM call here: build_prompt and parse_triage_decision are pure functions over plain data. The
real API call is lesson 011's job.
"""
import pytest

TRANSACTION = {"step": 5, "type": "TRANSFER", "amount": 181.0}
RETRIEVED = [
    {"id": "TY-001", "title": "Structuring / smurfing", "text": "...", "score": 0.9},
    {"id": "TY-003", "title": "Rapid pass-through", "text": "...", "score": 0.4},
]


def test_prompt_includes_transaction_score_and_retrieved_titles():
    from aml_triage.triage_schema import build_prompt

    prompt = build_prompt(TRANSACTION, classifier_score=0.87, retrieved=RETRIEVED)
    assert "TRANSFER" in prompt
    assert "181.0" in prompt
    assert "0.87" in prompt
    assert "Structuring / smurfing" in prompt
    assert "Rapid pass-through" in prompt


def test_tool_schema_names_a_forced_tool_with_the_expected_fields():
    from aml_triage.triage_schema import TRIAGE_TOOL_SCHEMA

    properties = TRIAGE_TOOL_SCHEMA["input_schema"]["properties"]
    assert set(properties.keys()) == {"decision", "rationale", "cited_typology_ids"}
    assert properties["decision"]["enum"] == ["escalate", "monitor", "close"]


def test_parse_returns_a_clean_dict_for_a_valid_tool_call():
    from aml_triage.triage_schema import parse_triage_decision

    tool_input = {"decision": "escalate", "rationale": "Matches structuring.", "cited_typology_ids": ["TY-001"]}
    result = parse_triage_decision(tool_input, known_typology_ids={"TY-001", "TY-003"})
    assert result == tool_input


def test_parse_rejects_an_invalid_decision_enum():
    from aml_triage.triage_schema import parse_triage_decision

    tool_input = {"decision": "ignore", "rationale": "x", "cited_typology_ids": []}
    with pytest.raises(ValueError):
        parse_triage_decision(tool_input, known_typology_ids=set())


def test_parse_rejects_a_citation_the_model_was_not_shown():
    from aml_triage.triage_schema import parse_triage_decision

    tool_input = {"decision": "escalate", "rationale": "x", "cited_typology_ids": ["TY-005"]}
    with pytest.raises(ValueError):
        parse_triage_decision(tool_input, known_typology_ids={"TY-001", "TY-003"})
```

- [ ] **Step 2: Run it to confirm it fails**

Run:
`PYTHONPATH=/tmp/aml-tutor-plan003-scratch/src python3 -m pytest tests/010_test_structured_decisions.py -v`
Expected: `ModuleNotFoundError: No module named 'aml_triage.triage_schema'`.

- [ ] **Step 3: Write the scratch reference implementation**

Create `/tmp/aml-tutor-plan003-scratch/src/aml_triage/triage_schema.py`:

```python
TRIAGE_TOOL_SCHEMA = {
    "name": "submit_triage_decision",
    "description": "Submit a structured triage decision for a flagged transaction.",
    "input_schema": {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["escalate", "monitor", "close"]},
            "rationale": {"type": "string"},
            "cited_typology_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["decision", "rationale", "cited_typology_ids"],
    },
}


def build_prompt(transaction, classifier_score, retrieved):
    typology_lines = "\n".join(f"- {t['id']} ({t['title']}): {t['text']}" for t in retrieved)
    return (
        f"A transaction was flagged by an automated fraud classifier with score {classifier_score}.\n"
        f"Transaction: type={transaction['type']}, amount={transaction['amount']}, step={transaction['step']}.\n"
        "Candidate AML typologies retrieved for this transaction:\n"
        f"{typology_lines}\n\n"
        "Decide whether to escalate, monitor, or close this transaction. Cite only typology ids "
        "from the list above that actually support your rationale."
    )


def parse_triage_decision(tool_input, known_typology_ids):
    decision = tool_input.get("decision")
    if decision not in ("escalate", "monitor", "close"):
        raise ValueError(f"invalid decision: {decision!r}")

    cited = tool_input.get("cited_typology_ids", [])
    unknown = set(cited) - set(known_typology_ids)
    if unknown:
        raise ValueError(f"cited typology ids not shown to the model: {sorted(unknown)}")

    return {
        "decision": decision,
        "rationale": tool_input["rationale"],
        "cited_typology_ids": cited,
    }
```

- [ ] **Step 4: Run it to confirm it passes**

Run:
`PYTHONPATH=/tmp/aml-tutor-plan003-scratch/src python3 -m pytest tests/010_test_structured_decisions.py -v`
Expected: 5 passed.

- [ ] **Step 5: Write the spec**

Create `aml-tutor/docs/specs/010-structured-triage-decisions.md`, four-part shape, covering:

- **Key concept:** forcing structured output at the model boundary (`tool_choice={"type": "tool",
  "name": ...}`) instead of parsing free text. `known_typology_ids` means the ids actually retrieved
  and shown for this call, not the whole corpus — a model citing a real typology it was never shown
  is rejected the same as a fabricated one. **At a regulated shop:** forcing a validated, typed
  contract at the model boundary — rather than trusting free text — is exactly what a model-risk
  review would require before this output could feed any downstream system.
- **Implementation order:** implement `TRIAGE_TOOL_SCHEMA`, `build_prompt(...)`, and
  `parse_triage_decision(...)` in `src/aml_triage/triage_schema.py`, all pure functions with no SDK
  import and no network call — deliberate, and why this lesson adds no dependency.
- **Checks:** comprehension questions on why an unknown-but-real citation is rejected the same way
  as a fabricated one; baked-in test command + validation block
  (`id: "010-structured-triage-decisions"`).
- **Pressure test:** lesson 011 is the first lesson to actually call an LLM with this schema — a
  model that ignores the forced-tool instruction and cites an id outside what `build_prompt` listed
  will be caught by `parse_triage_decision`'s guard; lesson 011's own test must exercise this guard
  through the full call path, not just this lesson's direct unit test.
- **Doer fallback note:** no shell command; the doer can write this file by hand if asked, since
  every rule here is fixed by this spec.

- [ ] **Step 6: Update the ledger**

Add row 010 as `Done`.

- [ ] **Step 7: Commit**

```bash
git add docs/specs/010-structured-triage-decisions.md tests/010_test_structured_decisions.py docs/specs/README.md
git commit -m "feat: lesson 010 — structured triage decisions"
```

---

## Task 5: Lesson 011 — the end-to-end triage agent

**Files:**
- Create: `aml-tutor/docs/specs/011-the-triage-agent.md`
- Create: `aml-tutor/tests/011_test_triage_agent.py`
- Modify: `aml-tutor/docs/specs/README.md`
- Create (scratch): `/tmp/aml-tutor-plan003-scratch/src/aml_triage/triage.py`

**Interfaces:**
- Consumes: Task 3's `top_k_typologies_hybrid`, Task 4's `TRIAGE_TOOL_SCHEMA` / `build_prompt` /
  `parse_triage_decision`.
- Produces: `triage(transaction, classifier_score, *, client=None, k=3, corpus_path=None, alpha=0.5)
  -> dict`, returning `{"decision", "rationale", "cited_typology_ids", "retrieved"}` — one lesson
  012 optionally calls to see the agent's opinion while labeling, and lesson 015's pipeline calls
  once per case; lessons 013/014 consume its `result` dict shape (including `retrieved`) without
  calling it themselves.

This is the first lesson where `aml_triage` code calls a real LLM. The baked-in test injects a fake
client through `client=` and never touches the network.

- [ ] **Step 1: Write the failing test**

Create `aml-tutor/tests/011_test_triage_agent.py`:

```python
"""Baked-in check for lesson 011 — the end-to-end triage agent.

Uses a fake client injected through triage()'s client= parameter. No network call, no
ANTHROPIC_API_KEY needed to pass this test.
"""
from types import SimpleNamespace

import pytest


class _FakeMessages:
    def __init__(self, tool_input):
        self._tool_input = tool_input
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = SimpleNamespace(type="tool_use", input=self._tool_input)
        return SimpleNamespace(content=[block])


class _FakeClient:
    def __init__(self, tool_input):
        self.messages = _FakeMessages(tool_input)


def test_triage_forces_the_structured_tool_and_returns_the_parsed_decision(typologies_path):
    from aml_triage.retrieval import top_k_typologies_hybrid
    from aml_triage.triage import triage
    from aml_triage.triage_schema import TRIAGE_TOOL_SCHEMA

    transaction = {"step": 5, "type": "TRANSFER", "amount": 181.0}
    retrieved = top_k_typologies_hybrid(
        "many small transfers just under a reporting threshold", k=2, corpus_path=str(typologies_path)
    )
    valid_id = retrieved[0]["id"]

    fake_client = _FakeClient(
        {"decision": "escalate", "rationale": "Matches a known pattern.", "cited_typology_ids": [valid_id]}
    )
    result = triage(
        transaction, classifier_score=0.9, client=fake_client, k=2, corpus_path=str(typologies_path)
    )

    assert result["decision"] == "escalate"
    assert result["cited_typology_ids"] == [valid_id]
    assert result["retrieved"] == retrieved

    call_kwargs = fake_client.messages.calls[0]
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": TRIAGE_TOOL_SCHEMA["name"]}
    assert call_kwargs["tools"] == [TRIAGE_TOOL_SCHEMA]


def test_triage_raises_when_the_model_cites_something_it_was_not_shown(typologies_path):
    from aml_triage.triage import triage

    transaction = {"step": 5, "type": "TRANSFER", "amount": 181.0}
    fake_client = _FakeClient(
        {"decision": "escalate", "rationale": "x", "cited_typology_ids": ["TY-999"]}
    )
    with pytest.raises(ValueError):
        triage(
            transaction, classifier_score=0.9, client=fake_client, k=1, corpus_path=str(typologies_path)
        )
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `PYTHONPATH=/tmp/aml-tutor-plan003-scratch/src python3 -m pytest tests/011_test_triage_agent.py -v`
Expected: `ModuleNotFoundError: No module named 'aml_triage.triage'`.

- [ ] **Step 3: Write the scratch reference implementation**

Create `/tmp/aml-tutor-plan003-scratch/src/aml_triage/triage.py`:

```python
import os

from .retrieval import top_k_typologies_hybrid
from .triage_schema import TRIAGE_TOOL_SCHEMA, build_prompt, parse_triage_decision

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def triage(transaction, classifier_score, *, client=None, k=3, corpus_path=None, alpha=0.5):
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    retrieved = top_k_typologies_hybrid(
        f"{transaction['type']} transaction of amount {transaction['amount']}",
        k=k,
        corpus_path=corpus_path,
        alpha=alpha,
    )
    known_ids = {t["id"] for t in retrieved}
    prompt = build_prompt(transaction, classifier_score, retrieved)

    response = client.messages.create(
        model=os.environ.get("TRIAGE_MODEL", DEFAULT_MODEL),
        max_tokens=1024,
        tools=[TRIAGE_TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": TRIAGE_TOOL_SCHEMA["name"]},
        messages=[{"role": "user", "content": prompt}],
    )
    tool_use = next(block for block in response.content if block.type == "tool_use")
    result = parse_triage_decision(tool_use.input, known_ids)
    result["retrieved"] = retrieved
    return result
```

Note: `triage`'s `corpus_path`/`alpha` thread straight through to `top_k_typologies_hybrid`.
Production callers omit both and get lesson 009's defaults; the baked-in test always passes
`corpus_path` explicitly, pinning both `triage`'s own retrieval call and the test's separate direct
`top_k_typologies_hybrid(...)` call to the same fixture corpus regardless of the test process's cwd.

`triage`'s return value carries a `"retrieved"` key alongside `decision`/`rationale`/
`cited_typology_ids` — one field wider than `parse_triage_decision`'s own three-key contract (lesson
010's contract is unchanged; `triage` just adds to what it hands back). This is deliberate and fixes
a real gap: lesson 014's `llm_judge_score` needs the retrieved-typology list to build its prompt, and
without `triage` carrying it forward, lesson 015's pipeline would have no way to recover the exact
list a given `result` was actually judged against.

- [ ] **Step 4: Run it to confirm it passes**

Run: `PYTHONPATH=/tmp/aml-tutor-plan003-scratch/src python3 -m pytest tests/011_test_triage_agent.py -v`
Expected: 2 passed.

- [ ] **Step 5: Write the spec**

Create `aml-tutor/docs/specs/011-the-triage-agent.md`, four-part shape, covering:

- **Key concept:** the LLM client as an injected, testable seam — `triage(...)` takes `client=None`
  and only constructs a real `anthropic.Anthropic()` when nothing was supplied. This is the concrete
  pattern that answers "how do you test an LLM feature" in an interview. **At a regulated shop:** this
  call would go through Bedrock or Azure OpenAI rather than a public API directly, for the enterprise
  DPA and VPC boundary — and its output would be a draft recommendation routed to a human analyst
  queue, never an auto-executed decision; human-in-the-loop is close to a hard requirement here.
- **Implementation order:** add the `anthropic` dependency (`uv add anthropic`); implement
  `triage(transaction, classifier_score, *, client=None, k=3, corpus_path=None, alpha=0.5) -> dict`
  in `src/aml_triage/triage.py`, wiring retrieval (008/009) → prompt + schema (010) → the forced tool
  call → parsing (010); the model name is read from a `TRIAGE_MODEL` env var (mirroring
  `aml-tutor`'s own `TUTOR_MODEL` convention), defaulting to a small, fast model since this runs once
  per flagged transaction.
- **Checks:** comprehension questions on why the client is a parameter rather than constructed
  unconditionally inside `triage`; baked-in test command + validation block
  (`id: "011-the-triage-agent"`).
- **Pressure test:** today's test only ever sees a fake client — it proves the wiring and the
  citation guard, not that a real Claude call produces sensible triage decisions. As a separate,
  manual (non-pytest) step, the spec has the student export `ANTHROPIC_API_KEY` and call `triage(...)`
  for real against one transaction the Phase 1 classifier actually flagged, and read the result.
- **Doer fallback note:** the doer cannot run `uv add anthropic` (no shell access); once installed,
  it can write `src/aml_triage/triage.py` by hand.

- [ ] **Step 6: Update the ledger**

Add row 011 as `Done`.

- [ ] **Step 7: Commit**

```bash
git add docs/specs/011-the-triage-agent.md tests/011_test_triage_agent.py docs/specs/README.md
git commit -m "feat: lesson 011 — the end-to-end triage agent"
```

---

## Task 6: Lesson 012 — the hand-labeled triage eval set

**Files:**
- Create: `aml-tutor/docs/specs/012-the-hand-labeled-eval-set.md`
- Create: `aml-tutor/tests/012_test_eval_set.py`
- Modify: `aml-tutor/docs/specs/README.md`
- Create (scratch): `/tmp/aml-tutor-plan003-scratch/src/aml_triage/eval.py` (started here, grown by
  Tasks 7–9)

**Interfaces:**
- Consumes: `tests/fixtures/triage_eval_candidates.jsonl` (Task 1) as the pool the student labels.
- Produces: `load_eval_set(path) -> list[dict]`, and — as the actual deliverable of this lesson, not
  code — the student's own `aml-triage/data/triage_eval_set.jsonl`, one row per candidate with a
  `label_decision` and `label_note` the student wrote themselves. Tasks 7–9 consume this file's shape
  by name.

Unlike every other lesson, there is no committed `aml-tutor` fixture to check this lesson's *content*
against — the point is that the student wrote the labels. The baked-in test only checks structure.

- [ ] **Step 1: Write the failing test**

Create `aml-tutor/tests/012_test_eval_set.py`:

```python
"""Baked-in check for lesson 012 — the hand-labeled triage eval set.

This test checks the STRUCTURE of the student's own aml-triage/data/triage_eval_set.jsonl: schema,
row count, allowed decision values, and that more than one decision type was actually used. Whether
the labels themselves are good judgment calls is checked by the tutor in conversation (see the
lesson's Checks section) — the same split every other lesson draws between code (pytest) and
understanding (conversation), applied here to labeling quality instead.
"""


def test_eval_set_has_one_row_per_candidate(eval_candidates_path):
    from aml_triage.eval import load_eval_set

    cases = load_eval_set("data/triage_eval_set.jsonl")
    with open(eval_candidates_path) as f:
        candidate_count = sum(1 for _ in f)
    assert len(cases) == candidate_count


def test_every_case_has_the_expected_keys():
    from aml_triage.eval import load_eval_set

    cases = load_eval_set("data/triage_eval_set.jsonl")
    for case in cases:
        assert set(case.keys()) == {"transaction", "classifier_score", "label_decision", "label_note"}


def test_label_decisions_are_within_the_allowed_set():
    from aml_triage.eval import load_eval_set

    cases = load_eval_set("data/triage_eval_set.jsonl")
    assert {c["label_decision"] for c in cases}.issubset({"escalate", "monitor", "close"})


def test_more_than_one_decision_type_was_actually_used():
    from aml_triage.eval import load_eval_set

    cases = load_eval_set("data/triage_eval_set.jsonl")
    assert len({c["label_decision"] for c in cases}) >= 2


def test_every_label_note_is_a_real_sentence_not_left_blank():
    from aml_triage.eval import load_eval_set

    cases = load_eval_set("data/triage_eval_set.jsonl")
    for case in cases:
        assert len(case["label_note"].strip()) >= 10
```

- [ ] **Step 2: Run it to confirm it fails**

Simulate a fresh `aml-triage` by treating a scratch directory as the cwd (this is the first lesson
in the tutorial where the test itself, not just `import aml_triage`, depends on `cwd`):

```bash
mkdir -p /tmp/aml-tutor-plan003-scratch/aml-triage/data
cd /tmp/aml-tutor-plan003-scratch/aml-triage
PYTHONPATH=/tmp/aml-tutor-plan003-scratch/src python3 -m pytest /Users/pebarna/projects/aml-tutor/tests/012_test_eval_set.py -v
```

Expected: `ModuleNotFoundError: No module named 'aml_triage.eval'`.

- [ ] **Step 3: Write the scratch reference implementation**

Create `/tmp/aml-tutor-plan003-scratch/src/aml_triage/eval.py`:

```python
import json


def load_eval_set(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
```

- [ ] **Step 4: Write a throwaway stand-in labeled file to prove the test's shape-checks pass**

This is verification scaffolding only, not a deliverable — it stands in for "a student's hand
labeling" so this task can confirm the test framework itself is sound before the spec is written:

```bash
python3 -c "
import json
rows = [json.loads(l) for l in open('/Users/pebarna/projects/aml-tutor/tests/fixtures/triage_eval_candidates.jsonl')]
out = []
for r in rows:
    decision = 'escalate' if r['classifier_score'] >= 0.5 else 'close'
    out.append({**r, 'label_decision': decision, 'label_note': 'stand-in label for plan verification only, not a real judgment call'})
with open('/tmp/aml-tutor-plan003-scratch/aml-triage/data/triage_eval_set.jsonl', 'w') as f:
    for row in out:
        f.write(json.dumps(row) + '\n')
"
```

- [ ] **Step 5: Run it to confirm it passes**

```bash
cd /tmp/aml-tutor-plan003-scratch/aml-triage
PYTHONPATH=/tmp/aml-tutor-plan003-scratch/src python3 -m pytest /Users/pebarna/projects/aml-tutor/tests/012_test_eval_set.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Write the spec**

Create `aml-tutor/docs/specs/012-the-hand-labeled-eval-set.md`, four-part shape, covering:

- **Key concept:** SEED.md's Phase 3 asks for hand-annotated cases for a reason: a mechanically
  generated label can never produce a genuine borderline `monitor` call, and it's exactly those
  judgment calls a real analyst's labeling would surface. **At a regulated shop:** this is what an
  SR 11-7-style model-risk review actually requires — a documented, SME-labeled golden set,
  periodically refreshed by real humans, not synthetic labels. State plainly that
  `tests/fixtures/triage_eval_candidates.jsonl` (Task 1) gives 16 *unlabeled* candidates (8
  fraud-flagged, 8 not, by the PaySim simulator's own `isFraud` flag) and that `classifier_score` in
  each is a documented stand-in, not a real Phase 1 model inference.
- **Implementation order:** (1) implement `load_eval_set(path) -> list[dict]` in
  `src/aml_triage/eval.py`; (2) for each of the 16 candidates in
  `../aml-tutor/tests/fixtures/triage_eval_candidates.jsonl`, look at the transaction fields (and
  optionally call your own `triage(...)` on it to see what the agent would say, then decide whether
  you agree), and write your own judgment as `label_decision` (`escalate`/`monitor`/`close`) and a
  one-sentence `label_note` explaining why; (3) save all 16 labeled rows to
  `aml-triage/data/triage_eval_set.jsonl`. Note the scope honestly: 16 hand-labeled cases here is a
  floor, not SEED.md's 30-50-case target — keep labeling more on your own for the real deliverable.
- **Checks:** rather than generic comprehension questions, the tutor asks about the student's own
  labels directly — e.g. "pick one case you marked `escalate` and one you marked `close` (or
  `monitor`) and explain, in your own words, what in the transaction data drove each call." The tutor
  validates that the stated reasoning is actually consistent with the transaction fields, not that
  it matches some hidden answer key — there isn't one. Baked-in test command + validation block
  (`id: "012-the-hand-labeled-eval-set"`).
- **Pressure test:** lesson 013's `decision_match` compares `triage()` output against these
  hand-written labels — labeling two very similar transactions differently with no principled reason
  will surface downstream as a confusingly low agreement rate that looks like an agent problem but is
  actually a labeling-consistency problem. This is a real, well-known issue with human-labeled eval
  sets (inter/intra-rater consistency), not unique to this tutorial.
- **Doer fallback note:** unlike every other lesson, there is **no doer fallback here** — deciding
  escalate/monitor/close is exactly the human judgment this lesson exists to practice, and the doer
  cannot make that call on the student's behalf. If asked, the tutor should say so plainly rather than
  offering to do the step.

- [ ] **Step 7: Update the ledger**

Add row 012 as `Done`.

- [ ] **Step 8: Commit**

```bash
git add docs/specs/012-the-hand-labeled-eval-set.md tests/012_test_eval_set.py docs/specs/README.md
git commit -m "feat: lesson 012 — the hand-labeled triage eval set"
```

---

## Task 7: Lesson 013 — deterministic triage checks

**Files:**
- Create: `aml-tutor/docs/specs/013-deterministic-triage-checks.md`
- Create: `aml-tutor/tests/013_test_deterministic_checks.py`
- Modify: `aml-tutor/docs/specs/README.md`
- Modify (scratch): `/tmp/aml-tutor-plan003-scratch/src/aml_triage/eval.py`

**Interfaces:**
- Consumes: an eval-set case (Task 6's shape) and a triage result (Task 5's `triage(...)` return
  shape).
- Produces: `deterministic_score(case: dict, result: dict) -> dict`, keys `decision_match`,
  `citation_present` — Task 9's `report` aggregates this by name.

- [ ] **Step 1: Write the failing test**

Create `aml-tutor/tests/013_test_deterministic_checks.py`:

```python
"""Baked-in check for lesson 013 — deterministic triage checks.

Citation *validity* (does the id exist and was it shown to the model) is already enforced by
lesson 010's parse_triage_decision guard, upstream of this function. deterministic_score only
checks citation *presence* — did the model cite anything at all.
"""

CASE_ESCALATE = {
    "transaction": {"step": 1, "type": "TRANSFER", "amount": 181.0},
    "classifier_score": 0.95,
    "label_decision": "escalate",
    "label_note": "Matches structuring: many small amounts, rapid succession.",
}


def test_decision_match_true_when_decisions_agree():
    from aml_triage.eval import deterministic_score

    result = {"decision": "escalate", "rationale": "x", "cited_typology_ids": ["TY-001"]}
    score = deterministic_score(CASE_ESCALATE, result)
    assert score == {"decision_match": True, "citation_present": True}


def test_decision_match_false_when_decisions_disagree():
    from aml_triage.eval import deterministic_score

    result = {"decision": "close", "rationale": "x", "cited_typology_ids": ["TY-001"]}
    score = deterministic_score(CASE_ESCALATE, result)
    assert score["decision_match"] is False


def test_citation_present_false_when_nothing_was_cited():
    from aml_triage.eval import deterministic_score

    result = {"decision": "escalate", "rationale": "x", "cited_typology_ids": []}
    score = deterministic_score(CASE_ESCALATE, result)
    assert score["citation_present"] is False
```

- [ ] **Step 2: Run it to confirm it fails**

Run:
`PYTHONPATH=/tmp/aml-tutor-plan003-scratch/src python3 -m pytest tests/013_test_deterministic_checks.py -v`
Expected: `AttributeError`/`ImportError` — `deterministic_score` does not exist yet.

- [ ] **Step 3: Write the scratch reference implementation**

Append to `/tmp/aml-tutor-plan003-scratch/src/aml_triage/eval.py`:

```python
def deterministic_score(case, result):
    return {
        "decision_match": result["decision"] == case["label_decision"],
        "citation_present": len(result["cited_typology_ids"]) > 0,
    }
```

- [ ] **Step 4: Run it to confirm it passes**

Run:
`PYTHONPATH=/tmp/aml-tutor-plan003-scratch/src python3 -m pytest tests/013_test_deterministic_checks.py -v`
Expected: 3 passed.

- [ ] **Step 5: Write the spec**

Create `aml-tutor/docs/specs/013-deterministic-triage-checks.md`, four-part shape, covering:

- **Key concept:** two of SEED.md's Phase 3 deterministic checks — "did it cite an actual typology"
  and "did its decision match the label" — split across lessons by design: citation *validity* (real
  id, actually shown) was already enforced at generation time by lesson 010's guard, so by the time
  a `result` reaches `deterministic_score` it structurally can only ever cite real, shown ids. What's
  left to check is citation *presence* and decision *agreement* with your lesson-012 label. **At a
  regulated shop:** deterministic, code-level checks like these are the cheap, always-on layer of a
  validation suite — necessary but not sufficient for the sign-off a model-risk review requires.
- **Implementation order:** implement `deterministic_score(case, result) -> dict` in
  `src/aml_triage/eval.py`.
- **Checks:** comprehension questions on why validity and presence are different checks enforced at
  different points; baked-in test command + validation block
  (`id: "013-deterministic-triage-checks"`).
- **Pressure test:** a model that always cites *something* real but never actually engages with the
  transaction would pass `citation_present` every time — this catches "cited nothing," not "cited
  something irrelevant"; the latter is what lesson 014's LLM-as-judge is for.
- **Doer fallback note:** no shell command; the doer can write `deterministic_score` by hand if
  asked.

- [ ] **Step 6: Update the ledger**

Add row 013 as `Done`.

- [ ] **Step 7: Commit**

```bash
git add docs/specs/013-deterministic-triage-checks.md tests/013_test_deterministic_checks.py docs/specs/README.md
git commit -m "feat: lesson 013 — deterministic triage checks"
```

---

## Task 8: Lesson 014 — LLM-as-judge scoring

**Files:**
- Create: `aml-tutor/docs/specs/014-llm-as-judge.md`
- Create: `aml-tutor/tests/014_test_llm_judge.py`
- Modify: `aml-tutor/docs/specs/README.md`
- Modify (scratch): `/tmp/aml-tutor-plan003-scratch/src/aml_triage/eval.py`

**Interfaces:**
- Consumes: a `triage()` result dict (Task 5's shape, including its `"retrieved"` key — this is the
  only way this task gets the retrieved-typology list; there is no separate way to reconstruct it).
- Produces: `llm_judge_score(result, *, client=None) -> dict`, keys `agrees`, `comment` — Task 9's
  `report` aggregates this by name. Same client-injection convention as lesson 011's `triage`.
  Earlier drafts of this task also threaded a `case` parameter through the signature; it was dropped
  because nothing in the judge's job (does the rationale follow from the cited typology text) reads
  the human label — that comparison is lesson 013's `deterministic_score`, not this lesson's.

- [ ] **Step 1: Write the failing test**

Create `aml-tutor/tests/014_test_llm_judge.py`:

```python
"""Baked-in check for lesson 014 — LLM-as-judge scoring.

Same injected-client convention as lesson 011: the baked-in test never calls a real model.
"""
from types import SimpleNamespace

RESULT = {
    "decision": "escalate",
    "rationale": "Matches structuring.",
    "cited_typology_ids": ["TY-001"],
    "retrieved": [{"id": "TY-001", "title": "Structuring / smurfing", "text": "...", "score": 0.9}],
}


class _FakeMessages:
    def __init__(self, tool_input):
        self._tool_input = tool_input
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = SimpleNamespace(type="tool_use", input=self._tool_input)
        return SimpleNamespace(content=[block])


class _FakeClient:
    def __init__(self, tool_input):
        self.messages = _FakeMessages(tool_input)


def test_judge_forces_a_structured_tool_and_returns_its_verdict():
    from aml_triage.eval import llm_judge_score

    fake_client = _FakeClient({"agrees": True, "comment": "Rationale follows from the cited typology."})
    score = llm_judge_score(RESULT, client=fake_client)

    assert score == {"agrees": True, "comment": "Rationale follows from the cited typology."}
    call_kwargs = fake_client.messages.calls[0]
    assert call_kwargs["tool_choice"]["name"] == "submit_judge_verdict"


def test_judge_prompt_includes_the_rationale_and_cited_typology_text():
    from aml_triage.eval import llm_judge_score

    fake_client = _FakeClient({"agrees": False, "comment": "Weak link."})
    llm_judge_score(RESULT, client=fake_client)

    sent_prompt = fake_client.messages.calls[0]["messages"][0]["content"]
    assert "Matches structuring." in sent_prompt
    assert "Structuring / smurfing" in sent_prompt
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `PYTHONPATH=/tmp/aml-tutor-plan003-scratch/src python3 -m pytest tests/014_test_llm_judge.py -v`
Expected: `AttributeError`/`ImportError` — `llm_judge_score` does not exist yet.

- [ ] **Step 3: Write the scratch reference implementation**

Append to `/tmp/aml-tutor-plan003-scratch/src/aml_triage/eval.py`:

```python
import os

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

- [ ] **Step 4: Run it to confirm it passes**

Run: `PYTHONPATH=/tmp/aml-tutor-plan003-scratch/src python3 -m pytest tests/014_test_llm_judge.py -v`
Expected: 2 passed.

- [ ] **Step 5: Write the spec**

Create `aml-tutor/docs/specs/014-llm-as-judge.md`, four-part shape, covering:

- **Key concept:** this judges something lesson 013's deterministic check cannot — whether the
  rationale is actually *supported* by the typology text it cited, not merely whether it cited
  something. Same client-injection pattern as lesson 011, for the same testability reason. **At a
  regulated shop:** LLM-as-judge is used as a fast, scalable *supplementary* signal in ongoing
  monitoring — never a replacement for lesson 012's SME-labeled golden set, which is what an
  SR 11-7-style model-risk review actually requires.
- **Implementation order:** implement `JUDGE_TOOL_SCHEMA`, `_judge_prompt(result)`, and
  `llm_judge_score(result, *, client=None) -> dict` in `src/aml_triage/eval.py`, reading the
  retrieved-typology list from `result["retrieved"]` (lesson 011's `triage()` always includes it) —
  do not add a separate `retrieved` parameter, since `result` already carries everything this
  function needs.
- **Checks:** comprehension questions on why the judge looks at the *cited typology text*
  specifically rather than asking "is this a good decision" in the abstract; baked-in test command +
  validation block (`id: "014-llm-as-judge"`).
- **Pressure test:** a judge model, like any model, can be wrong. This lesson's test proves wiring
  and parsing, not judge accuracy; lesson 015's aggregate `judge_agreement_rate` is a signal to read
  alongside the deterministic numbers and lesson 012's own labels, not a ground truth that overrides
  either.
- **Doer fallback note:** the `anthropic` dependency is already installed (lesson 011); the doer can
  write this lesson's code by hand if asked.

- [ ] **Step 6: Update the ledger**

Add row 014 as `Done`.

- [ ] **Step 7: Commit**

```bash
git add docs/specs/014-llm-as-judge.md tests/014_test_llm_judge.py docs/specs/README.md
git commit -m "feat: lesson 014 — LLM-as-judge scoring"
```

---

## Task 9: Lesson 015 — the agreement-rate report

**Files:**
- Create: `aml-tutor/docs/specs/015-the-agreement-rate-report.md`
- Create: `aml-tutor/tests/015_test_agreement_report.py`
- Modify: `aml-tutor/docs/specs/README.md` (adds the fifteenth and final new row — there is no
  pre-existing `Todo` row to flip; see Step 6)
- Modify (scratch): `/tmp/aml-tutor-plan003-scratch/src/aml_triage/eval.py`

**Interfaces:**
- Consumes: parallel lists produced across Tasks 6–8: `cases`, `results`, `deterministic_scores`,
  `judge_scores`.
- Produces: `report(cases, results, deterministic_scores, judge_scores) -> dict` — the Phase 3
  deliverable.

- [ ] **Step 1: Write the failing test**

Create `aml-tutor/tests/015_test_agreement_report.py`:

```python
"""Baked-in check for lesson 015 — the agreement-rate report."""
import pytest

CASES = [{"label_decision": "escalate"}, {"label_decision": "close"}]
RESULTS = [{"decision": "escalate"}, {"decision": "escalate"}]
DETERMINISTIC = [
    {"decision_match": True, "citation_present": True},
    {"decision_match": False, "citation_present": False},
]
JUDGE = [{"agrees": True, "comment": "ok"}, {"agrees": False, "comment": "weak"}]


def test_report_computes_the_three_rates():
    from aml_triage.eval import report

    result = report(CASES, RESULTS, DETERMINISTIC, JUDGE)
    assert result["n_cases"] == 2
    assert result["decision_agreement_rate"] == pytest.approx(0.5)
    assert result["citation_present_rate"] == pytest.approx(0.5)
    assert result["judge_agreement_rate"] == pytest.approx(0.5)


def test_report_raises_on_mismatched_list_lengths():
    from aml_triage.eval import report

    with pytest.raises(ValueError):
        report(CASES, RESULTS[:1], DETERMINISTIC, JUDGE)
```

- [ ] **Step 2: Run it to confirm it fails**

Run:
`PYTHONPATH=/tmp/aml-tutor-plan003-scratch/src python3 -m pytest tests/015_test_agreement_report.py -v`
Expected: `AttributeError`/`ImportError` — `report` does not exist yet.

- [ ] **Step 3: Write the scratch reference implementation**

Append to `/tmp/aml-tutor-plan003-scratch/src/aml_triage/eval.py`:

```python
def report(cases, results, deterministic_scores, judge_scores):
    lengths = {len(cases), len(results), len(deterministic_scores), len(judge_scores)}
    if len(lengths) != 1:
        raise ValueError(f"mismatched list lengths: {lengths}")

    n = len(cases)
    return {
        "n_cases": n,
        "decision_agreement_rate": sum(d["decision_match"] for d in deterministic_scores) / n,
        "citation_present_rate": sum(d["citation_present"] for d in deterministic_scores) / n,
        "judge_agreement_rate": sum(j["agrees"] for j in judge_scores) / n,
    }
```

- [ ] **Step 4: Run it to confirm it passes**

Run:
`PYTHONPATH=/tmp/aml-tutor-plan003-scratch/src python3 -m pytest tests/015_test_agreement_report.py -v`
Expected: 2 passed.

- [ ] **Step 5: Write the spec**

Create `aml-tutor/docs/specs/015-the-agreement-rate-report.md`, four-part shape, covering:

- **Key concept:** SEED.md names this report's headline number — the decision-agreement rate — "the
  single highest-leverage output of this whole project." It is now backed by real hand-labeled
  ground truth (lesson 012), not a generated scaffold, which is exactly what makes it a defensible
  number to cite. **At a regulated shop:** this report is the shape of what a model-validation
  document's metrics section would show, but the real artifact would be versioned, tied to a
  specific model/prompt version, and re-run on a refreshed golden set on a schedule — not generated
  once and forgotten.
- **Implementation order:** implement `report(cases, results, deterministic_scores, judge_scores) ->
  dict` in `src/aml_triage/eval.py`; then, as a separate step, run the full pipeline over your 16
  hand-labeled cases: `cases = load_eval_set(...)` (012); `results = [triage(c["transaction"],
  c["classifier_score"]) for c in cases]` (011 — each `result` already carries its own `"retrieved"`
  list, so no separate retrieval call is needed here); `deterministic_scores = [deterministic_score(c,
  r) for c, r in zip(cases, results)]` (013); `judge_scores = [llm_judge_score(r) for r in results]`
  (014 — takes `result` alone now, not `case`/`retrieved` separately); then `report(...)`. Save the
  result to `aml-triage/reports/phase3_triage_eval.json` via `json.dump(result, open(...), indent=2)`.
  This requires a real `ANTHROPIC_API_KEY` and real network calls; it is not part of `uv run pytest`
  and is not asserted against fixed expected values.
- **Checks:** comprehension questions on why three separate rates rather than one combined score,
  and what a low `citation_present_rate` versus a low `judge_agreement_rate` would each separately
  suggest is wrong upstream; baked-in test command + validation block
  (`id: "015-the-agreement-rate-report"`).
- **Closing note (last lesson of Parts 2 and 3):** the project described in `aml-triage/SEED.md` is
  now complete end to end: a classifier with a documented operating point
  (`reports/phase1_report.json`, lesson 007) and a triage agent with a documented, hand-labeled
  agreement rate (`reports/phase3_triage_eval.json`, this lesson) — "problem → method → measured
  result," twice, exactly the shape SEED.md's Shipping section asks the eventual README/write-up to
  lead with. Name explicitly that 16 hand-labeled cases is a floor, not SEED.md's 30-50-case target.
- **Doer fallback note:** the doer can write `report` by hand (a pure aggregation function); running
  the full pipeline and saving the JSON file needs the student's own `ANTHROPIC_API_KEY`, same
  boundary as lesson 011.

- [ ] **Step 6: Update the ledger**

Add row 015 as `Done`. After this step, `docs/specs/README.md`'s ledger has all fifteen lesson rows
(001–015) marked `Done` — unlike iteration 002's equivalent step, there is no pre-existing `Todo` row
to flip here; every row this iteration touches is new.

- [ ] **Step 7: Commit**

```bash
git add docs/specs/015-the-agreement-rate-report.md tests/015_test_agreement_report.py docs/specs/README.md
git commit -m "feat: lesson 015 — the agreement-rate report"
```

---

## Task 10: Docs wrap-up, full dry run, and scratch cleanup

**Files:**
- Modify: `aml-tutor/docs/specs/README.md` (header prose: replace "Seven lessons, one linear
  sequence — no parts" with a Part 1/2/3 grouping)
- Modify: `aml-tutor/docs/ARCHITECTURE.md` (§6 lesson-to-code API table gains rows 008–015)
- Modify: `aml-tutor/README.md` (drop the "lessons 002–007 are paused..." sentence; it's stale)

**Interfaces:** None new — this task verifies Tasks 0–9's output together and removes the scratch
workspace, mirroring iteration 002's Task 8.

- [ ] **Step 1: Regroup the ledger by Part**

In `aml-tutor/docs/specs/README.md`, replace the opening paragraph with:

```markdown
Fifteen lessons across the three phases `aml-triage/SEED.md` describes — grouped below as Part 1
(the classifier), Part 2 (the triage agent), and Part 3 (the eval harness) for readability, but still
one linear sequence: each lesson's baked-in test assumes every earlier lesson's function already
exists in `aml-triage`.
```

Restructure the table into three headed groups (`### Part 1 — the classifier`, `### Part 2 — the
triage agent`, `### Part 3 — the eval harness`), keeping the existing columns, with lessons 008–011
under Part 2 and 012–015 under Part 3.

- [ ] **Step 2: Extend `ARCHITECTURE.md`'s lesson-to-code API table**

Add to the table in §6, copying every parameter name exactly as pinned earlier in this plan — do not
abbreviate `classifier_score`→`score` or `known_typology_ids`→`known_ids`; those exact names are
part of the public contract lessons 010/011 already pin:

```markdown
| 008 | `retrieval.top_k_typologies` | `(query, k=3, corpus_path=None) -> list[dict]` |
| 009 | `retrieval.top_k_typologies_hybrid` | `(query, k=3, corpus_path=None, alpha=0.5) -> list[dict]` |
| 010 | `triage_schema.TRIAGE_TOOL_SCHEMA` / `build_prompt` / `parse_triage_decision` | `(transaction, classifier_score, retrieved) -> str` / `(tool_input, known_typology_ids) -> dict` |
| 011 | `triage.triage` | `(transaction, classifier_score, *, client=None, k=3, corpus_path=None, alpha=0.5) -> dict`, returns `{decision, rationale, cited_typology_ids, retrieved}` |
| 012 | `eval.load_eval_set` | `(path) -> list[dict]` |
| 013 | `eval.deterministic_score` | `(case, result) -> dict` |
| 014 | `eval.llm_judge_score` | `(result, *, client=None) -> dict` — reads `result["retrieved"]` |
| 015 | `eval.report` | `(cases, results, deterministic_scores, judge_scores) -> dict` |
```

Note for whoever edits this table: lesson 007 already put an unrelated `report(y_true, scores,
objective) -> dict` in this same table, in `evaluate.py`. Lesson 015's `report` lives in a
similarly-named but distinct module (`eval.py`) with a completely different signature — add a
one-line parenthetical next to row 015 (e.g. "not to be confused with `evaluate.report`, lesson
007") so the table itself doesn't read as a duplicate/typo to someone skimming it later.

- [ ] **Step 2a: Fix `ARCHITECTURE.md`'s stale title**

The document's `# Architecture contract — Iteration 001` header and its opening "Pinned before Phase
A touches the engine, per `docs/iterations/001-classifier-tutorial/plan.md`" line are both stale
after two more iterations have extended this same document. Retitle it `# Architecture contract`
(drop the iteration number — it's a living contract, not a one-iteration artifact) and update the
opening line to note it's been extended by iterations 002 and 003.

- [ ] **Step 3: Drop the stale "paused" sentence from `README.md`**

Remove: "As of this writing, lessons 002–007 are paused pending a real PaySim CSV to carve fixtures
from — see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)'s 'Known gap' section." — both halves are
now false (lessons 002–007 finished in iteration 002; the "Known gap" section was deleted in Task 0).

- [ ] **Step 4: Run every baked-in test together against the scratch package**

```bash
cd /Users/pebarna/projects/aml-tutor
PYTHONPATH=/tmp/aml-tutor-plan003-scratch/src python3 -m pytest \
  tests/008_test_retrieval.py tests/009_test_hybrid_retrieval.py \
  tests/010_test_structured_decisions.py tests/011_test_triage_agent.py \
  tests/013_test_deterministic_checks.py tests/014_test_llm_judge.py \
  tests/015_test_agreement_report.py -v
```

(Lesson 012's test is excluded from this batch run because it depends on `cwd`/a hand-labeled file —
verified separately in Task 6 Step 5.) Expected: all pass together — this catches any cross-lesson
signature mismatch that per-task runs in isolation would miss.

- [ ] **Step 5: Confirm the real `aml-triage` is untouched beyond lessons 001–007's committed work**

Use `find`, not `git status` — iteration 002's Task 8 Step 2 used `find` specifically because
`git status` reads clean for anything covered by `.gitignore`, and after this iteration's Task 0 fix,
`aml-triage/.gitignore` still ignores `data/`, which is exactly where Task 6's real (student)
deliverable, `triage_eval_set.jsonl`, is supposed to live — `git status` would never flag a leak
there even if one happened:

```bash
find /Users/pebarna/projects/aml-triage -mindepth 1 -not -path '*/.git/*'
```

Expected: `SEED.md`, `.gitignore`, and whatever lessons 001–007 already committed — nothing from
`/tmp/aml-tutor-plan003-scratch`, and no `data/triage_eval_set.jsonl` (that file is a real student's
job during lesson 012 itself, not something this plan's own verification should have produced).

- [ ] **Step 6: Confirm `npm run check` still passes**

Run (from `aml-tutor`'s root): `npm run check`
Expected: same clean result as iterations 001–002 — this iteration touched no engine code.

- [ ] **Step 7: Discard the scratch workspace**

Confirm with the user before deleting anything outside the repo, then remove
`/tmp/aml-tutor-plan003-scratch/`.

- [ ] **Step 8: Final commit**

```bash
git add docs/specs/README.md docs/ARCHITECTURE.md README.md docs/iterations/003-triage-and-eval-lessons/plan.md
git commit -m "docs: iteration 003 plan and ledger regrouping"
git log --oneline -20
```

---

## Definition of done

- `aml-triage/SEED.md` exists (Task 0), and `docs/ARCHITECTURE.md`'s "Known gap" section is gone.
- All eight new lesson specs (008–015) exist, are listed in the ledger as `Done`, grouped under Part
  2 / Part 3 headings, and each has a passing baked-in test — verified individually and together
  (Task 10 Step 4; lesson 012 verified separately per its own `cwd` requirement).
- No baked-in test in `aml-tutor/tests/` requires `ANTHROPIC_API_KEY` or a call to a hosted LLM API
  to pass — every LLM-touching function is exercised through an injected fake client. (Lesson 009's
  test needs network once, for its local embedding model's first-run download — a documented,
  narrower exception, not a violation of this bullet.)
- Lesson 012's spec and baked-in test require the student to produce a real, hand-labeled
  `aml-triage/data/triage_eval_set.jsonl` with at least two distinct decision types used — the lesson
  itself ships a structure-only test and a candidate pool, not a generated scaffold; the labeled file
  is the student's own deliverable when they take the lesson, not something this plan commits.
- `docs/ARCHITECTURE.md` §6's lesson-to-code API table covers all fourteen lessons that produce a
  function (002–015; lesson 001 has no function and was never in this table).
- Lesson 015 documents the manual (non-pytest, real-API-key) step that produces
  `aml-triage/reports/phase3_triage_eval.json`, completing the pair with lesson 007's
  `reports/phase1_report.json`.
- Every new spec (008, 009, 010, 011, 012, 013, 014, 015) includes an "At a regulated shop" callout.
- `npm run check` still passes, confirming this iteration changed no engine behavior.

## Open risks / watch items

- **Scope vs. SEED.md's time budget.** Eight lessons plus real hand-labeling is bigger than a literal
  reading of "Weekend 2: RAG + triage agent + triage eval" — an accepted tradeoff for the "closest to
  prod" choices, not an oversight; flag it again if actual authoring time runs well past a weekend.
- **16 hand-labeled cases vs. SEED.md's 30-50.** Lesson 015's closing note and report should make the
  gap explicit rather than let the smaller number look like the finished target.
- **Model naming drift.** `DEFAULT_MODEL` in lesson 011's reference implementation names a specific
  model id as a placeholder; confirm against current model availability before locking the spec's
  own example commands to it, and keep `TRIAGE_MODEL` as the documented way to point at whatever's
  current without editing code.
- **TF-IDF/hybrid test tuning.** Like `carve_fixture.py`'s `STEP_RANGE` in iteration 002, lessons
  008/009's "ranks TY-001 first/in the top two" tests may need query wording adjusted once actually
  run, if another entry's text happens to overlap more; adjust the test's query, not the corpus.
- **`sentence-transformers`/`anthropic` SDK shapes.** The reference implementations assume specific
  method signatures and response shapes (`model.encode(...)`, `response.content` as a list of blocks
  with `.type`/`.input`) current at time of writing — confirm against whatever versions `uv add`
  resolves to during Tasks 3/5, and adjust parsing if an installed version's shape differs.
- **Lesson 012's inter-rater consistency is unverifiable by pytest.** By design — the pressure test
  says so — but it means a low agreement rate in lesson 015 needs a human (the student, or the tutor
  in conversation) to distinguish "the agent is wrong" from "my own labels were inconsistent" before
  concluding anything about the agent.
