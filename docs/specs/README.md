# AML triage tutorial lessons

Fifteen lessons across the three phases `aml-triage/SEED.md` describes — grouped below as Part 1
(the classifier), Part 2 (the triage agent), and Part 3 (the eval harness) for readability, but still
one linear sequence: each lesson's baked-in test assumes every earlier lesson's function already
exists in `aml-triage`.

The **Status** column tracks *authoring* completeness for tutorial maintainers (has the spec, its
baked-in test, and — where relevant — its fixture been written and does the test pass against a
freshly scaffolded `aml-triage`?). It is not read by the tutorial engine and carries no meaning for
an individual learner's progress — that lives in `aml-triage`'s own git history plus the engine's
local progress store, not in this table.

### Part 1 — the classifier

| Lesson | Goal | Status |
| --- | --- | --- |
| [001](001-project-setup.md) | Project setup — `uv`, package layout, dependencies | Done |
| [002](002-load-and-explore-the-data.md) | Load and explore the data | Done |
| [003](003-time-based-split.md) | Time-based train/test split | Done |
| [004](004-feature-engineering.md) | Feature engineering | Done |
| [005](005-class-imbalance.md) | Class imbalance handling | Done |
| [006](006-train-the-baseline.md) | Train the XGBoost baseline | Done |
| [007](007-evaluation-and-threshold.md) | Evaluation and threshold selection | Done |

### Part 2 — the triage agent

| Lesson | Goal | Status |
| --- | --- | --- |
| [008](008-typology-retrieval.md) | Typology retrieval (keyword/TF-IDF half) | Done |
| [009](009-hybrid-retrieval.md) | Hybrid retrieval (embeddings + blending) | Done |
| [010](010-structured-triage-decisions.md) | Structured triage decisions | Done |
| [011](011-the-triage-agent.md) | The end-to-end triage agent | Done |

### Part 3 — the eval harness

| Lesson | Goal | Status |
| --- | --- | --- |
| [012](012-the-hand-labeled-eval-set.md) | The hand-labeled triage eval set | Done |
| [013](013-deterministic-triage-checks.md) | Deterministic triage checks | Done |
| [014](014-llm-as-judge.md) | LLM-as-judge scoring | Done |
| [015](015-the-agreement-rate-report.md) | The agreement-rate report | Done |
