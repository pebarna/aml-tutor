# AML classifier tutorial lessons

Seven lessons, one linear sequence — no parts. Each lesson writes into `../aml-triage`, the
sibling repo that is the actual Phase 1 deliverable; this repo holds only the tutorial content
and engine.

The **Status** column tracks *authoring* completeness for tutorial maintainers (has the spec, its
baked-in test, and — where relevant — its fixture been written and does the test pass against a
freshly scaffolded `aml-triage`?). It is not read by the tutorial engine and carries no meaning for
an individual learner's progress — that lives in `aml-triage`'s own git history plus the engine's
local progress store, not in this table.

| Lesson | Goal | Status |
| --- | --- | --- |
| [001](001-project-setup.md) | Project setup — `uv`, package layout, dependencies | Done |
| [002](002-load-and-explore-the-data.md) | Load and explore the data | Done |
| [003](003-time-based-split.md) | Time-based train/test split | Done |
| [004](004-feature-engineering.md) | Feature engineering | Done |
| [005](005-class-imbalance.md) | Class imbalance handling | Done |
| [006](006-train-the-baseline.md) | Train the XGBoost baseline | Done |
| [007](007-evaluation-and-threshold.md) | Evaluation and threshold selection | Done |
| [008](008-typology-retrieval.md) | Typology retrieval (keyword/TF-IDF half) | Done |
| [009](009-hybrid-retrieval.md) | Hybrid retrieval (embeddings + blending) | Done |
| [010](010-structured-triage-decisions.md) | Structured triage decisions | Done |
| [011](011-the-triage-agent.md) | The end-to-end triage agent | Done |
| [012](012-the-hand-labeled-eval-set.md) | The hand-labeled triage eval set | Done |
