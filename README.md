# AML triage tutorial

You're going to build an AML/fraud triage system from scratch, across all three phases described
in [`aml-triage/SEED.md`](../aml-triage/SEED.md): a fraud-detection classifier (an XGBoost model
trained on the PaySim mobile-money dataset, with a time-based split, engineered features,
class-imbalance handling, and a chosen operating threshold backed by precision/recall/PR-AUC), a
RAG-backed triage agent that drafts an escalate/monitor/close recommendation with a cited
rationale, and an eval harness that measures both.

This repo is the tutorial: engine, lesson content, and baked-in tests. Your actual code goes into
a sibling repo, [`../aml-triage`](../aml-triage), which is the real deliverable — this repo never
writes there except through the tutor's tools.

## Before you start: sibling repos

This tutorial assumes `aml-tutor` and `aml-triage` are sibling directories:

```
projects/
  aml-tutor/     (this repo)
  aml-triage/    (your classifier, built lesson by lesson)
```

If you cloned them anywhere else, move one so they sit next to each other before running the
tutorial — every validation command it runs resolves `../aml-triage` relative to this repo's root.

## Setup

You need Node.js 24.2+ on the 24.x line, npm 11+, a browser, and [`uv`](https://docs.astral.sh/uv/)
on your `PATH` for the Python side (`aml-triage` itself).

From this repo's root:

```sh
npm install
npm run setup
```

`npm run setup` checks that the Claude Agent SDK can actually reach a model — it wraps the same
`claude` CLI you already use day to day, so there is no separate login step for this tutorial. If
it reports it isn't ready, either run `claude login` or export `ANTHROPIC_API_KEY`, then run
`npm run setup` again.

It also names the two models the tutorial runs:

| Agent | Wants | Chosen with |
| --- | --- | --- |
| The tutor | To teach well, at whatever a good explanation costs | `TUTOR_MODEL` |
| The doer (writes into `aml-triage` on your behalf) | To be cheap and fast | left to its default |

Leave `TUTOR_MODEL` unset and the SDK picks the tutor's model for you. Name a model that doesn't
exist or isn't authenticated and the tutor falls back to that default rather than failing;
`npm run setup` reports whichever happened.

```sh
export TUTOR_MODEL=<provider>/<model>
```

## Start the tutorial

```sh
npm run tutorial
```

This opens the local tutor in your browser. It listens on loopback only; if no browser opens,
visit the printed address yourself. The tutor has no authentication and edits `aml-triage`'s
working tree, so only run it on a network you trust.

Leave it running and open a second terminal, inside `aml-triage`, for the commands each lesson
gives you — `uv init`, `uv add`, `uv run pytest`, and so on. Every command is written to run from
`aml-triage`'s own root.

## The doer, and its boundary

Each lesson can offer to do a step for you instead of walking you through it by hand. That doer:

- can read this repo (lesson specs, baked-in tests) and `aml-triage` (your code);
- can only write or edit files inside `aml-triage`;
- has no shell access — it cannot run `uv`, `pytest`, or anything else, which is why lesson 001
  documents exactly what it writes by hand when it stands in for `uv init`/`uv add`.

Full detail on why the boundary is drawn there is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Checks are baked in, not judged on the fly

Every lesson that produces code ships with its own `pytest` file in [`tests/`](tests), written
before you start the lesson. The tutor runs it via one fixed command per lesson — you can see
exactly what will be checked before you write anything. Comprehension questions, by contrast, are
judged by the tutor in conversation: code correctness and understanding are deliberately checked
two different ways.

## Where to begin, and how to resume

The first time you run `npm run tutorial`, it asks whether to start fresh or resume. Resuming
keeps `aml-triage`'s files exactly as they are and asks a fresh tutor process to inspect them
before continuing — nothing in `aml-triage` is ever deleted or reset for you. There is no
"start over" option here on purpose: unlike a disposable kata workspace, `aml-triage` is your real
project.

## Lesson ledger

See [`docs/specs/README.md`](docs/specs/README.md) for the full lesson list and authoring status.

## Inspiration

This tutorial's engine and teaching pattern are forked from
[`software-factory-tutorial`](../software-factory-tutorial); see that repo for the pattern this one
reuses across a different domain and a different repo boundary.
