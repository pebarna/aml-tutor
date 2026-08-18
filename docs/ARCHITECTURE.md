# Architecture contract — Iteration 001

Pinned before Phase A touches the engine, per `docs/iterations/001-classifier-tutorial/plan.md`.

## 1. Test ownership

Baked-in pytest suites are authoritative in `aml-tutor/tests/` and are never authored or modified
by the student. Validation commands run them from `aml-triage` by passing the test file path as
`../aml-tutor/tests/NNN_test_*.py` (resolved relative to the tutorial root, not the shell cwd — see
§5), with `cwd` set to `../aml-triage` so `import aml_triage` resolves to the student's package.

## 2. Library stack

XGBoost is the canonical gradient-boosting library for this iteration. LightGBM is out of scope;
no lesson, spec, or dependency list offers it as an alternative.

## 3. SDK auth

`@anthropic-ai/claude-agent-sdk` (0.3.231 at time of writing) wraps the Claude Code CLI binary
rather than implementing its own auth flow. There is no `pi`-style `/login` step to check.

- **Credential mode:** whatever already authenticates the `claude` CLI on this machine —
  `ANTHROPIC_API_KEY`, an existing `claude login` session, or (as on enterprise installs, e.g. this
  org's `init-claude` launcher) `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN` synced into
  `~/.claude/settings.json`'s `env` block. The tutor and doer both use this; neither implements its
  own credential handling.
- **`settingSources` is scoped to `['user']`, never `[]`.** All three `query()` calls (tutor, doer,
  `scripts/setup.mjs`'s probe) set `settings: { disableAllHooks: true }` to block ambient hooks, but
  deliberately keep `settingSources: ["user"]` rather than fully isolating with `[]`. A first version
  used `[]` and it broke exactly the enterprise-install credential mode above: `settingSources: []`
  blocks *all* filesystem settings, including the `env` block a launcher like `init-claude` depends
  on, and the tutor session failed with "Not logged in · Please run /login" on an already-authenticated
  machine. `'project'`/`'local'` stay excluded (no repo `CLAUDE.md` or project hooks leak into the
  tutor/doer's context); the actual security boundary is `canUseTool` plus the explicit
  `allowedTools` allowlist in `claude-agent-adapter.ts`, neither of which `settingSources` affects.
- **Readiness check (`scripts/setup.mjs`):** issue one minimal `query()` call (a short prompt,
  `options: { maxTurns: 2, settingSources: ["user"], settings: { disableAllHooks: true } }`) and
  treat a thrown error or non-`success` result as "not ready." This mirrors what the Pi version did
  with `ModelRuntime.getAvailable()` — a live probe, not a guess from env vars alone, because a
  credential can be present and still be invalid. On failure, print guidance to run `claude login` or
  export `ANTHROPIC_API_KEY` — accurate for a personal machine; on an enterprise install the real fix
  is usually re-running that org's launcher so it can refresh `settings.json`'s `env` block.
- **Model separation:** the tutor keeps a `TUTOR_MODEL`-equivalent override (env var name carried
  over unchanged) so it can be pointed at the largest available model, independent of whatever the
  doer session defaults to.

## 4. Doer tool grant

- Read access: `aml-tutor` (lesson specs, baked-in tests) and `aml-triage` (student code).
- Write/edit access: `aml-triage` only.
- No `Bash` tool. No access outside the two sibling repo roots.
- Enforcement: `disallowedTools` removes the SDK's built-in `Read`/`Write`/`Edit`/`Bash`/`Glob`/
  `Grep` from both the tutor and doer sessions. The only file tools either session gets are custom
  ones built with `tool()` and served via `createSdkMcpServer`, each resolving paths through
  `WorkspaceBoundary` before touching the filesystem — the same audited-wrapper shape
  `workspace-boundary.ts` already uses, just re-registered as SDK MCP tools instead of Pi
  `ToolDefinition`s. `allowedTools` allowlists exactly those `mcp__tutorial__*` names.

## 5. `cwd` resolution

`validationCommands[].cwd` is a path resolved relative to the discovered tutorial root (the
directory containing `README.md` and `docs/specs/`) — never the shell process's cwd. `ValidationCommand`
gains an optional `cwd` field; `ValidationRunner.run` resolves it per-command instead of taking one
fixed `cwd` at construction, since lesson 001's command runs in `aml-triage` while any command that
inspects the tutorial content itself would run in `aml-tutor`.

## 6. Lesson-to-code API

Each lesson produces one small, named public function in `aml-triage/src/aml_triage/`:

| Lesson | Function | Signature |
|---|---|---|
| 002 | `data.load_transactions` | `(path) -> DataFrame` |
| 003 | `split.temporal_split` | `(df, ...) -> (train_df, test_df)` |
| 004 | `features.add_features` | `(df) -> DataFrame` |
| 005 | `imbalance.compute_scale_pos_weight` | `(y) -> float` |
| 006 | `model.train_baseline` | `(X_train, y_train, weight) -> fitted XGBoost` |
| 007 | `evaluate.report` | `(y_true, scores, objective) -> metrics + threshold` |

## Known gap (as of this writing)

Phase C (lessons 002–007) needs a real, small, deterministic slice of the PaySim CSV to carve
fixtures from. Neither `aml-triage/data/` nor anywhere else in this checkout has that CSV, and
`aml-triage/SEED.md` (which `aml-tutor/SEED.md` names as the source for the Phase 1 project
description) does not exist either. Phase C is paused until the CSV (and ideally that SEED.md) are
supplied — see the iteration plan's Phase C task for tracking.
