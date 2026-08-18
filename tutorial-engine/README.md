# Tutorial engine

A local, browser-led tutorial runner. It embeds `@anthropic-ai/claude-agent-sdk` directly as a TypeScript SDK; it does not shell out to the `claude` CLI itself beyond what the SDK does internally.

## Run a tutorial

Point the engine at the tutorial directory (this repository) and, implicitly, its sibling `../aml-triage`:

```sh
cd tutorial-engine
npm install
npm run dev -- ..
```

Add `--triage <path>` to point at a differently-located `aml-triage` checkout, `--no-open` to suppress browser launch, or `--port 4310` to choose a port. The server binds only to `127.0.0.1`. Model credentials remain in the server process; the browser has no filesystem or provider-credential access.

Keep the launching terminal open. It prints timestamped startup, browser, tutor, tool, validation, and shutdown events. It also saves each run to `~/Library/Logs/AmlTutor/` on macOS (or `$XDG_STATE_HOME/aml-tutor/` elsewhere); the terminal prints the exact path. While the tutor is working, a heartbeat every 15 seconds names its current activity, so a browser spinner always has a corresponding server-side status.

## Tutorial convention

A tutorial needs no engine configuration file. The engine infers it from the directory:

- the first `#` heading in `README.md` is the title and the README is the whole-exercise orientation;
- `docs/specs/README.md` is the lesson ledger, one flat list (this tutorial has no "Part 1 / Part 2" split);
- the first lesson not recorded as finished is the current one;
- the linked spec tells the tutor what to teach, and may embed a fenced ` ```json validation ``` ` block naming its baked-in check.

The tutor reads those files, guides one small step at a time, and offers to let the learner make a change or delegate it to the doer — a separate, one-shot session with write access to `../aml-triage` only.

## Commands

```sh
npm run build  # compile server and browser client
npm test       # unit tests
npm run check  # TypeScript and tests
```
