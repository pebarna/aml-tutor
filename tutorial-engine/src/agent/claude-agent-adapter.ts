import {
  createSdkMcpServer,
  query,
  tool,
  type CanUseTool,
  type Options,
  type Query,
  type SdkMcpToolDefinition,
  type SDKMessage,
  type SDKUserMessage
} from "@anthropic-ai/claude-agent-sdk";
import { isAbsolute, relative } from "node:path";
import { z } from "zod";
import type { LessonDefinition } from "../lesson/contract.js";
import type { RunState, TutorialEvent } from "../protocol/events.js";
import { TutorialEventBus } from "../protocol/event-bus.js";
import { ValidationRunner } from "../validation/runner.js";
import { ChoiceManager } from "./choice-manager.js";
import { createTutorialTools } from "./tutorial-tools.js";
import { createWorkspaceTools, WorkspaceBoundary } from "./workspace-boundary.js";
import type { TutorialLogger } from "../runtime-log.js";

/** The MCP server name every tutor tool is registered under; tool names on the wire are `mcp__tutorial__<name>`. */
const TUTOR_SERVER_NAME = "tutorial";
/** The MCP server name every doer tool is registered under; tool names on the wire are `mcp__triage__<name>`. */
const DOER_SERVER_NAME = "triage";

const TUTOR_TOOL_NAMES = [
  "read", "grep", "find", "ls",
  "present_markdown", "present_diagram", "offer_choices", "run_validation", "show_file_excerpt", "complete_lesson",
  "automate_step"
];
const DOER_TOOL_NAMES = ["read", "write", "edit", "move", "grep", "find", "ls"];

/**
 * The SDK's built-in tools, removed from both sessions so the only file or
 * shell access either one has is through our own boundary-audited MCP tools
 * (ARCHITECTURE.md §4). `tools: []` on `Options` already excludes every
 * built-in from the model's tool list; this is defence in depth in case a
 * future SDK version widens that default.
 */
const BUILTIN_DISALLOWED = ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebFetch", "WebSearch", "Task", "NotebookEdit"];

/**
 * Names the tutor's model. Deliberately separate from the doer's: the doer
 * (see `runDoer` below) is meant to be cheap and fast — and whose mistakes are
 * part of the lesson — while the tutor wants the opposite. The env var name
 * is carried over unchanged from the Pi-based engine this was forked from;
 * `scripts/setup.mjs` reports the same resolution to the learner.
 */
export const TUTOR_MODEL_ENV = "TUTOR_MODEL";

/**
 * Resolve the tutor's model from the environment.
 *
 * The Pi-based original cross-checked a requested model against a live
 * registry of authenticated providers before committing to it. The Claude
 * Agent SDK has no equivalent registry call on the `Options` surface — it
 * wraps the `claude` CLI's own model resolution, which validates lazily on
 * first use. So this is deliberately thinner than the original: it trims the
 * env var and passes it straight through as `Options.model`, and leaves
 * "is this model actually usable" to the one live probe `scripts/setup.mjs`
 * already runs, plus `Options.fallbackModel` if the primary is unavailable
 * at request time.
 */
export function resolveTutorModel(requested: string | undefined): { model?: string } {
  const wanted = requested?.trim();
  return wanted ? { model: wanted } : {};
}

/** Log operational identifiers, but never tool content or learner chat text. */
function toolDetail(args: unknown): string {
  if (!args || typeof args !== "object") return "";
  const value = args as Record<string, unknown>;
  if (typeof value.path === "string") return ` for ${value.path}`;
  if (typeof value.commandId === "string") return ` for validation ${value.commandId}`;
  return "";
}

/**
 * The same identifiers as `toolDetail`, shortened for the learner's spinner:
 * the workspace prefix is noise on screen, where every path is inside it.
 * Content and chat text stay out of this for the same reason as the log.
 */
export function activityDetail(args: unknown, workspace: string): string {
  if (!args || typeof args !== "object") return "";
  const value = args as Record<string, unknown>;
  if (typeof value.path === "string") return ` ${isAbsolute(value.path) ? relative(workspace, value.path) || value.path : value.path}`;
  if (typeof value.commandId === "string") return ` ${value.commandId}`;
  return "";
}

/** Keep a caption to a glanceable length: the learner has the log for the rest. */
export function summarise(actions: readonly string[], limit = 3): string {
  if (actions.length <= limit) return actions.join(", ");
  return `${actions.slice(0, limit).join(", ")} and ${actions.length - limit} more`;
}

/**
 * The tutor's system prompt: patient, one step at a time, AML/uv/pytest
 * framing. Replaces the calculator tutorial's Kent-Beck/Pi-CLI framing
 * (plan.md Phase A step 7) with fresh prose for this curriculum's own tone —
 * see SEED.md and the iteration plan's "Why these choices" for what that
 * tone should be, not a find/replace of the original's wording.
 */
export function coachingSystemPrompt(lesson: LessonDefinition, currentSpec?: string): string {
  // How far the learner has got is the engine's to know: it lives in the
  // engine's own hidden state directory, outside the curriculum, and naming
  // the file here saves the tutor working it out. Without one — an exhausted
  // or unreadable ledger — fall back to asking for the first unfinished
  // lesson rather than opening nothing.
  const routing = currentSpec
    ? `then ${currentSpec}, which is the specification for the lesson the learner is on`
    : "then the first specification the learner has not finished";

  return `You are a patient tutor for "${lesson.title}". The learner has a strong software-engineering background but is new to Python data tooling and applied machine learning. They are building a real PaySim-based XGBoost fraud classifier from scratch, in their own sibling repository \`../aml-triage\` — that repository, not anything inside this one, is what they walk away with.

At the beginning, silently read README.md, then docs/specs/README.md, ${routing}. The ledger and specifications are your routing information, not the learner's lesson: do not mention the ledger, lesson numbers, or those file paths unless the learner asks. Orient the learner in plain language from the README before discussing implementation. Read no \`aml-triage\` source until the current spec requires it. Introduce only the vocabulary the current specification uses; a later lesson's words are that lesson's to teach.

Teach only the current lesson, one small step at a time, in the implementation order stated by the current specification. Explain what each step achieves before explaining how, and connect it back to why it matters for a fraud classifier specifically — accuracy being meaningless under class imbalance, why a time-based split avoids leaking the future into training, why gradient boosting suits tabular transaction data — rather than teaching pandas or XGBoost as abstract API surface.

A specification's phrasing is not always the clearest way to say a thing. Where it states a principle figuratively, teach the mechanism instead: name what actually runs, what is written, what is read, and which capability was removed. Do not repeat a figure of speech the learner would have to decode, and never stack two of them in one sentence. The learner should be able to predict what the line will do next, not recall a slogan.

Every offer_choices option must supply an icon category. Use the standard mapping: "I'll do it"=do; "Make it for me"=automate; "I've made this step"=confirm; "Show me exactly what to type"=show; and "Make this step for me"=automate. Use pause for a stop or pause choice.

For a new change, use offer_choices to offer "I'll do it" and "Make it for me." If the learner selects "I'll do it," first use present_markdown to give a short conceptual outline of the few moves ahead. Then immediately begin the first guided step. Name the file and relevant nearby code, explain the intent, and show a small code snippet the learner can type. Do not give a large finished-file replacement. After every guided step, use offer_choices with these labels: "I've made this step", "Show me exactly what to type", and "Make this step for me". If they ask for exact typing instructions, give the precise small edit. If the learner asks you to make a step for them, call automate_step with precise, self-contained instructions for exactly that step: the doer it runs has no memory of this conversation, so name the file, the function signature, and the behaviour it must satisfy. If the learner says they are done or asks for feedback, read the relevant file and compare it to the current spec. If they say it is not working, inspect the relevant files and evidence before offering a correction.

Every lesson's "Checks" section pairs a deterministic pytest command with two or three comprehension questions. Ask them in your own words, in conversation, once the implementation step they cover is in place; do not accept a memorized-sounding answer without asking the learner to apply it to this lesson's own data or code. Code correctness is what run_validation exists for — never take the learner's word that a baked-in test passes; run it yourself with run_validation and read its output before saying so. Leave validation, error handling, and defensive code until the learner's current lesson teaches it or it becomes necessary. Do not make changes unless the learner explicitly chooses that option.

Do not act as the doer yourself: you have no write access to \`aml-triage\`, only automate_step does, and only when the learner asks for it. Do not run \`uv\` or \`pytest\` commands narratively — run_validation is the only way a check actually runs. Keep the transcript calm: use present_markdown for teaching, present_diagram for flows, and show_file_excerpt only for small relevant excerpts. Do not expose secrets or read outside \`aml-tutor\` and \`aml-triage\`.

When the current specification creates no files, do not offer to build anything and do not invent an artefact to make the lesson feel like the others. Work through what it asks the learner to run, notice, or answer; treat its checks as questions the learner answers in their own words, and confirm or correct those answers against the specification.

At the end of every lesson, stop there. Recap what the learner built, then use complete_lesson once, and then offer a choice between pausing for now and continuing to the next lesson. Do not begin the next lesson until that choice is made. A lesson is finished when its steps are done and its baked-in test passes, whether the learner made the changes or automate_step did, and whether or not they continue immediately: record it before the choice, not after, so pausing still leaves the outline correct. Do not announce the tool or describe the outline moving; it is bookkeeping, and the learner can see it.

When the current specification is the last lesson (evaluation and threshold selection), stop with the stronger, more specific version of that beat instead: recap the whole classifier the learner built — its precision, recall, PR-AUC, and chosen operating threshold — say plainly that Phase 1 is complete and defensible even if nothing further is built, use complete_lesson, and offer a choice between finishing for now and reviewing the classifier's final report together.`;
}

/**
 * The doer's system prompt: no tutoring framing at all, because the doer has
 * no conversation with the learner — it receives one self-contained
 * instruction from the tutor (via `automate_step`) and either finishes or
 * fails, then is discarded.
 */
function doerSystemPrompt(): string {
  return `You are a focused implementation assistant. Your current working directory is the learner's real Python project, \`aml-triage\`; you may also read a second, read-only directory, \`aml-tutor\`, which holds this tutorial's lesson specifications and baked-in pytest files so you know what "done" looks like. You may write, edit, or move files only inside \`aml-triage\` — never inside \`aml-tutor\`. You have no shell or bash access: every filesystem change goes through your read/write/edit/move tools, which enforce this boundary regardless of what you ask for. Implement exactly the instructions you are given for the current tutorial step, nothing more and nothing less. Do not add error handling, tests, or extra abstraction the instructions did not ask for. When you believe the step is complete, say so in one short sentence describing what you changed; that sentence is the only part of your work the learner's tutor will see.`;
}

/** Every allowed root, resolved once, that both the tutor and the doer sessions are constructed from. */
export interface TutorialRoots {
  /** The tutorial workspace (this repository) — read-only for both sessions. */
  tutorial: string;
  /** The sibling `aml-triage` repository — the learner's real deliverable; writable only by the doer. */
  triage: string;
}

function canUseToolAllowing(allowed: readonly string[]): CanUseTool {
  const allowedSet = new Set(allowed);
  return async (toolName) => {
    if (allowedSet.has(toolName)) return { behavior: "allow" };
    return { behavior: "deny", message: `${toolName} is not available in this session.` };
  };
}

/**
 * Run one doer turn to completion and return its final report.
 *
 * This is the Claude Agent SDK's replacement for the Pi-based tutorial's
 * `pi -p` doer: there, the *learner* ran a documented shell command by hand.
 * Here, the tutor calls this from the `automate_step` tool on the learner's
 * behalf, so the doer is one-shot and non-interactive rather than a command
 * the learner types. It gets its own `WorkspaceBoundary` (read both repos,
 * write only `aml-triage`) and its own MCP server; nothing it does can be
 * confused with the tutor's own tool calls, and it never sees the tutor's
 * conversation.
 */
export async function buildDoerBoundary(roots: TutorialRoots): Promise<WorkspaceBoundary> {
  return WorkspaceBoundary.create({ primary: roots.triage, readRoots: [roots.tutorial, roots.triage], writeRoots: [roots.triage] });
}

/**
 * Assemble the doer's `Options`, without calling `query()`. Split out so the
 * smoke test required by plan.md step 9 (§4's tool grant: read both repos,
 * write only `aml-triage`, no bash) can inspect exactly what governs tool
 * access — `tools`, `disallowedTools`, `allowedTools`, and `canUseTool` — and
 * exercise the boundary's actual filesystem behaviour, without needing a real
 * model call.
 */
export function buildDoerOptions(roots: TutorialRoots, boundary: WorkspaceBoundary, instructions: string): { prompt: string; options: Options } {
  const doerServer = createSdkMcpServer({
    name: DOER_SERVER_NAME,
    tools: createWorkspaceTools(boundary, () => {}, { write: true })
  });
  const allowedTools = DOER_TOOL_NAMES.map((name) => `mcp__${DOER_SERVER_NAME}__${name}`);
  return {
    prompt: instructions,
    options: {
      cwd: roots.triage,
      additionalDirectories: [roots.tutorial],
      systemPrompt: doerSystemPrompt(),
      tools: [] as string[],
      disallowedTools: BUILTIN_DISALLOWED,
      mcpServers: { [DOER_SERVER_NAME]: doerServer },
      allowedTools,
      canUseTool: canUseToolAllowing(allowedTools),
      maxTurns: 40,
      // Same isolation as the tutor session above, and for the same reason.
      settingSources: ["user"],
      settings: { disableAllHooks: true }
    }
  };
}

export async function runDoer(roots: TutorialRoots, instructions: string, log: TutorialLogger): Promise<string> {
  const boundary = await buildDoerBoundary(roots);
  log.info(`Doer starting on ../${roots.triage.split("/").pop()} for: ${instructions.slice(0, 120)}${instructions.length > 120 ? "…" : ""}`);
  const doerQuery = query(buildDoerOptions(roots, boundary, instructions));
  let summary = "The doer finished without reporting a result.";
  for await (const message of doerQuery as AsyncIterable<SDKMessage>) {
    if (message.type === "result") {
      summary = message.subtype === "success" ? message.result : `The doer could not finish: ${message.subtype}.`;
    }
  }
  log.info(`Doer finished: ${summary.slice(0, 200)}`);
  return summary;
}

/** The `automate_step` tool: the tutor's only path to changing `aml-triage`, always by delegating to a fresh doer. */
function createAutomateStepTool(roots: TutorialRoots, log: TutorialLogger, emit: (event: TutorialEvent) => void) {
  return tool(
    "automate_step",
    "Delegate the current step to the doer: a separate, one-shot assistant with write access to ../aml-triage only (no shell access, and no memory of this conversation). Use this when the learner chooses 'Make it for me'. Give it precise, self-contained instructions for exactly the current step.",
    { instructions: z.string().min(1).max(8_000) },
    async (params) => {
      const summary = await runDoer(roots, params.instructions, log);
      emit({ type: "presentation", presentation: { kind: "markdown", title: "The doer's report", markdown: summary } });
      return { content: [{ type: "text" as const, text: summary }] };
    }
  );
}

/** A minimal streaming-input source `query()` can iterate, that `.chat()` can push onto after construction. */
class UserMessageQueue implements AsyncIterable<SDKUserMessage> {
  readonly #pending: SDKUserMessage[] = [];
  #waiting: ((result: IteratorResult<SDKUserMessage>) => void) | undefined;
  #closed = false;

  push(message: SDKUserMessage): void {
    if (this.#closed) return;
    if (this.#waiting) {
      const resolve = this.#waiting;
      this.#waiting = undefined;
      resolve({ value: message, done: false });
    } else {
      this.#pending.push(message);
    }
  }

  close(): void {
    if (this.#closed) return;
    this.#closed = true;
    if (this.#waiting) {
      const resolve = this.#waiting;
      this.#waiting = undefined;
      resolve({ value: undefined as unknown as SDKUserMessage, done: true });
    }
  }

  [Symbol.asyncIterator](): AsyncIterator<SDKUserMessage> {
    return {
      next: (): Promise<IteratorResult<SDKUserMessage>> => {
        if (this.#pending.length > 0) return Promise.resolve({ value: this.#pending.shift()!, done: false });
        if (this.#closed) return Promise.resolve({ value: undefined as unknown as SDKUserMessage, done: true });
        return new Promise((resolve) => { this.#waiting = resolve; });
      }
    };
  }
}

function userMessage(text: string, priority: "now" | "next"): SDKUserMessage {
  return { type: "user", message: { role: "user", content: text }, parent_tool_use_id: null, priority };
}

export class ClaudeTutorialAdapter {
  readonly choices = new ChoiceManager();
  readonly validation: ValidationRunner;
  readonly #bus: TutorialEventBus;
  readonly #roots: TutorialRoots;
  readonly #queue = new UserMessageQueue();
  #query!: Query;
  #state: RunState = "idle";
  #activity = "waiting for the tutor";
  #workingSince = 0;
  #heartbeat?: NodeJS.Timeout;
  #runningTools = new Map<string, string>();
  #batch: string[] = [];
  #respondingMessages = new Set<string>();
  /**
   * The SDK mints a fresh `uuid` on every `stream_event` envelope — it identifies
   * that one streaming chunk, not the assistant turn it belongs to (unlike Pi's
   * `message.id`, which stayed stable across a whole turn). Minting our own id
   * from the first delta of a turn and reusing it for every later delta plus the
   * final `assistant-message` is what lets the browser append into one bubble
   * instead of opening a new one per chunk.
   */
  #currentStreamMessageId: string | undefined;
  #eventLoop!: Promise<void>;

  private constructor(readonly lesson: LessonDefinition, readonly workspace: string, roots: TutorialRoots, bus: TutorialEventBus, private readonly log: TutorialLogger) {
    this.#bus = bus;
    this.#roots = roots;
    this.validation = new ValidationRunner(lesson.validationCommands, workspace);
  }

  static async create(lesson: LessonDefinition, workspace: string, triageWorkspace: string, bus: TutorialEventBus, log: TutorialLogger, currentSpec?: string): Promise<ClaudeTutorialAdapter> {
    log.info(`Resolving tutorial workspace ${workspace} and triage workspace ${triageWorkspace}.`);
    const roots: TutorialRoots = { tutorial: workspace, triage: triageWorkspace };
    const boundary = await WorkspaceBoundary.create({ primary: workspace, readRoots: [workspace, triageWorkspace], writeRoots: [] });
    const canonicalWorkspace = boundary.root;
    const adapter = new ClaudeTutorialAdapter(lesson, canonicalWorkspace, roots, bus, log);

    log.info("Creating restricted tutor tools.");
    const workspaceTools = createWorkspaceTools(boundary, (event) => bus.publish(event), { write: false });
    const tutorialTools = createTutorialTools({
      lesson,
      workspace: canonicalWorkspace,
      choices: adapter.choices,
      validation: adapter.validation,
      boundary,
      emit: (event) => bus.publish(event),
      setRunState: (state) => adapter.setState(state)
    });
    const automateStep = createAutomateStepTool(roots, log, (event) => bus.publish(event));
    const tools = [...workspaceTools, ...tutorialTools, automateStep].map((definition) => instrumented(definition, adapter));
    const allowedTools = TUTOR_TOOL_NAMES.map((name) => `mcp__${TUTOR_SERVER_NAME}__${name}`);

    const { model } = resolveTutorModel(process.env[TUTOR_MODEL_ENV]);
    log.info("Creating Claude Agent SDK tutor session.");
    adapter.#query = query({
      prompt: adapter.#queue,
      options: {
        cwd: canonicalWorkspace,
        additionalDirectories: [triageWorkspace],
        systemPrompt: coachingSystemPrompt(lesson, currentSpec),
        model,
        tools: [],
        disallowedTools: BUILTIN_DISALLOWED,
        mcpServers: { [TUTOR_SERVER_NAME]: createSdkMcpServer({ name: TUTOR_SERVER_NAME, tools }) },
        allowedTools,
        canUseTool: canUseToolAllowing(allowedTools),
        includePartialMessages: true,
        persistSession: false,
        // Isolate the tutor from ambient hooks and (via disableAllHooks) any
        // stray permission-prompt/hook round-trip that would otherwise burn a
        // turn or real API cost on every session (observed during testing).
        //
        // Deliberately scoped to disableAllHooks rather than a blanket
        // settingSources: []. This machine's Claude Code auth is not
        // ANTHROPIC_API_KEY or `claude login` — it's whatever `init-claude`
        // (this org's installer) synced into ~/.claude/settings.json's `env`
        // block (ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN), which only a
        // spawned `claude` CLI process reading 'user' settings ever sees.
        // settingSources: [] blocked that block too, which is what produced
        // "Not logged in · Please run /login" in the browser even though the
        // machine itself was authenticated. 'project'/'local' stay excluded
        // (no repo CLAUDE.md/hooks leak in); the actual security boundary
        // here is canUseTool + the explicit allowedTools list above, neither
        // of which settingSources affects.
        settingSources: ["user"],
        settings: { disableAllHooks: true }
      }
    });
    adapter.#eventLoop = adapter.consume();
    log.info(`Tutor session created${model ? ` on ${model}` : ""}; event monitoring is active.`);
    return adapter;
  }

  get state(): RunState { return this.#state; }

  /** What the spinner should say, for a browser that connects mid-turn. */
  get activity(): string { return this.#activity; }

  async begin(): Promise<void> {
    this.log.info("Submitting the initial welcome request to the tutor.");
    await this.chat("Begin the tutorial. Silently identify the current lesson. Welcome the learner in plain language, present its flow, then offer exactly one first-step choice.", "steer", false);
  }

  async resume(): Promise<void> {
    this.log.info("Submitting the saved-session continuation request to the tutor.");
    await this.chat("The learner has resumed a saved tutorial. Their previous browser transcript is visible to them, but this is a fresh tutor process. Inspect ../aml-triage and the current specification, briefly identify the next unfinished small step, then offer exactly one appropriate choice. Do not repeat the full welcome or assume an unfinished choice is still active.", "steer", false);
  }

  async chat(text: string, delivery: "steer" | "followUp" = "steer", showInTranscript = true): Promise<void> {
    if (!text.trim() || text.length > 12_000) throw new Error("Chat messages must be between 1 and 12,000 characters.");
    if (showInTranscript) this.#bus.publish({ type: "user-message", markdown: text });
    if (this.#state === "awaiting-choice") {
      const pendingChoices = this.choices.pendingIds;
      this.log.info(`Learner request arrived while awaiting choice ${pendingChoices.join(", ") || "(not yet registered)"}; it will supersede that choice.`);
      const superseded = this.choices.cancelAll("Learner message superseded this choice.");
      if (superseded.length) this.log.info("Learner message superseded the outstanding choice; releasing the tutor to respond.");
    }
    this.log.info(`Submitting ${showInTranscript ? "learner" : "initial"} request to the tutor (${text.length} characters; ${delivery}).`);
    this.setState("working");
    try {
      // "steer" interrupts whatever the tutor is doing before its message is
      // delivered, which is the closest approximation this SDK offers to Pi's
      // `streamingBehavior: "steer"`; "followUp" only queues the message with
      // `priority: "next"` and lets the current turn finish first. There is no
      // exact equivalent of Pi's distinct steer-vs-queue message ordering, so
      // this is a best-effort mapping, not a literal port.
      if (delivery === "steer") await this.#query.interrupt().catch(() => {});
      this.#queue.push(userMessage(text, delivery === "steer" ? "now" : "next"));
    } catch (error) {
      this.log.error("The tutor could not process the request", error);
      this.setState("failed");
      this.#bus.publish({ type: "error", message: error instanceof Error ? error.message : "The tutor failed to start.", retryable: true });
    }
  }

  choose(choiceId: string, optionId: string): void {
    if (!this.choices.choose(choiceId, optionId)) throw new Error("That choice is no longer available.");
  }

  async runValidation(commandId: string): Promise<void> {
    const command = this.lesson.validationCommands.find((item) => item.id === commandId);
    if (!command) throw new Error(`Validation command '${commandId}' is not allowed.`);
    this.log.info(`Starting validation "${command.label}" (${commandId}).`);
    this.setState("working", `running validation "${command.label}"`);
    try {
      const result = await this.validation.run(commandId, (text) => this.#bus.publish({ type: "tool-progress", toolId: `validation:${commandId}`, text }));
      this.log.info(`Validation "${command.label}" ${result.passed ? "passed" : "failed"} in ${result.durationMs}ms (exit ${result.exitCode ?? "no code"}).`);
      this.#bus.publish({ type: "validation", id: command.id, label: command.label, ...result });
      this.setState("idle");
    } catch (error) {
      this.log.error(`Validation "${command.label}" could not run`, error);
      this.setState("failed");
      this.#bus.publish({ type: "error", message: error instanceof Error ? error.message : "Validation could not start.", retryable: true });
    }
  }

  async abort(): Promise<void> {
    this.log.info("Aborting the current tutor request at the learner's request.");
    this.choices.cancelAll("Learner stopped the current step.");
    await this.#query.interrupt().catch(() => {});
    this.setState("idle");
  }

  dispose(): void {
    this.log.info("Disposing the tutor session.");
    this.stopHeartbeat();
    this.choices.cancelAll("Tutorial server stopped.");
    this.#queue.close();
    this.#query.close();
  }

  private setState(state: RunState, activity = "waiting for the tutor"): void {
    const changed = this.#state !== state;
    this.#state = state;
    if (state === "working") {
      this.setActivity(activity);
      if (changed) this.startHeartbeat();
    } else {
      this.stopHeartbeat();
    }
    if (changed) this.log.info(`Tutor state: ${state}${state === "working" ? ` (${this.#activity}).` : "."}`);
    this.#bus.publish({ type: "run-state", state });
  }

  /**
   * The one line the learner sees under the spinner, and the one the heartbeat
   * repeats into the log. Phrased so both read well: "running read on
   * README.md" becomes "Tutor is still running read on README.md (30 seconds)"
   * in the log and "Running read on README.md…" in the browser.
   */
  private setActivity(activity: string): void {
    if (this.#activity === activity) return;
    this.#activity = activity;
    this.#bus.publish({ type: "activity", text: activity });
  }

  private describeToolActivity(): void {
    const running = [...this.#runningTools.values()];
    if (running.length === 1) return this.setActivity(`running ${running[0]}`);
    if (running.length > 1) return this.setActivity(`running ${running.length} tools: ${summarise(running)}`);
    this.setActivity(this.#batch.length > 0 ? `waiting for the tutor… ${summarise(this.#batch)}` : "waiting for the tutor");
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.#workingSince = Date.now();
    this.#heartbeat = setInterval(() => {
      this.log.info(`Tutor is still working: ${this.#activity} (${Math.round((Date.now() - this.#workingSince) / 1_000)} seconds).`);
    }, 15_000);
    this.#heartbeat.unref();
  }

  private stopHeartbeat(): void {
    if (this.#heartbeat) clearInterval(this.#heartbeat);
    this.#heartbeat = undefined;
  }

  /** Invoked by the `instrumented()` wrapper around every MCP tool, before it runs. */
  onToolStart(id: string, name: string, args: unknown): void {
    if (this.#runningTools.size === 0) this.#batch = [];
    const display = `${name}${activityDetail(args, this.workspace)}`;
    this.#runningTools.set(id, display);
    this.describeToolActivity();
    this.log.info(`Tutor started tool "${name}"${toolDetail(args)} (${id}).`);
    this.#bus.publish({ type: "tool-start", tool: { id, name, label: name } });
  }

  /** Invoked by `instrumented()` after a tool call settles, successfully or not. */
  onToolEnd(id: string, name: string, outcome: { text: string; isError: boolean }): void {
    const display = this.#runningTools.get(id);
    if (display) this.#batch.push(display);
    this.#runningTools.delete(id);
    this.describeToolActivity();
    if (outcome.isError) {
      this.log.info(`Tutor tool "${name}" failed: ${outcome.text || `${name} failed.`}`);
      this.#bus.publish({ type: "tool-error", toolId: id, message: outcome.text || `${name} failed.`, retryable: true });
    } else {
      const detail = name === "run_validation" && outcome.text ? `: ${outcome.text}` : "";
      this.log.info(`Tutor completed tool "${name}"${detail}.`);
      this.#bus.publish({ type: "tool-complete", toolId: id, summary: outcome.text });
    }
  }

  /** Drains the Query's async-generator side, translating SDK messages into TutorialEvents. */
  private async consume(): Promise<void> {
    try {
      for await (const message of this.#query as AsyncIterable<SDKMessage>) {
        this.onMessage(message);
      }
    } catch (error) {
      this.log.error("The tutor session ended unexpectedly", error);
      this.setState("failed");
      this.#bus.publish({ type: "error", message: error instanceof Error ? error.message : "The tutor session ended unexpectedly.", retryable: true });
    }
  }

  private onMessage(message: SDKMessage): void {
    if (message.type === "stream_event") {
      const event = message.event as { type?: string; delta?: { type?: string; text?: string } };
      if (event.type === "content_block_delta" && event.delta?.type === "text_delta" && typeof event.delta.text === "string") {
        if (!this.#currentStreamMessageId) {
          this.#currentStreamMessageId = message.uuid;
          this.#respondingMessages.add(this.#currentStreamMessageId);
          this.setActivity("receiving the tutor's response");
          this.log.info("Tutor started responding.");
        }
        this.#bus.publish({ type: "assistant-delta", messageId: this.#currentStreamMessageId, delta: event.delta.text });
      }
      return;
    }
    if (message.type === "assistant") {
      const content = (message.message as { content?: Array<{ type: string; text?: string }> }).content ?? [];
      const markdown = content.filter((item) => item.type === "text").map((item) => item.text ?? "").join("");
      if (markdown) {
        const messageId = this.#currentStreamMessageId ?? message.uuid;
        this.#respondingMessages.delete(messageId);
        this.#currentStreamMessageId = undefined;
        this.log.info(`Tutor completed a response (${markdown.length} characters).`);
        this.#bus.publish({ type: "assistant-message", messageId, markdown });
      }
      return;
    }
    if (message.type === "result") {
      this.log.info(`Tutor turn settled (${message.subtype}).`);
      if (message.subtype !== "success") {
        this.setState("failed");
        this.#bus.publish({ type: "error", message: `The tutor stopped: ${message.subtype}.`, retryable: true });
        return;
      }
      if (this.#state !== "awaiting-choice") this.setState("idle");
    }
  }
}

/** Wraps one MCP tool definition so every call reports its own start/end through the adapter, regardless of which tool it is. */
function instrumented(definition: SdkMcpToolDefinition<any>, adapter: ClaudeTutorialAdapter): SdkMcpToolDefinition<any> {
  return {
    ...definition,
    handler: async (args: unknown, extra: unknown) => {
      const id = crypto.randomUUID();
      adapter.onToolStart(id, definition.name, args);
      try {
        const result = await definition.handler(args as Record<string, unknown>, extra);
        const outcomeText = (result.content ?? []).filter((item): item is { type: "text"; text: string } => item.type === "text").map((item) => item.text ?? "").join("");
        adapter.onToolEnd(id, definition.name, { text: outcomeText, isError: Boolean(result.isError) });
        return result;
      } catch (error) {
        const message = error instanceof Error ? error.message : "Tool failed.";
        adapter.onToolEnd(id, definition.name, { text: message, isError: true });
        throw error;
      }
    }
  };
}
