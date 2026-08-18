import { readFile, realpath } from "node:fs/promises";
import { basename, resolve } from "node:path";
import type { LessonDefinition, ValidationCommand } from "./contract.js";
import { LessonProgressStore, type LessonProgress } from "./progress-store.js";

export type ProgressState = "done" | "current" | "upcoming";
export interface ProgressItem { id: string; label: string; state: ProgressState; spec?: string; }

export interface LoadedLesson {
  definition: LessonDefinition;
  workspace: string;
  progress: ProgressItem[];
}

function titleFrom(readme: string, workspace: string): string {
  return readme.match(/^#\s+(.+)$/m)?.[1]?.trim() || basename(workspace);
}

/**
 * A row is a lesson when its first cell links to a specification, which no
 * header or separator row does. The ledger carries no per-learner status: how
 * far one learner has got is held in the engine's own state directory, not in
 * the curriculum. This tutorial's ledger is a single flat list — there is no
 * "Part 1 / Part 2" split, so a row is recognised the same way whether or not
 * any other prose precedes it.
 */
const LESSON_LINK = /^\[([^\]]+)\]\(([^)]+)\)$/;

export function ledgerPath(workspace: string): string {
  return resolve(workspace, "docs/specs/README.md");
}

function lessonRowCells(line: string): string[] | undefined {
  if (!line.trimStart().startsWith("|")) return undefined;
  const cells = line.split("|").slice(1, -1).map((cell) => cell.trim());
  return LESSON_LINK.test(cells[0] ?? "") ? cells : undefined;
}

interface LedgerEntry { id: string; label: string; spec?: string; }

function ledgerEntries(ledger: string): LedgerEntry[] {
  const entries: LedgerEntry[] = [];
  for (const line of ledger.split(/\r?\n/)) {
    const cells = lessonRowCells(line);
    if (!cells) continue;
    const link = cells[0]?.match(LESSON_LINK);
    const id = link?.[1] ?? "";
    if (!id) continue;
    entries.push({ id, label: cells[1] ?? "", spec: link?.[2] });
  }
  return entries;
}

/**
 * Turn the ledger's lesson rows into the outline. `completed` marks lessons the
 * learner finished; the first lesson not in that set is the one they are on. A
 * gap — lesson 004 finished but 003 somehow not — leaves 003 current rather
 * than skipping past it.
 */
export function readProgress(ledger: string, progress: Partial<LessonProgress> = {}): ProgressItem[] {
  const completed = progress.completed ?? new Set<string>();
  let foundCurrent = false;
  return [
    { id: "orientation", label: "Orientation", state: "done" as const },
    ...ledgerEntries(ledger).map((entry) => {
      const item = { id: entry.id, label: entry.label, spec: entry.spec };
      if (completed.has(entry.id)) return { ...item, state: "done" as const };
      if (!foundCurrent) { foundCurrent = true; return { ...item, state: "current" as const }; }
      return { ...item, state: "upcoming" as const };
    })
  ];
}

/** The lesson the learner is on, which is what the tutor is told to open. */
export function currentLesson(progress: readonly ProgressItem[]): ProgressItem | undefined {
  return progress.find((item) => item.state === "current");
}

/** Where the current lesson's specification lives, relative to the workspace. */
export function currentSpecPath(progress: readonly ProgressItem[]): string | undefined {
  const spec = currentLesson(progress)?.spec;
  return spec ? `docs/specs/${spec}` : undefined;
}

/** The outline as it stands on disk: curriculum from the ledger, state from the engine's own directory. */
export async function loadProgress(workspace: string): Promise<ProgressItem[]> {
  const ledger = await readFile(ledgerPath(workspace), "utf8");
  return readProgress(ledger, await new LessonProgressStore(workspace).read());
}

/**
 * Matches a fenced code block whose info string is `json validation`,
 * capturing its contents. A spec embeds one of these when its lesson has a
 * baked-in check to run; a lesson that creates no files can omit it.
 */
const VALIDATION_BLOCK = /```json validation\r?\n([\s\S]*?)\r?\n```/g;

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Parse every `json validation` block in one spec's markdown into `ValidationCommand`s. */
function parseValidationBlocks(markdown: string, specFile: string): ValidationCommand[] {
  const commands: ValidationCommand[] = [];
  for (const match of markdown.matchAll(VALIDATION_BLOCK)) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(match[1] ?? "");
    } catch (error) {
      throw new Error(`${specFile} has an invalid \`json validation\` block: ${error instanceof Error ? error.message : "invalid JSON"}.`);
    }
    if (!Array.isArray(parsed)) throw new Error(`${specFile}'s \`json validation\` block must be a JSON array of validation commands.`);
    for (const entry of parsed) {
      if (!isPlainObject(entry) || typeof entry.id !== "string" || typeof entry.label !== "string" || typeof entry.command !== "string") {
        throw new Error(`${specFile} has a validation command missing a string id, label, or command.`);
      }
      commands.push(entry as unknown as ValidationCommand);
    }
  }
  return commands;
}

/**
 * Every lesson's baked-in validation command, aggregated across the whole
 * ledger in ledger order.
 *
 * `loadLesson` used to leave `validationCommands` permanently empty: nothing
 * in the forked engine ever populated it, because the tutorial it came from
 * only ran checks conversationally. This lesson's "Checks" are a real,
 * deterministic pytest command (ARCHITECTURE.md §1/§5), so each spec
 * (`docs/specs/NNN-slug.md`) may embed a fenced
 * ` ```json validation … ``` ` block holding a JSON array shaped like
 * `ValidationCommand`. A lesson that creates no files can omit the block.
 *
 * IDs must be unique across the *entire* ledger, not just within one spec:
 * `run_validation` and the browser's validation buttons address a command by
 * id alone, with no per-lesson scoping.
 */
export async function loadValidationCommands(workspace: string, entries: readonly { spec?: string }[]): Promise<ValidationCommand[]> {
  const commands: ValidationCommand[] = [];
  const specById = new Map<string, string>();
  for (const entry of entries) {
    if (!entry.spec) continue;
    const specFile = `docs/specs/${entry.spec}`;
    let markdown: string;
    try {
      markdown = await readFile(resolve(workspace, specFile), "utf8");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") continue; // Not authored yet.
      throw error;
    }
    for (const command of parseValidationBlocks(markdown, specFile)) {
      const existingSpec = specById.get(command.id);
      if (existingSpec) throw new Error(`Validation command id '${command.id}' is defined in both ${existingSpec} and ${specFile}; ids must be unique across the whole ledger.`);
      specById.set(command.id, specFile);
      commands.push(command);
    }
  }
  return commands;
}

/** Infer one tutorial from its README, lesson ledger, and every lesson's baked-in checks. */
export async function loadLesson(directory: string): Promise<LoadedLesson> {
  const workspace = await realpath(resolve(directory));
  const readme = await readFile(resolve(workspace, "README.md"), "utf8");
  const ledger = await readFile(ledgerPath(workspace), "utf8");
  const entries = ledgerEntries(ledger);
  const validationCommands = await loadValidationCommands(workspace, entries);
  const definition: LessonDefinition = {
    title: titleFrom(readme, workspace),
    workspace,
    validationCommands
  };
  const progress = await new LessonProgressStore(workspace).read();
  return { definition, workspace, progress: readProgress(ledger, progress) };
}

/**
 * Record the lesson the learner is on as finished.
 *
 * Returns the progress after the write, or `undefined` when every lesson is
 * already finished and there is nothing left to advance.
 */
export async function markCurrentLessonDone(workspace: string): Promise<{ progress: ProgressItem[]; id: string } | undefined> {
  const store = new LessonProgressStore(workspace);
  const ledger = await readFile(ledgerPath(workspace), "utf8");
  const current = currentLesson(readProgress(ledger, await store.read()));
  if (!current) return undefined;
  const progress = await store.add(current.id);
  return { progress: readProgress(ledger, progress), id: current.id };
}
