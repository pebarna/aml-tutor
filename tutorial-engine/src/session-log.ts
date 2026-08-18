import { appendFile, mkdir, readFile, rm } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { isTranscriptEvent, parseTutorialEvent, type TutorialEvent } from "./protocol/events.js";

const SESSION_LOG_NAME = "tutorial-session.jsonl";

/**
 * Everything the engine keeps for itself, under one directory the learner
 * never has to look in.
 *
 * Unlike the calculator tutorial this forked from, this workspace (aml-tutor)
 * is pure curriculum plus engine — there is no learner-owned `factory/`
 * scratch directory here, because lessons write the learner's real code into
 * the sibling `../aml-triage` repository instead. So the engine's own
 * bookkeeping — the session transcript and lesson progress — lives directly
 * under this hidden directory at the tutorial workspace root, not nested
 * inside a kata folder. One directory means one ignore rule covers it.
 */
export const ENGINE_STATE_DIRECTORY = ".tutorial-state";

/**
 * Append-only browser transcript storage. It deliberately records protocol
 * events, rather than the agent SDK's own session, so a stopped tool call is
 * never revived.
 */
export class TutorialSessionLog {
  readonly path: string;
  #writes: Promise<void> = Promise.resolve();

  constructor(workspace: string) {
    this.path = resolve(workspace, ENGINE_STATE_DIRECTORY, SESSION_LOG_NAME);
  }

  async exists(): Promise<boolean> {
    try {
      const contents = await readFile(this.path, "utf8");
      return contents.trim().length > 0;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
      throw error;
    }
  }

  async read(): Promise<TutorialEvent[]> {
    let contents: string;
    try {
      contents = await readFile(this.path, "utf8");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
      throw error;
    }
    const events: TutorialEvent[] = [];
    for (const [index, line] of contents.split(/\r?\n/).entries()) {
      if (!line.trim()) continue;
      try {
        const event = parseTutorialEvent(line);
        if (event.type === "snapshot") throw new Error("Snapshots are not transcript events.");
        events.push(event);
      } catch (error) {
        throw new Error(`Could not read saved tutorial session at line ${index + 1}: ${error instanceof Error ? error.message : "invalid event"}`);
      }
    }
    return events;
  }

  append(event: TutorialEvent): void {
    if (!isTranscriptEvent(event)) return;
    this.#writes = this.#writes.then(async () => {
      // Created on demand: a fresh clone has no engine state directory at all.
      await mkdir(dirname(this.path), { recursive: true });
      await appendFile(this.path, `${JSON.stringify(event)}\n`, "utf8");
    });
  }

  async clear(): Promise<void> {
    await this.flush();
    await rm(this.path, { force: true });
  }

  async flush(): Promise<void> {
    await this.#writes;
  }
}

/**
 * Reset the engine's own bookkeeping — lesson progress and the saved session
 * transcript — so a "start over" returns the learner to lesson one.
 *
 * Deliberately narrower than the calculator tutorial's `resetFactory`: that
 * one deleted the learner's kata workspace, which was disposable scratch this
 * engine owned. `../aml-triage` is the learner's real deliverable and is never
 * touched here or anywhere else in this engine — only this hidden directory,
 * which holds nothing but the engine's own progress tracking, is cleared.
 */
export async function resetEngineState(workspace: string): Promise<void> {
  await rm(resolve(workspace, ENGINE_STATE_DIRECTORY), { recursive: true, force: true });
}
