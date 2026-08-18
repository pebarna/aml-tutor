import { mkdtemp, readdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { ENGINE_STATE_DIRECTORY, resetEngineState, TutorialSessionLog } from "../src/session-log.js";

const workspaces: string[] = [];

afterEach(async () => {
  await Promise.all(workspaces.splice(0).map(async (workspace) => {
    const { rm } = await import("node:fs/promises");
    await rm(workspace, { recursive: true, force: true });
  }));
});

async function workspace(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "tutorial-session-"));
  workspaces.push(root);
  return root;
}

describe("TutorialSessionLog", () => {
  it("persists protocol events and loads them in order", async () => {
    const root = await workspace();
    const log = new TutorialSessionLog(root);

    log.append({ type: "user-message", markdown: "I wrote pyproject.toml." });
    log.append({ type: "assistant-message", messageId: "assistant-1", markdown: "Great. Next, add pytest." });
    await log.flush();

    expect(await log.exists()).toBe(true);
    await expect(log.read()).resolves.toEqual([
      { type: "user-message", markdown: "I wrote pyproject.toml." },
      { type: "assistant-message", messageId: "assistant-1", markdown: "Great. Next, add pytest." }
    ]);
  });

  it("keeps its state under the engine's own hidden directory, never inside ../aml-triage", async () => {
    const root = await workspace();
    const log = new TutorialSessionLog(root);

    expect(log.path.startsWith(join(root, ENGINE_STATE_DIRECTORY))).toBe(true);
  });

  it("starts over by removing only the engine's own bookkeeping", async () => {
    const root = await workspace();
    const log = new TutorialSessionLog(root);
    log.append({ type: "user-message", markdown: "temporary work" });
    await log.flush();

    await resetEngineState(root);

    // Nothing is left of the engine's own state, and nothing outside it (in
    // particular, no sibling repository) was ever touched by this call.
    await expect(readdir(root)).resolves.toEqual([]);
    expect(await log.exists()).toBe(false);
  });

  it("is harmless when there is nothing to reset yet", async () => {
    const root = await workspace();
    await expect(resetEngineState(root)).resolves.toBeUndefined();
  });
});
