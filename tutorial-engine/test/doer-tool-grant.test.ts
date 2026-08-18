import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { buildDoerBoundary, buildDoerOptions, type TutorialRoots } from "../src/agent/claude-agent-adapter.js";
import { createWorkspaceTools } from "../src/agent/workspace-boundary.js";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true }))); });

async function fixtureRoots(): Promise<TutorialRoots> {
  const tutorial = await mkdtemp(join(tmpdir(), "aml-tutor-")); roots.push(tutorial);
  const triage = await mkdtemp(join(tmpdir(), "aml-triage-")); roots.push(triage);
  return { tutorial, triage };
}

/**
 * Smoke test for ARCHITECTURE.md §4's doer tool grant: read `aml-tutor` and
 * `aml-triage`, write/edit only inside `aml-triage`, no `Bash`, no escape
 * outside the two sibling roots. This exercises the exact function that
 * assembles the doer's `Options` (`buildDoerOptions`) and the boundary it is
 * built from — it never calls `query()` or contacts a real model.
 */
describe("the doer's tool grant", () => {
  it("disables every built-in tool, so Bash is not merely unlisted but structurally unavailable", async () => {
    const fixture = await fixtureRoots();
    const boundary = await buildDoerBoundary(fixture);
    const { options } = buildDoerOptions(fixture, boundary, "do the thing");

    expect(options.tools).toEqual([]);
    expect(options.disallowedTools).toContain("Bash");
  });

  it("allowlists only the doer's own file tools, none of them shell-shaped", async () => {
    const fixture = await fixtureRoots();
    const boundary = await buildDoerBoundary(fixture);
    const { options } = buildDoerOptions(fixture, boundary, "do the thing");

    expect(options.allowedTools.length).toBeGreaterThan(0);
    for (const name of options.allowedTools) {
      expect(name.startsWith("mcp__triage__")).toBe(true);
      expect(name.toLowerCase()).not.toContain("bash");
    }
  });

  it("denies any tool call by a name outside its own allowlist, including a bare Bash", async () => {
    const fixture = await fixtureRoots();
    const boundary = await buildDoerBoundary(fixture);
    const { options } = buildDoerOptions(fixture, boundary, "do the thing");

    const denyBash = await options.canUseTool("Bash", { command: "rm -rf /" }, { signal: new AbortController().signal, toolUseID: "x", requestId: "r" } as never);
    expect(denyBash?.behavior).toBe("deny");

    const allowRead = await options.canUseTool("mcp__triage__read", { path: "README.md" }, { signal: new AbortController().signal, toolUseID: "x", requestId: "r" } as never);
    expect(allowRead?.behavior).toBe("allow");
  });

  it("can actually write inside aml-triage", async () => {
    const fixture = await fixtureRoots();
    const boundary = await buildDoerBoundary(fixture);
    const tools = createWorkspaceTools(boundary, () => {}, { write: true });
    const write = tools.find((item) => item.name === "write")!;

    const result = await write.handler({ path: "src/aml_triage/data.py", content: "def load_transactions(path):\n    ...\n" }, undefined);
    expect(result.isError).toBeFalsy();
    await expect(readFile(join(fixture.triage, "src/aml_triage/data.py"), "utf8")).resolves.toContain("load_transactions");
  });

  it("cannot write outside aml-triage, even into the aml-tutor root it can read", async () => {
    const fixture = await fixtureRoots();
    const boundary = await buildDoerBoundary(fixture);
    const tools = createWorkspaceTools(boundary, () => {}, { write: true });
    const write = tools.find((item) => item.name === "write")!;

    const result = await write.handler({ path: join(fixture.tutorial, "sneaky.md"), content: "not allowed" }, undefined);
    expect(result.isError).toBe(true);
    expect(result.content[0]?.text).toMatch(/outside/);
  });

  it("has no move, write, or edit tool at all when constructed read-only (the tutor's own shape)", async () => {
    const fixture = await fixtureRoots();
    const boundary = await buildDoerBoundary(fixture);
    const readOnlyTools = createWorkspaceTools(boundary, () => {}, { write: false });
    expect(readOnlyTools.map((item) => item.name)).not.toContain("write");
    expect(readOnlyTools.map((item) => item.name)).not.toContain("edit");
    expect(readOnlyTools.map((item) => item.name)).not.toContain("move");
  });
});
