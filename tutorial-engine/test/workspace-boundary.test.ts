import { mkdtemp, mkdir, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { createWorkspaceTools, WorkspaceBoundary } from "../src/agent/workspace-boundary.js";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true }))); });

async function fixture() {
  const tutor = await mkdtemp(join(tmpdir(), "aml-tutor-")); roots.push(tutor);
  const triage = await mkdtemp(join(tmpdir(), "aml-triage-")); roots.push(triage);
  await mkdir(join(tutor, "docs/specs"), { recursive: true });
  await writeFile(join(tutor, "README.md"), "tutor readme");
  await writeFile(join(tutor, "docs/specs/001.md"), "spec one");
  await mkdir(join(triage, "src"), { recursive: true });
  await writeFile(join(triage, "README.md"), "triage readme");
  return { tutor, triage };
}

const readOnlyBoundary = ({ tutor, triage }: { tutor: string; triage: string }) =>
  WorkspaceBoundary.create({ primary: tutor, readRoots: [tutor, triage], writeRoots: [] });

const doerBoundary = ({ tutor, triage }: { tutor: string; triage: string }) =>
  WorkspaceBoundary.create({ primary: triage, readRoots: [tutor, triage], writeRoots: [triage] });

describe("WorkspaceBoundary.create", () => {
  it("requires the primary root to also be a read root", async () => {
    const { tutor, triage } = await fixture();
    await expect(WorkspaceBoundary.create({ primary: tutor, readRoots: [triage], writeRoots: [] })).rejects.toThrow("primary root");
  });

  it("requires every write root to also be a read root", async () => {
    const { tutor, triage } = await fixture();
    await expect(WorkspaceBoundary.create({ primary: tutor, readRoots: [tutor], writeRoots: [triage] })).rejects.toThrow("read roots");
  });
});

describe("a read-only two-root boundary (the tutor's shape)", () => {
  it("reads workspace-relative paths against the primary root", async () => {
    const paths = await fixture();
    const boundary = await readOnlyBoundary(paths);
    await expect(boundary.readFile("README.md")).resolves.toEqual(Buffer.from("tutor readme"));
    await expect(boundary.resolve("docs/specs/001.md")).resolves.toMatchObject({ relative: "docs/specs/001.md" });
  });

  it("also reads the second allowed root, labelled so it can never be confused with the primary", async () => {
    const paths = await fixture();
    const boundary = await readOnlyBoundary(paths);
    await expect(boundary.readFile(join(paths.triage, "README.md"))).resolves.toEqual(Buffer.from("triage readme"));
    const resolved = await boundary.resolve(join(paths.triage, "README.md"));
    expect(resolved.relative.startsWith(`${basename(paths.triage)}/`)).toBe(true);
  });

  it("rejects parent traversal and an escaping symlink", async () => {
    const paths = await fixture();
    const outside = await mkdtemp(join(tmpdir(), "outside-")); roots.push(outside);
    await writeFile(join(outside, "secret.txt"), "secret");
    await symlink(outside, join(paths.tutor, "escape"));
    const boundary = await readOnlyBoundary(paths);
    await expect(boundary.readFile("../secret.txt")).rejects.toThrow("outside");
    await expect(boundary.readFile("escape/secret.txt")).rejects.toThrow("outside");
  });

  it("rejects a third, unrelated directory even if it exists on disk", async () => {
    const paths = await fixture();
    const third = await mkdtemp(join(tmpdir(), "third-repo-"));
    roots.push(third);
    await writeFile(join(third, "file.txt"), "not part of this tutorial");
    const boundary = await readOnlyBoundary(paths);
    await expect(boundary.readFile(join(third, "file.txt"))).rejects.toThrow("outside");
  });

  it("refuses every mutation, because it has no write roots", async () => {
    const paths = await fixture();
    const boundary = await readOnlyBoundary(paths);
    await expect(boundary.writeFile("notes.md", "hello")).rejects.toThrow("outside");
    await expect(boundary.writeFile(join(paths.triage, "notes.md"), "hello")).rejects.toThrow("outside");
  });

  it("omits write, edit, and move from the tool set entirely", async () => {
    const paths = await fixture();
    const boundary = await readOnlyBoundary(paths);
    const tools = createWorkspaceTools(boundary, () => {}, { write: false });
    expect(tools.map((item) => item.name).sort()).toEqual(["find", "grep", "ls", "read"]);
  });
});

describe("a read-everywhere, write-in-aml-triage boundary (the doer's shape)", () => {
  it("reads across both roots but writes only inside aml-triage", async () => {
    const paths = await fixture();
    const boundary = await doerBoundary(paths);
    await expect(boundary.readFile(join(paths.tutor, "docs/specs/001.md"))).resolves.toEqual(Buffer.from("spec one"));
    await boundary.writeFile("src/main.py", "print('hi')\n");
    await expect(boundary.readFile("src/main.py")).resolves.toEqual(Buffer.from("print('hi')\n"));
  });

  it("refuses to write into aml-tutor even though it can read there", async () => {
    const paths = await fixture();
    const boundary = await doerBoundary(paths);
    await expect(boundary.writeFile(join(paths.tutor, "docs/specs/002.md"), "sneaky")).rejects.toThrow("outside");
  });

  it("moves a file within aml-triage and audits both ends as a mutation", async () => {
    const paths = await fixture();
    const boundary = await doerBoundary(paths);
    await boundary.writeFile("src/old.py", "content");
    const audits: Array<{ tool: string; paths: string[]; mutation: boolean; outcome: string }> = [];
    const tools = createWorkspaceTools(boundary, (event) => audits.push(event), { write: true });
    const move = tools.find((item) => item.name === "move")!;

    await expect(move.handler({ path: "src/old.py", destination: "src/new.py" }, undefined))
      .resolves.toMatchObject({ content: [{ type: "text", text: "Moved src/old.py to src/new.py" }] });
    await expect(boundary.exists("src/old.py")).resolves.toBe(false);
    await expect(boundary.readFile("src/new.py")).resolves.toEqual(Buffer.from("content"));
    expect(audits).toEqual([{ type: "audit", id: expect.any(String), tool: "move", paths: ["src/old.py", "src/new.py"], mutation: true, outcome: "ok" }]);
  });

  it("refuses a move whose source is only readable, not writable", async () => {
    const paths = await fixture();
    const boundary = await doerBoundary(paths);
    const tools = createWorkspaceTools(boundary, () => {}, { write: true });
    const move = tools.find((item) => item.name === "move")!;

    const result = await move.handler({ path: join(paths.tutor, "docs/specs/001.md"), destination: "src/moved.md" }, undefined);
    expect(result).toMatchObject({ isError: true, content: [{ type: "text", text: expect.stringContaining("outside") }] });
  });

  it("includes write, edit, and move alongside the read-only tools", async () => {
    const paths = await fixture();
    const boundary = await doerBoundary(paths);
    const tools = createWorkspaceTools(boundary, () => {}, { write: true });
    expect(tools.map((item) => item.name).sort()).toEqual(["edit", "find", "grep", "ls", "move", "read", "write"]);
  });
});
