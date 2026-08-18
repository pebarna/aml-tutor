import { mkdtemp, mkdir, realpath } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { ValidationRunner } from "../src/validation/runner.js";

describe("ValidationRunner", () => {
  it("runs an allowlisted executable without a shell", async () => {
    const runner = new ValidationRunner([{ id: "node", label: "Node", command: process.execPath, args: ["-e", "process.stdout.write('green')"] }], process.cwd());
    const result = await runner.run("node");
    expect(result.passed).toBe(true);
    expect(result.output).toBe("green");
  });

  it("preserves HOME for executable shims", async () => {
    const runner = new ValidationRunner([{ id: "home", label: "Home", command: process.execPath, args: ["-e", "if (!process.env.HOME) process.exit(1)"] }], process.cwd());
    await expect(runner.run("home")).resolves.toMatchObject({ passed: true });
  });

  it("rejects commands not in the lesson allowlist", async () => {
    const runner = new ValidationRunner([], process.cwd());
    await expect(runner.run("anything")).rejects.toThrow("not allowed");
  });

  it("defaults to the tutorial root when a command names no cwd", async () => {
    const runner = new ValidationRunner([{ id: "here", label: "Here", command: process.execPath, args: ["-p", "process.cwd()"] }], process.cwd());
    const result = await runner.run("here");
    expect(result.output.trim()).toBe(process.cwd());
  });

  it("resolves a relative cwd against the tutorial root, not the shell's process cwd", async () => {
    const tutorialRoot = await mkdtemp(join(tmpdir(), "validation-runner-"));
    const triage = resolve(tutorialRoot, "..", "aml-triage-fixture");
    await mkdir(triage, { recursive: true });
    const runner = new ValidationRunner(
      [{ id: "triage", label: "Triage", command: process.execPath, args: ["-p", "process.cwd()"], cwd: "../aml-triage-fixture" }],
      tutorialRoot
    );
    const result = await runner.run("triage");
    // A spawned process's own reported cwd can come back through a different
    // filesystem alias than the lexical path used to launch it (e.g. macOS's
    // /var vs. /private/var) — realpath both sides before comparing.
    expect(await realpath(result.output.trim())).toBe(await realpath(triage));
  });

  it("resolves an absolute cwd as-is", async () => {
    const tutorialRoot = await mkdtemp(join(tmpdir(), "validation-runner-abs-"));
    const runner = new ValidationRunner(
      [{ id: "abs", label: "Abs", command: process.execPath, args: ["-p", "process.cwd()"], cwd: tmpdir() }],
      tutorialRoot
    );
    const result = await runner.run("abs");
    expect(result.output.trim().length).toBeGreaterThan(0);
  });

  it("reports where a command will actually run via cwdFor, without spawning it", () => {
    const tutorialRoot = "/tutorials/aml-tutor";
    const runner = new ValidationRunner(
      [
        { id: "in-place", label: "In place", command: "true" },
        { id: "cross-repo", label: "Cross repo", command: "true", cwd: "../aml-triage" }
      ],
      tutorialRoot
    );
    expect(runner.cwdFor({ id: "in-place", label: "In place", command: "true" })).toBe(tutorialRoot);
    expect(runner.cwdFor({ id: "cross-repo", label: "Cross repo", command: "true", cwd: "../aml-triage" })).toBe("/tutorials/aml-triage");
  });
});
