import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { currentSpecPath, loadLesson, loadValidationCommands, readProgress } from "../src/lesson/load.js";

const fixture = fileURLToPath(new URL("./fixtures/sample-lesson", import.meta.url));
const realTutorialRoot = fileURLToPath(new URL("../../", import.meta.url));

const workspaces: string[] = [];
afterEach(async () => {
  const { rm } = await import("node:fs/promises");
  await Promise.all(workspaces.splice(0).map((workspace) => rm(workspace, { recursive: true, force: true })));
});
async function tempWorkspace(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "lesson-load-"));
  workspaces.push(root);
  await mkdir(join(root, "docs/specs"), { recursive: true });
  return root;
}

describe("loadLesson", () => {
  it("infers the tutorial from its README and ledger", async () => {
    const loaded = await loadLesson(fixture);
    expect(loaded.definition.title).toBe("Fixture tutorial");
    expect(loaded.workspace).toBe(fixture);
    expect(loaded.progress).toEqual([
      { id: "orientation", label: "Orientation", state: "done" },
      { id: "001", label: "Fixture step", state: "current", spec: "001.md" },
      { id: "002", label: "Second fixture step", state: "upcoming", spec: "002.md" },
    ]);
  });

  it("ignores a header row whatever its first column is called", async () => {
    const loaded = await loadLesson(fixture);
    expect(loaded.progress.some((item) => item.label === "Goal")).toBe(false);
    expect(loaded.progress.some((item) => item.id === "Lesson")).toBe(false);
  });

  it("aggregates every lesson's json validation block into the lesson's validationCommands", async () => {
    // 001.md ships a `json validation` block; 002.md does not exist on disk yet
    // (Phase C content is authored separately), which loadValidationCommands
    // must tolerate rather than fail the whole tutorial over.
    const loaded = await loadLesson(fixture);
    expect(loaded.definition.validationCommands).toEqual([
      { id: "fixture-001", label: "Fixture check", command: "true" }
    ]);
  });

  it("loads this repository's own real ledger and every lesson's baked-in check, flattening the ledger's Part 1/2/3 grouping", async () => {
    // A guard against the loader and the real lesson content drifting apart —
    // not new curriculum content, which is Phase B/C's to write.
    const loaded = await loadLesson(realTutorialRoot);
    expect(loaded.definition.title).toBe("AML triage tutorial");
    expect(loaded.progress[0]).toEqual({ id: "orientation", label: "Orientation", state: "done" });
    expect(loaded.progress.slice(1).map((item) => item.id)).toEqual([
      "001", "002", "003", "004", "005", "006", "007",
      "008", "009", "010", "011", "012", "013", "014", "015"
    ]);
    expect(loaded.definition.validationCommands).toEqual([
      { id: "001-project-setup", label: "Project setup", command: "uv", args: ["run", "pytest", "../aml-tutor/tests/001_test_project_setup.py"], cwd: "../aml-triage" },
      { id: "002-load-and-explore-the-data", label: "Load and explore the data", command: "uv", args: ["run", "pytest", "../aml-tutor/tests/002_test_data_loading.py"], cwd: "../aml-triage" },
      { id: "003-time-based-split", label: "Time-based split", command: "uv", args: ["run", "pytest", "../aml-tutor/tests/003_test_time_based_split.py"], cwd: "../aml-triage" },
      { id: "004-feature-engineering", label: "Feature engineering", command: "uv", args: ["run", "pytest", "../aml-tutor/tests/004_test_feature_engineering.py"], cwd: "../aml-triage" },
      { id: "005-class-imbalance", label: "Class imbalance", command: "uv", args: ["run", "pytest", "../aml-tutor/tests/005_test_class_imbalance.py"], cwd: "../aml-triage" },
      { id: "006-train-the-baseline", label: "Train the baseline", command: "uv", args: ["run", "pytest", "../aml-tutor/tests/006_test_train_baseline.py"], cwd: "../aml-triage" },
      { id: "007-evaluation-and-threshold", label: "Evaluation and threshold", command: "uv", args: ["run", "pytest", "../aml-tutor/tests/007_test_evaluation.py"], cwd: "../aml-triage" },
      { id: "008-typology-retrieval", label: "Typology retrieval", command: "uv", args: ["run", "pytest", "../aml-tutor/tests/008_test_retrieval.py"], cwd: "../aml-triage" },
      { id: "009-hybrid-retrieval", label: "Hybrid retrieval", command: "uv", args: ["run", "pytest", "../aml-tutor/tests/009_test_hybrid_retrieval.py"], cwd: "../aml-triage" },
      { id: "010-structured-triage-decisions", label: "Structured triage decisions", command: "uv", args: ["run", "pytest", "../aml-tutor/tests/010_test_structured_decisions.py", "-v"], cwd: "../aml-triage" },
      { id: "011-the-triage-agent", label: "The end-to-end triage agent", command: "uv", args: ["run", "pytest", "../aml-tutor/tests/011_test_triage_agent.py", "-v"], cwd: "../aml-triage" },
      { id: "012-the-hand-labeled-eval-set", label: "The hand-labeled triage eval set", command: "uv", args: ["run", "pytest", "../aml-tutor/tests/012_test_eval_set.py", "-v"], cwd: "../aml-triage" },
      { id: "013-deterministic-triage-checks", label: "Deterministic triage checks", command: "uv", args: ["run", "pytest", "../aml-tutor/tests/013_test_deterministic_checks.py", "-v"], cwd: "../aml-triage" },
      { id: "014-llm-as-judge", label: "LLM-as-judge scoring", command: "uv", args: ["run", "pytest", "../aml-tutor/tests/014_test_llm_judge.py", "-v"], cwd: "../aml-triage" },
      { id: "015-the-agreement-rate-report", label: "The agreement-rate report", command: "uv", args: ["run", "pytest", "../aml-tutor/tests/015_test_agreement_report.py", "-v"], cwd: "../aml-triage" }
    ]);
  });
});

describe("loadValidationCommands", () => {
  it("returns nothing for a lesson whose spec has no json validation block", async () => {
    const workspace = await tempWorkspace();
    await writeFile(join(workspace, "docs/specs/001-no-checks.md"), "# 001\n\nNothing to run yet.\n", "utf8");
    await expect(loadValidationCommands(workspace, [{ spec: "001-no-checks.md" }])).resolves.toEqual([]);
  });

  it("skips a ledger entry whose spec file does not exist yet", async () => {
    const workspace = await tempWorkspace();
    await expect(loadValidationCommands(workspace, [{ spec: "999-not-written.md" }])).resolves.toEqual([]);
  });

  it("skips a ledger entry with no spec at all", async () => {
    const workspace = await tempWorkspace();
    await expect(loadValidationCommands(workspace, [{}])).resolves.toEqual([]);
  });

  it("parses a cwd alongside the command, so a check can run inside ../aml-triage", async () => {
    const workspace = await tempWorkspace();
    await writeFile(join(workspace, "docs/specs/001-project-setup.md"), [
      "# 001 — Project setup",
      "",
      "```json validation",
      JSON.stringify([{ id: "001-tests", label: "Project setup tests", command: "uv", args: ["run", "pytest", "../aml-tutor/tests/001_test_project_setup.py"], cwd: "../aml-triage" }]),
      "```",
      ""
    ].join("\n"), "utf8");

    await expect(loadValidationCommands(workspace, [{ spec: "001-project-setup.md" }])).resolves.toEqual([
      { id: "001-tests", label: "Project setup tests", command: "uv", args: ["run", "pytest", "../aml-tutor/tests/001_test_project_setup.py"], cwd: "../aml-triage" }
    ]);
  });

  it("rejects a duplicate command id across two different specs", async () => {
    const workspace = await tempWorkspace();
    await writeFile(join(workspace, "docs/specs/001-first.md"), "```json validation\n[{\"id\": \"shared\", \"label\": \"First\", \"command\": \"true\"}]\n```\n", "utf8");
    await writeFile(join(workspace, "docs/specs/002-second.md"), "```json validation\n[{\"id\": \"shared\", \"label\": \"Second\", \"command\": \"true\"}]\n```\n", "utf8");

    await expect(loadValidationCommands(workspace, [{ spec: "001-first.md" }, { spec: "002-second.md" }]))
      .rejects.toThrow(/shared.*001-first\.md.*002-second\.md/s);
  });

  it("rejects a json validation block that is not a JSON array", async () => {
    const workspace = await tempWorkspace();
    await writeFile(join(workspace, "docs/specs/001-bad.md"), "```json validation\n{\"id\": \"not-an-array\"}\n```\n", "utf8");
    await expect(loadValidationCommands(workspace, [{ spec: "001-bad.md" }])).rejects.toThrow(/JSON array/);
  });

  it("rejects a validation command missing a required field", async () => {
    const workspace = await tempWorkspace();
    await writeFile(join(workspace, "docs/specs/001-incomplete.md"), "```json validation\n[{\"id\": \"no-command\", \"label\": \"Missing command\"}]\n```\n", "utf8");
    await expect(loadValidationCommands(workspace, [{ spec: "001-incomplete.md" }])).rejects.toThrow(/missing a string id, label, or command/);
  });
});

const ledger = [
  "# Lessons",
  "",
  "| Lesson | Goal |",
  "| --- | --- |",
  "| [001](001-first.md) | First step |",
  "| [002](002-second.md) | Second step |",
  ""
].join("\n");

describe("readProgress", () => {
  it("takes the outline's shape from the ledger and its state from the finished set", () => {
    const states = (completed: string[]) =>
      readProgress(ledger, { completed: new Set(completed) }).slice(1).map((item) => [item.id, item.state]);

    expect(states([])).toEqual([["001", "current"], ["002", "upcoming"]]);
    expect(states(["001"])).toEqual([["001", "done"], ["002", "current"]]);
    expect(states(["001", "002"])).toEqual([["001", "done"], ["002", "done"]]);
  });

  it("leaves an earlier unfinished lesson current rather than skipping to the gap", () => {
    // A ledger row can only be finished through the tool, but a hand-edited
    // progress file should not be able to strand the learner past a lesson
    // they have not done.
    expect(readProgress(ledger, { completed: new Set(["002"]) }).slice(1).map((item) => [item.id, item.state]))
      .toEqual([["001", "current"], ["002", "done"]]);
  });

  it("reads state only from the finished set, even if a status column reappears", () => {
    // The ledger used to carry a Status cell. Should one come back, it must not
    // be able to claim a lesson is done: the curriculum ships to everyone, and
    // only the engine's own state directory knows about this learner.
    const withStatus = ledger
      .replace("| Lesson | Goal |", "| Lesson | Goal | Status |")
      .replace("| --- | --- |", "| --- | --- | --- |")
      .replace("| [001](001-first.md) | First step |", "| [001](001-first.md) | First step | Done |");

    expect(readProgress(withStatus).slice(1).map((item) => [item.id, item.state]))
      .toEqual([["001", "current"], ["002", "upcoming"]]);
  });

  it("skips header and separator rows, which carry no specification link", () => {
    expect(readProgress(ledger).slice(1).map((item) => item.id)).toEqual(["001", "002"]);
  });

  it("carries each lesson's specification filename for routing the tutor", () => {
    expect(currentSpecPath(readProgress(ledger))).toBe("docs/specs/001-first.md");
    expect(currentSpecPath(readProgress(ledger, { completed: new Set(["001"]) }))).toBe("docs/specs/002-second.md");
    expect(currentSpecPath(readProgress(ledger, { completed: new Set(["001", "002"]) }))).toBeUndefined();
  });
});
