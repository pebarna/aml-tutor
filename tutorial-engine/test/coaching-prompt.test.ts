import { describe, expect, it } from "vitest";
import { coachingSystemPrompt } from "../src/agent/claude-agent-adapter.js";

const lesson = {
  title: "Example lesson",
  workspace: "/tmp/example",
  validationCommands: []
};

describe("coachingSystemPrompt", () => {
  it("guides learner-led work in the current specification's stated order", () => {
    const prompt = coachingSystemPrompt(lesson);

    expect(prompt).toContain("implementation order stated by the current specification");
    expect(prompt).toContain("short conceptual outline");
    expect(prompt).toContain("small code snippet");
  });

  it("names the current specification, so the tutor never resolves progress itself", () => {
    // Progress lives in the engine's own hidden state directory, outside the
    // ledger the tutor can read, so the engine has to say which lesson the
    // learner is on.
    const prompt = coachingSystemPrompt(lesson, "docs/specs/003-time-based-split.md");

    expect(prompt).toContain("then docs/specs/003-time-based-split.md, which is the specification for the lesson the learner is on");
    expect(prompt).not.toContain("the first specification whose ledger status is Todo");
  });

  it("falls back to the first unfinished lesson when there is no current specification", () => {
    const prompt = coachingSystemPrompt(lesson);

    expect(prompt).toContain("then the first specification the learner has not finished");
  });

  it("teaches the mechanism where a specification states a principle figuratively", () => {
    const prompt = coachingSystemPrompt(lesson);

    expect(prompt).toContain("teach the mechanism instead");
    expect(prompt).toContain("which capability was removed");
    expect(prompt).toContain("never stack two of them in one sentence");
  });

  it("names the learner's real deliverable as the sibling aml-triage repository", () => {
    const prompt = coachingSystemPrompt(lesson);

    expect(prompt).toContain("../aml-triage");
    expect(prompt).toContain("PaySim-based XGBoost fraud classifier");
  });

  it("offers progressive help after each learner-led step", () => {
    const prompt = coachingSystemPrompt(lesson);

    expect(prompt).toContain("I'll do it");
    expect(prompt).toContain("I've made this step");
    expect(prompt).toContain("Show me exactly what to type");
    expect(prompt).toContain("Make this step for me");
    expect(prompt).toContain("Every offer_choices option must supply an icon category.");
    expect(prompt).toContain('"I\'ll do it"=do');
    expect(prompt).toContain('"Make it for me"=automate');
    expect(prompt).toContain('"I\'ve made this step"=confirm');
    expect(prompt).toContain('"Show me exactly what to type"=show');
    expect(prompt).toContain("Use pause for a stop or pause choice.");
  });

  it("delegates to automate_step rather than acting as the doer itself", () => {
    const prompt = coachingSystemPrompt(lesson);

    expect(prompt).toContain("call automate_step with precise, self-contained instructions");
    expect(prompt).toContain("Do not act as the doer yourself");
    expect(prompt).toContain("has no memory of this conversation");
  });

  it("requires run_validation before claiming a check passes", () => {
    const prompt = coachingSystemPrompt(lesson);

    expect(prompt).toContain("never take the learner's word that a baked-in test passes");
    expect(prompt).toContain("run it yourself with run_validation");
  });

  it("pairs comprehension questions with the deterministic check, and asks them in the tutor's own words", () => {
    const prompt = coachingSystemPrompt(lesson);

    expect(prompt).toContain("comprehension questions");
    expect(prompt).toContain("in your own words");
  });

  it("holds the learner at the end of every lesson", () => {
    const prompt = coachingSystemPrompt(lesson);

    expect(prompt).toContain("At the end of every lesson, stop there.");
    expect(prompt).toContain("offer a choice between pausing for now and continuing to the next lesson");
  });

  it("records the lesson as finished before offering the choice, so pausing still advances the outline", () => {
    const prompt = coachingSystemPrompt(lesson);

    expect(prompt).toContain("use complete_lesson once, and then offer a choice");
    expect(prompt).toContain("record it before the choice, not after");
  });

  it("does not invent an artefact for a lesson that builds nothing", () => {
    const prompt = coachingSystemPrompt(lesson);

    expect(prompt).toContain("creates no files");
    expect(prompt).toContain("do not offer to build anything");
    expect(prompt).toContain("questions the learner answers in their own words");
  });

  it("holds the learner at the final lesson with the stronger, more specific beat", () => {
    const prompt = coachingSystemPrompt(lesson);

    expect(prompt).toContain("precision, recall, PR-AUC, and chosen operating threshold");
    expect(prompt).toContain("Phase 1 is complete and defensible");
  });
});
