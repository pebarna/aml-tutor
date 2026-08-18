import { readFile } from "node:fs/promises";
import { tool, type SdkMcpToolDefinition } from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";
import type { LessonDefinition } from "../lesson/contract.js";
import { markCurrentLessonDone } from "../lesson/load.js";
import { choiceIconCategories, type ChoiceOption, type TutorialEvent } from "../protocol/events.js";
import type { ValidationRunner } from "../validation/runner.js";
import { ChoiceManager } from "./choice-manager.js";
import { WorkspaceBoundary } from "./workspace-boundary.js";

const bounded = (value: string, max: number, field: string): string => {
  if (!value.trim() || value.length > max) throw new Error(`${field} must be between 1 and ${max} characters.`);
  return value;
};

export const choiceOptionSchema = z.object({
  id: z.string().min(1).max(80),
  label: z.string().min(1).max(160),
  icon: z.enum(choiceIconCategories),
  description: z.string().max(500).optional()
});

export interface TutorialToolDependencies {
  lesson: LessonDefinition;
  workspace: string;
  choices: ChoiceManager;
  validation: ValidationRunner;
  boundary: WorkspaceBoundary;
  emit(event: TutorialEvent): void;
  setRunState(state: "working" | "awaiting-choice"): void;
}

const text = (value: string): { content: Array<{ type: "text"; text: string }> } => ({ content: [{ type: "text", text: value }] });

/**
 * The engine's own tutorial-specific tools — tutoring semantics, not file
 * semantics, so unlike `workspace-boundary.ts`'s tools these have no
 * calculator-tutorial equivalent to fork loosely; they are the same six tools
 * the original engine built with Pi's `defineTool`, ported to the Claude
 * Agent SDK's `tool()` (zod schemas instead of typebox) with the same names,
 * parameters, and behaviour. Only the tutor session is given these: a one-shot
 * doer has no browser transcript to present into and no lesson outline to
 * advance.
 */
export function createTutorialTools(deps: TutorialToolDependencies): SdkMcpToolDefinition<any>[] {
  const presentMarkdown = tool(
    "present_markdown",
    "Present a concise titled tutorial explanation or instruction in the browser transcript.",
    { title: z.string().min(1).max(160), markdown: z.string().min(1).max(12_000) },
    async (params) => {
      deps.emit({ type: "presentation", presentation: { kind: "markdown", title: bounded(params.title, 160, "title"), markdown: bounded(params.markdown, 12_000, "markdown") } });
      return text(`Presented: ${params.title}`);
    }
  );

  const presentDiagram = tool(
    "present_diagram",
    "Present a Mermaid diagram with an accessible text fallback in the browser transcript.",
    {
      title: z.string().min(1).max(160),
      mermaid: z.string().min(1).max(12_000),
      text: z.string().min(1).max(4_000)
    },
    async (params) => {
      deps.emit({ type: "presentation", presentation: { kind: "diagram", title: bounded(params.title, 160, "title"), mermaid: bounded(params.mermaid, 12_000, "mermaid"), text: bounded(params.text, 4_000, "text") } });
      return text(`Presented diagram: ${params.title}`);
    }
  );

  const offerChoices = tool(
    "offer_choices",
    "Ask one fixed, short choice question that gives the learner control over the next tutorial step. Use it before making a change and after each guided step. Waits for a browser selection.",
    {
      question: z.string().min(1).max(1_000),
      options: z.array(choiceOptionSchema).min(2).max(6)
    },
    async (params, extra) => {
      const toolCallId = typeof (extra as { toolUseID?: unknown } | undefined)?.toolUseID === "string" ? (extra as { toolUseID: string }).toolUseID : crypto.randomUUID();
      const options = params.options as ChoiceOption[];
      if (new Set(options.map((option) => option.id)).size !== options.length) throw new Error("Choice option IDs must be unique.");
      const question = bounded(params.question, 1_000, "question");
      deps.emit({ type: "choice", id: toolCallId, question, options });
      deps.setRunState("awaiting-choice");
      const signal = (extra as { signal?: AbortSignal } | undefined)?.signal;
      const selected = await deps.choices.wait(toolCallId, options, signal);
      deps.emit({ type: "choice-resolved", id: toolCallId, optionId: selected });
      deps.setRunState("working");
      return text(`Learner selected: ${selected}`);
    }
  );

  const runValidation = tool(
    "run_validation",
    `Run one allowlisted validation command for this lesson. Allowed command IDs: ${deps.validation.commandIds.join(", ")}.`,
    { commandId: z.string().min(1).max(100) },
    async (params) => {
      const commandId = bounded(params.commandId, 100, "commandId");
      const command = deps.lesson.validationCommands.find((item) => item.id === commandId);
      if (!command) throw new Error(`Validation command '${commandId}' is not allowed.`);
      const result = await deps.validation.run(commandId, () => {});
      deps.emit({ type: "validation", id: commandId, label: command.label, ...result });
      return text(result.passed ? "Validation passed." : "Validation failed.");
    }
  );

  const showFileExcerpt = tool(
    "show_file_excerpt",
    "Show a small relevant file excerpt from aml-tutor or aml-triage. Do not use it for whole files.",
    {
      title: z.string().min(1).max(160),
      path: z.string().min(1).max(500),
      startLine: z.number().int().min(1).max(100_000),
      endLine: z.number().int().min(1).max(100_000)
    },
    async (params) => {
      if (params.endLine < params.startLine || params.endLine - params.startLine > 200) throw new Error("Excerpt must contain at most 201 lines.");
      const id = crypto.randomUUID();
      let auditPath = params.path.replaceAll("\\", "/");
      try {
        const safePath = await deps.boundary.resolve(params.path);
        auditPath = safePath.relative;
        const lines = (await readFile(safePath.absolute, "utf8")).split(/\r?\n/);
        const selected = lines.slice(params.startLine - 1, params.endLine);
        const content = selected.join("\n");
        deps.emit({ type: "audit", id, tool: "show_file_excerpt", paths: [auditPath], mutation: false, outcome: "ok" });
        deps.emit({ type: "file-excerpt", title: bounded(params.title, 160, "title"), path: auditPath, startLine: params.startLine, content, truncated: params.endLine < lines.length });
        return text(`Displayed ${auditPath}:${params.startLine}-${params.endLine}`);
      } catch (error) {
        const message = error instanceof Error ? error.message : "Workspace path rejected.";
        deps.emit({ type: "audit", id, tool: "show_file_excerpt", paths: [auditPath], mutation: false, outcome: "rejected", message });
        return { content: [{ type: "text" as const, text: message }], isError: true };
      }
    }
  );

  const completeLesson = tool(
    "complete_lesson",
    "Record the current lesson as finished, which advances the outline in the page header. Use it once, after the closing recap and before offering the choice to continue. It takes no arguments: the lesson being finished is the one the learner is on.",
    {},
    async () => {
      const id = crypto.randomUUID();
      const completed = await markCurrentLessonDone(deps.workspace);
      deps.emit({ type: "audit", id, tool: "complete_lesson", paths: [".tutorial-state/tutorial-progress.json"], mutation: completed !== undefined, outcome: "ok" });
      if (completed) deps.emit({ type: "progress", progress: completed.progress });
      return text(completed ? `Recorded lesson ${completed.id} as finished.` : "Every lesson is already finished; the outline is unchanged.");
    }
  );

  return [presentMarkdown, presentDiagram, offerChoices, runValidation, showFileExcerpt, completeLesson];
}
