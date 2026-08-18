export interface ValidationCommand {
  /** Stable identifier referenced by the browser protocol. Unique across every lesson's spec. */
  id: string;
  /** Human-readable label shown in the transcript. */
  label: string;
  /** Executable and arguments; never interpreted by a shell. */
  command: string;
  args?: string[];
  /**
   * Where the command runs, relative to the discovered tutorial root (the
   * directory containing README.md and docs/specs/) — never the shell
   * process's cwd. Omit it to run in the tutorial root itself; lesson 001's
   * command sets it to "../aml-triage" so `import aml_triage` resolves to the
   * student's package while the test file path stays inside this repository.
   */
  cwd?: string;
  /** Optional timeout, capped by the engine at two minutes. */
  timeoutMs?: number;
}

/** Internal tutorial information inferred from the tutorial directory. */
export interface LessonDefinition {
  title: string;
  workspace: string;
  validationCommands: ValidationCommand[];
}

export interface MarkdownPresentation {
  kind: "markdown";
  title: string;
  markdown: string;
}

export interface DiagramPresentation {
  kind: "diagram";
  title: string;
  mermaid: string;
  /** Text equivalent required for learners who cannot use the diagram. */
  text: string;
}

export type InitialPresentation = MarkdownPresentation | DiagramPresentation;
