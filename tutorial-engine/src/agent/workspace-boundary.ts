import { access, lstat, mkdir, readFile, readdir, realpath, rename, stat, writeFile } from "node:fs/promises";
import { basename, dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { tool, type SdkMcpToolDefinition } from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";
import type { AuditEvent } from "../protocol/events.js";

export type AuditSink = (event: AuditEvent) => void;

/**
 * The two sibling repositories a session may touch, and what it may do to
 * each. `primary` is where a workspace-relative tool path resolves by
 * default; every root named here must also appear in `readRoots` (`primary`
 * itself is never implicitly read-only or read-write — say so explicitly).
 *
 * Per ARCHITECTURE.md §4: the tutor is constructed with `writeRoots: []`
 * (read-only across both `aml-tutor` and `aml-triage`) and the doer with
 * `writeRoots: [amlTriage]` (it may also read `aml-tutor`'s specs and
 * baked-in tests, to know what it is building, but never write there).
 */
export interface WorkspaceRoots {
  primary: string;
  readRoots: string[];
  writeRoots: string[];
}

interface ResolvedRoot { readonly path: string; readonly label: string; }

/**
 * Resolves every path through one of a small set of real, allowed
 * directories. Deliberately a filesystem boundary, not a prompt convention:
 * absolute paths, `..`, and symlinks which leave every allowed root are
 * rejected before a tool touches them.
 *
 * Generalizes the single-root boundary the calculator tutorial forked from:
 * that tutorial had one kata workspace, so relative-path resolution and the
 * escape check both had exactly one root to be relative to. This tutorial's
 * lessons live in `aml-tutor` but write the learner's real code into the
 * sibling `aml-triage`, so a path may legitimately resolve into either
 * directory — `resolve()` tries every root the requested operation (read vs.
 * write) is allowed into, in order, and rejects the path only when none of
 * them contain it.
 */
export class WorkspaceBoundary {
  readonly #primary: string;
  readonly #readRoots: readonly ResolvedRoot[];
  readonly #writeRoots: readonly ResolvedRoot[];

  private constructor(primary: string, readRoots: readonly ResolvedRoot[], writeRoots: readonly ResolvedRoot[]) {
    this.#primary = primary;
    this.#readRoots = readRoots;
    this.#writeRoots = writeRoots;
  }

  static async create(roots: WorkspaceRoots): Promise<WorkspaceBoundary> {
    const canonicalize = async (path: string): Promise<ResolvedRoot> => ({ path: await realpath(path), label: basename(path) || path });
    const primary = (await canonicalize(roots.primary)).path;
    const readRoots = await Promise.all(roots.readRoots.map(canonicalize));
    const writeRoots = await Promise.all(roots.writeRoots.map(canonicalize));
    if (!readRoots.some((root) => root.path === primary)) throw new Error("The primary root must also be listed among the read roots.");
    for (const root of writeRoots) {
      if (!readRoots.some((read) => read.path === root.path)) throw new Error(`Write root '${root.label}' must also be listed among the read roots.`);
    }
    return new WorkspaceBoundary(primary, readRoots, writeRoots);
  }

  /** The root a workspace-relative tool path resolves against by default. */
  get root(): string { return this.#primary; }

  async resolve(rawPath: string, forWrite = false): Promise<{ absolute: string; relative: string }> {
    if (!rawPath || typeof rawPath !== "string") throw new Error("A workspace path is required.");
    const lexical = isAbsolute(rawPath) ? resolve(rawPath) : resolve(this.#primary, rawPath);
    // A relative path is already anchored to `this.#primary`, which is itself
    // a realpath, so it lexically matches. An *absolute* path a caller passed
    // in (e.g. naming the second allowed root directly) is not guaranteed to
    // use the same spelling: on macOS /var and /private/var name the same
    // directory, and only the latter is what `realpath` returns. Canonicalize
    // through the nearest existing ancestor before matching roots, so an
    // absolute path spelled either way still resolves to the same root.
    const candidate = await this.canonicalize(lexical);
    const allowed = forWrite ? this.#writeRoots : this.#readRoots;
    // Reject lexical escapes before probing the filesystem for anything else.
    // Otherwise a missing external path can leak an ENOENT rather than the
    // boundary decision.
    const matched = allowed.find((root) => this.isInside(candidate, root));
    if (!matched) throw new Error("Path is outside the tutorial workspace.");
    const existing = await this.nearestExisting(candidate, forWrite);
    const realExisting = await realpath(existing);
    if (!this.isInside(realExisting, matched)) throw new Error("Path is outside the tutorial workspace.");
    // A path can contain a symlink below the nearest ancestor. Resolve it when it
    // exists; for a new write its existing parent is already checked above.
    try {
      const realCandidate = await realpath(candidate);
      if (!this.isInside(realCandidate, matched)) throw new Error("Path is outside the tutorial workspace.");
    } catch (error) {
      if (!(error instanceof Error) || !/ENOENT|no such file/i.test(error.message)) throw error;
    }
    return { absolute: candidate, relative: this.displayPath(candidate, matched) };
  }

  /** Resolve as much of `candidate` as already exists through `realpath`, without requiring the whole path to exist. */
  private async canonicalize(candidate: string): Promise<string> {
    try {
      const realParent = await realpath(dirname(candidate));
      return resolve(realParent, basename(candidate));
    } catch {
      return candidate; // Nothing above it exists yet either; leave it lexical.
    }
  }

  async readFile(path: string): Promise<Buffer> { return readFile((await this.resolve(path)).absolute); }
  async access(path: string): Promise<void> { await access((await this.resolve(path)).absolute); }
  async writeFile(path: string, content: string): Promise<void> {
    const safePath = await this.resolve(path, true);
    await mkdir(dirname(safePath.absolute), { recursive: true });
    await writeFile(safePath.absolute, content, "utf8");
  }
  async mkdir(path: string): Promise<void> { await mkdir((await this.resolve(path, true)).absolute, { recursive: true }); }
  /**
   * Relocate one file. Both ends must resolve into a write root: a move
   * deletes its source, which is exactly as much of a mutation as writing the
   * destination, so (unlike a plain read) the source cannot be outside every
   * writable root even when it is inside a readable one.
   */
  async move(path: string, destination: string): Promise<{ from: string; to: string }> {
    const source = await this.resolve(path, true);
    const target = await this.resolve(destination, true);
    if (source.absolute === target.absolute) throw new Error("The source and destination are the same path.");
    if (await this.exists(target.absolute)) throw new Error(`'${target.relative}' already exists; a move never overwrites.`);
    await mkdir(dirname(target.absolute), { recursive: true });
    await rename(source.absolute, target.absolute);
    return { from: source.relative, to: target.relative };
  }
  async isDirectory(path: string): Promise<boolean> { return (await stat((await this.resolve(path)).absolute)).isDirectory(); }
  async stat(path: string) { return stat((await this.resolve(path)).absolute); }
  async readdir(path: string): Promise<string[]> { return readdir((await this.resolve(path)).absolute); }
  async exists(path: string): Promise<boolean> {
    try { await lstat((await this.resolve(path)).absolute); return true; }
    catch { return false; }
  }

  private isInside(path: string, root: ResolvedRoot): boolean { return path === root.path || path.startsWith(root.path + sep); }

  /** Workspace-relative for the primary root; `<label>/…` for any other allowed root, so the two never collide in a log or transcript. */
  private displayPath(candidate: string, root: ResolvedRoot): string {
    const relativePath = relative(root.path, candidate).split(sep).join("/");
    if (root.path === this.#primary) return relativePath || ".";
    return relativePath ? `${root.label}/${relativePath}` : root.label;
  }

  private async nearestExisting(candidate: string, forWrite: boolean): Promise<string> {
    let current = candidate;
    for (;;) {
      try { await lstat(current); return current; }
      catch (error) {
        if (!forWrite && current === candidate) throw error;
        const parent = dirname(current);
        if (parent === current) throw new Error("Path is outside the tutorial workspace.");
        current = parent;
      }
    }
  }
}

function audited<Args>(
  name: string,
  mutation: boolean,
  boundary: WorkspaceBoundary,
  sink: AuditSink,
  pathsOf: (args: Args) => string[],
  execute: (args: Args) => Promise<{ content: Array<{ type: "text"; text: string }> }>
): (args: Args, extra: unknown) => Promise<{ content: Array<{ type: "text"; text: string }>; isError?: boolean }> {
  return async (args) => {
    const id = crypto.randomUUID();
    const rawPaths = pathsOf(args);
    let paths: string[] = [];
    try {
      paths = await Promise.all(rawPaths.map(async (path) => (await boundary.resolve(path, mutation)).relative));
    } catch (error) {
      const message = error instanceof Error ? error.message : "Workspace path rejected.";
      sink({ type: "audit", id, tool: name, paths: rawPaths.map((path) => path.replaceAll("\\", "/")), mutation, outcome: "rejected", message });
      return { content: [{ type: "text", text: message }], isError: true };
    }
    try {
      const result = await execute(args);
      sink({ type: "audit", id, tool: name, paths, mutation, outcome: "ok" });
      return result;
    } catch (error) {
      const message = error instanceof Error ? error.message : "Tool failed.";
      sink({ type: "audit", id, tool: name, paths, mutation, outcome: "error", message });
      return { content: [{ type: "text", text: message }], isError: true };
    }
  };
}

const text = (value: string): { content: Array<{ type: "text"; text: string }> } => ({ content: [{ type: "text", text: value }] });

/**
 * Custom MCP file tools, replacing the SDK's built-in Read/Write/Edit/Grep/
 * Glob/Bash for both the tutor and the doer (ARCHITECTURE.md §4). Every
 * operation is resolved through `boundary` before touching the filesystem, so
 * the boundary — not the model's good behaviour — is what actually enforces
 * the read/write split between `aml-tutor` and `aml-triage`.
 *
 * `write` controls whether the mutating tools (edit, write, move) are
 * included at all: the tutor is never offered them (its coaching prompt
 * already says not to act as the doer, and the boundary would refuse the
 * write anyway since the tutor's `writeRoots` is empty — omitting the tools
 * avoids a confusing "why can't I do this" round trip for the model).
 */
export function createWorkspaceTools(boundary: WorkspaceBoundary, sink: AuditSink, options: { write: boolean }): SdkMcpToolDefinition<any>[] {
  const read = tool(
    "read",
    "Read a text file inside the allowed workspaces. Optionally a line range.",
    {
      path: z.string().min(1).max(500).describe("The file to read, relative to the primary workspace, or an absolute path inside an allowed root."),
      offset: z.number().int().min(1).max(1_000_000).optional().describe("First line to read, 1-indexed."),
      limit: z.number().int().min(1).max(5_000).optional().describe("Maximum number of lines to read.")
    },
    audited("read", false, boundary, sink, (args) => [args.path], async (args) => {
      const content = (await boundary.readFile(args.path)).toString("utf8");
      const lines = content.split(/\r?\n/);
      const start = Math.max(1, args.offset ?? 1);
      const end = args.limit ? Math.min(lines.length, start + args.limit - 1) : lines.length;
      const numbered = lines.slice(start - 1, end).map((line, index) => `${start + index}\t${line}`).join("\n");
      return text(numbered);
    })
  );

  const grep = tool(
    "grep",
    "Search text files inside the allowed workspaces for a regular expression, and report matching file:line.",
    {
      pattern: z.string().min(1).max(500).describe("A JavaScript-flavoured regular expression."),
      path: z.string().min(1).max(500).optional().describe("Directory to search, relative to the primary workspace. Defaults to its root."),
      maxResults: z.number().int().min(1).max(500).optional()
    },
    audited("grep", false, boundary, sink, (args) => [args.path ?? "."], async (args) => {
      const regex = new RegExp(args.pattern);
      const root = args.path ?? ".";
      const limit = args.maxResults ?? 200;
      const matches: string[] = [];
      const walk = async (relativeDir: string): Promise<void> => {
        if (matches.length >= limit) return;
        const absoluteDir = (await boundary.resolve(relativeDir)).absolute;
        for (const entry of await readdir(absoluteDir, { withFileTypes: true })) {
          if (matches.length >= limit) return;
          if (entry.name.startsWith(".") || entry.name === "node_modules") continue;
          const childRelative = relativeDir === "." ? entry.name : `${relativeDir}/${entry.name}`;
          if (entry.isDirectory()) { await walk(childRelative); continue; }
          if (!entry.isFile()) continue;
          let contents: string;
          try { contents = await readFile((await boundary.resolve(childRelative)).absolute, "utf8"); }
          catch { continue; } // Binary or unreadable; skip rather than fail the whole search.
          contents.split(/\r?\n/).forEach((line, index) => {
            if (matches.length < limit && regex.test(line)) matches.push(`${childRelative}:${index + 1}: ${line}`);
          });
        }
      };
      await walk(root);
      return text(matches.length ? matches.join("\n") : "No matches.");
    })
  );

  const find = tool(
    "find",
    "Find files inside the allowed workspaces by a glob-style name pattern (supports * and **).",
    {
      pattern: z.string().min(1).max(300).describe("Glob against the path relative to the search root, e.g. '**/*.py'."),
      path: z.string().min(1).max(500).optional().describe("Directory to search, relative to the primary workspace. Defaults to its root."),
      maxResults: z.number().int().min(1).max(500).optional()
    },
    audited("find", false, boundary, sink, (args) => [args.path ?? "."], async (args) => {
      const regex = globToRegExp(args.pattern);
      const root = args.path ?? ".";
      const limit = args.maxResults ?? 200;
      const found: string[] = [];
      const walk = async (relativeDir: string): Promise<void> => {
        if (found.length >= limit) return;
        const absoluteDir = (await boundary.resolve(relativeDir)).absolute;
        for (const entry of await readdir(absoluteDir, { withFileTypes: true })) {
          if (found.length >= limit) return;
          if (entry.name.startsWith(".") || entry.name === "node_modules") continue;
          const childRelative = relativeDir === "." ? entry.name : `${relativeDir}/${entry.name}`;
          if (entry.isDirectory()) { await walk(childRelative); continue; }
          if (regex.test(childRelative)) found.push(childRelative);
        }
      };
      await walk(root);
      return text(found.length ? found.join("\n") : "No files matched.");
    })
  );

  const ls = tool(
    "ls",
    "List the entries of one directory inside the allowed workspaces.",
    { path: z.string().min(1).max(500).optional().describe("Directory to list, relative to the primary workspace. Defaults to its root.") },
    audited("ls", false, boundary, sink, (args) => [args.path ?? "."], async (args) => {
      const relativePath = args.path ?? ".";
      const absolute = (await boundary.resolve(relativePath)).absolute;
      const entries = await readdir(absolute, { withFileTypes: true });
      const lines = entries.map((entry) => `${entry.isDirectory() ? "d" : "-"} ${entry.name}`).sort();
      return text(lines.length ? lines.join("\n") : "(empty)");
    })
  );

  const tools: SdkMcpToolDefinition<any>[] = [read, grep, find, ls];
  if (!options.write) return tools;

  const write = tool(
    "write",
    "Create or overwrite a text file inside a writable workspace (aml-triage). Creates missing parent directories.",
    { path: z.string().min(1).max(500), content: z.string().max(400_000) },
    audited("write", true, boundary, sink, (args) => [args.path], async (args) => {
      await boundary.writeFile(args.path, args.content);
      return text(`Wrote ${args.path}`);
    })
  );

  const edit = tool(
    "edit",
    "Replace an exact, unique text match inside a file in a writable workspace (aml-triage).",
    {
      path: z.string().min(1).max(500),
      oldString: z.string().min(1).max(200_000),
      newString: z.string().max(200_000),
      replaceAll: z.boolean().optional().describe("Replace every occurrence instead of requiring exactly one.")
    },
    audited("edit", true, boundary, sink, (args) => [args.path], async (args) => {
      const current = (await boundary.readFile(args.path)).toString("utf8");
      const occurrences = current.split(args.oldString).length - 1;
      if (occurrences === 0) throw new Error(`oldString was not found in ${args.path}.`);
      if (!args.replaceAll && occurrences > 1) throw new Error(`oldString matches ${occurrences} places in ${args.path}; either make it unique or pass replaceAll.`);
      const updated = args.replaceAll ? current.split(args.oldString).join(args.newString) : current.replace(args.oldString, args.newString);
      await boundary.writeFile(args.path, updated);
      return text(`Edited ${args.path} (${occurrences} replacement${occurrences === 1 ? "" : "s"}).`);
    })
  );

  // Read, write and edit together can copy a file but cannot retire the
  // original, so without this the doer could only ever leave two of
  // everything when a lesson's step is a rename or relocation.
  const move = tool(
    "move",
    "Move or rename one file inside a writable workspace (aml-triage). Refuses a destination that already exists.",
    {
      path: z.string().min(1).max(400).describe("The file to move."),
      destination: z.string().min(1).max(400).describe("Where it should end up.")
    },
    audited("move", true, boundary, sink, (args) => [args.path, args.destination], async (args) => {
      const moved = await boundary.move(args.path, args.destination);
      return text(`Moved ${moved.from} to ${moved.to}`);
    })
  );

  return [...tools, write, edit, move];
}

/** A minimal glob-to-regex conversion: `**` crosses path separators, `*` does not, everything else is literal. */
function globToRegExp(pattern: string): RegExp {
  const escaped = pattern
    .split("**").map((part) => part.split("*").map((literal) => literal.replace(/[.+^${}()|[\]\\]/g, "\\$&")).join("[^/]*"))
    .join(".*");
  return new RegExp(`(^|/)${escaped}$`);
}
