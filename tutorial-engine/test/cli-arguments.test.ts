import { describe, expect, it } from "vitest";
import { ArgumentError, parseArguments } from "../src/cli-arguments.js";

function run(argv: string[]) {
  const parsed = parseArguments(argv);
  if (parsed.kind !== "run") throw new Error("Expected a run, not help.");
  return parsed.options;
}

describe("parseArguments", () => {
  it("serves the named directory on an ephemeral loopback port by default", () => {
    expect(run(["/tutorials/aml-tutor"])).toEqual({ target: "/tutorials/aml-tutor", triage: undefined, port: undefined, host: undefined, noOpen: false });
  });

  it("reads the port, host, triage, and browser flags", () => {
    expect(run(["/tutorials/aml-tutor", "--port", "4310", "--host", "0.0.0.0", "--triage", "/tutorials/aml-triage", "--no-open"]))
      .toEqual({ target: "/tutorials/aml-tutor", triage: "/tutorials/aml-triage", port: 4310, host: "0.0.0.0", noOpen: true });
  });

  it("accepts flags before the directory, because a flag value is never the target", () => {
    expect(run(["--host", "0.0.0.0", "--port", "4310", "/tutorials/aml-tutor"]))
      .toEqual({ target: "/tutorials/aml-tutor", triage: undefined, port: 4310, host: "0.0.0.0", noOpen: false });
  });

  it("accepts --flag=value", () => {
    expect(run(["--port=4310", "--host=::1", "--triage=../aml-triage", "/tutorials/aml-tutor"]))
      .toMatchObject({ target: "/tutorials/aml-tutor", triage: "../aml-triage", port: 4310, host: "::1" });
  });

  it("treats everything after -- as a path, so directories may start with a dash", () => {
    expect(run(["--no-open", "--", "--tutorials"])).toMatchObject({ target: "--tutorials", noOpen: true });
  });

  it("asks for a directory when none is named", () => {
    expect(() => parseArguments([])).toThrow(ArgumentError);
    expect(() => parseArguments(["--no-open"])).toThrow(/Name the tutorial directory/);
  });

  it("refuses to guess between two directories", () => {
    expect(() => parseArguments(["/one", "/two"])).toThrow(/one tutorial directory at a time/);
  });

  it("rejects an unknown option rather than ignoring a typo", () => {
    expect(() => parseArguments(["/tutorials/aml-tutor", "--noopen"])).toThrow(/Unknown option '--noopen'/);
    expect(() => parseArguments(["/tutorials/aml-tutor", "--open=yes"])).toThrow(/Unknown option '--open'/);
  });

  it("rejects a value on a flag that takes none", () => {
    expect(() => parseArguments(["/tutorials/aml-tutor", "--no-open=true"])).toThrow(/--no-open does not take a value/);
  });

  it("will not let a value flag swallow the following flag", () => {
    expect(() => parseArguments(["/tutorials/aml-tutor", "--port", "--no-open"])).toThrow(/--port needs a value/);
    expect(() => parseArguments(["/tutorials/aml-tutor", "--host"])).toThrow(/--host needs a value/);
    expect(() => parseArguments(["/tutorials/aml-tutor", "--host="])).toThrow(/--host needs a value/);
    expect(() => parseArguments(["/tutorials/aml-tutor", "--triage"])).toThrow(/--triage needs a value/);
  });

  it("insists on a plain port number in range", () => {
    for (const value of ["", "web", "0x10", "1e3", " 80", "80.5", "-1"]) {
      expect(() => parseArguments(["/tutorials/aml-tutor", `--port=${value}`])).toThrow(ArgumentError);
    }
    expect(() => parseArguments(["/tutorials/aml-tutor", "--port=65536"])).toThrow(/between 0 and 65535/);
    expect(run(["/tutorials/aml-tutor", "--port=65535"]).port).toBe(65535);
  });

  it("rejects a URL where a host belongs", () => {
    expect(() => parseArguments(["/tutorials/aml-tutor", "--host=http://0.0.0.0"])).toThrow(/not a URL or path/);
  });

  it("reports help without needing a directory", () => {
    expect(parseArguments(["--help"])).toEqual({ kind: "help" });
    expect(parseArguments(["-h"])).toEqual({ kind: "help" });
  });
});
