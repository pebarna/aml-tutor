#!/usr/bin/env node
import { query } from "@anthropic-ai/claude-agent-sdk";
import { stderr, stdout } from "node:process";

/**
 * The tutorial runs two agents on purpose, and they want opposite things.
 *
 * The tutor teaches, so it wants the largest model available; TUTOR_MODEL
 * names it. The doer `automate_step` drives wants to be cheap and fast, and
 * its mistakes are teaching material, so it is left to the SDK's ordinary
 * default. Neither should silently become the other.
 *
 * `tutorial-engine/src/agent/claude-agent-adapter.ts` resolves TUTOR_MODEL for
 * real; this mirrors it so `npm run setup` can report what the tutor will do.
 * The env var name is carried over unchanged from the Pi-based tutorial this
 * was forked from.
 */
export const TUTOR_MODEL_ENV = "TUTOR_MODEL";

/**
 * Check that the Claude Agent SDK can actually reach a model, by issuing one
 * minimal, live `query()` call rather than guessing from environment
 * variables alone — `ANTHROPIC_API_KEY` can be set and still be invalid, and
 * an existing Claude Code CLI login session leaves no env var to check at
 * all. This mirrors what the Pi-based tutorial's setup script did with
 * `ModelRuntime.getAvailable()`: a live probe, not an assumption.
 */
export async function checkReadiness(runQuery = query) {
  try {
    // disableAllHooks isolates this probe from ambient hooks (a stray
    // permission prompt, a SessionStart hook, etc. can otherwise intercept the
    // turn, costing real money on every `npm run setup` and burning through
    // maxTurns before the model's own reply lands, turning a real
    // "not authenticated" into a false one).
    //
    // settingSources stays scoped to 'user' rather than [] because on a
    // machine whose Claude Code auth comes from an installer syncing
    // ANTHROPIC_BASE_URL/ANTHROPIC_AUTH_TOKEN into ~/.claude/settings.json's
    // `env` block (common in enterprise setups — no `claude login`, no
    // ANTHROPIC_API_KEY env var to check), settingSources: [] blocks that
    // block too, and this probe (and every tutor/doer session) would then
    // report "not ready" on an already-authenticated machine.
    const probe = runQuery({
      prompt: "Reply with the single word ready.",
      options: { maxTurns: 2, settingSources: ["user"], settings: { disableAllHooks: true } }
    });
    let sawResult = false;
    for await (const message of probe) {
      if (message.type === "result") {
        sawResult = true;
        if (message.subtype !== "success") return { ready: false, reason: message.subtype };
      }
    }
    return { ready: sawResult, reason: sawResult ? undefined : "no result was returned" };
  } catch (error) {
    return { ready: false, reason: error instanceof Error ? error.message : String(error) };
  }
}

export function describeTutorModel(requested) {
  const wanted = requested?.trim();
  return wanted ? `${wanted} (from ${TUTOR_MODEL_ENV})` : `left to the SDK's default, because ${TUTOR_MODEL_ENV} is unset`;
}

function report() {
  return [
    `Tutor model: ${describeTutorModel(process.env[TUTOR_MODEL_ENV])}`,
    "Doer model:  left to the SDK's default — cheap and fast is the point; the lessons teach you to catch its mistakes.",
    "",
    `Give the tutor a capable model by exporting ${TUTOR_MODEL_ENV}=<model>, e.g. claude-opus-4-8.`
  ];
}

async function main() {
  const result = await checkReadiness();
  if (!result.ready) {
    stderr.write(`The Claude Agent SDK could not reach a model${result.reason ? ` (${result.reason})` : ""}.\n`);
    stderr.write("Run 'claude login', or export ANTHROPIC_API_KEY, then run 'npm run setup' again.\n");
    process.exitCode = 1;
    return;
  }
  stdout.write("The Claude Agent SDK is authenticated and ready for the tutorial.\n");
  stdout.write(`${report().join("\n")}\n`);
}

if (import.meta.main) {
  main().catch((error) => {
    stderr.write(`Unable to check readiness: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
