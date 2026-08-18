import { describe, expect, it } from "vitest";
import { choiceOptionSchema } from "../src/agent/tutorial-tools.js";
import { choiceIconCategories } from "../src/protocol/events.js";

describe("choice option schema", () => {
  it("requires one of the fixed icon categories", () => {
    for (const icon of choiceIconCategories) {
      expect(choiceOptionSchema.safeParse({ id: "next", label: "Continue", icon }).success).toBe(true);
    }
    expect(choiceOptionSchema.safeParse({ id: "next", label: "Continue" }).success).toBe(false);
    expect(choiceOptionSchema.safeParse({ id: "next", label: "Continue", icon: "custom" }).success).toBe(false);
  });
});
