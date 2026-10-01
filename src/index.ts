// dsh-errkb plugin entry. This file declares the plugin's surface and nothing
// else: capture, fingerprinting, matching and injection all arrive in T10-T16,
// and every registration point below is marked with the task that owns it.
import type { Context } from "@deepseek-ai/cordis";
import z from "@deepseek-ai/schemastery";
import { defaultProbe, formatKbLog, resolveKbDir } from "./paths";

// The plugin's runtime name, matching `id` in cordis.patch.yml. The installed
// harness plugins follow the same rule: dsh-spill-policy exports "spill-policy",
// dsh-persona exports "persona", and the third-party dsh-pr-watch exports
// "pr-watch" next to `id: pr-watch` in its own patch.
export const name = "err-kb";

// Real service names, copied from installed plugins rather than invented: the
// `tools` service backs the five tools registered in T15, and `systemPrompt`
// backs the standing section registered in T13.
export const inject = ["tools", "systemPrompt"];

// Declared, not consumed - apply() reads no setting yet. Every default mirrors
// the Settings table in README.md item for item. The documented value sets
// (captureFix: prompt-once|off, inject: hit-only|always|off, sessionDigest:
// off|counts|index, share: public|private) are not enforced yet; validation
// tightens when the settings are actually read in T10-T16. `labels` is the
// exception: it is new, so it starts out as the union it documents (§17 Q2).
// The store reads both label sets whatever this says; it only decides the
// language of blocks it writes.
export const Config = z.object({
  kbDir: z.string().default(""),
  idPrefix: z.string().default("E-"),
  idWidth: z.number().default(4),
  capture: z.array(z.string()).default(["tool", "command", "llm", "agent"]),
  captureExitCodes: z.boolean().default(true),
  transientThreshold: z.number().default(5),
  fuzzyThreshold: z.number().default(0.72),
  captureFix: z.string().default("prompt-once"),
  inject: z.string().default("hit-only"),
  sessionDigest: z.string().default("counts"),
  systemPromptHint: z.boolean().default(true),
  providers: z.array(z.string()).default(["*"]),
  share: z.string().default("public"),
  maxEntries: z.number().default(200),
  maxSampleChars: z.number().default(500),
  exportDir: z.string().default(""),
  labels: z.union(["en", "zh"]).default("en"),
});

// The whole behaviour of this task: one startup line, now carrying the resolved
// directory and the tier that produced it (T05).
export function apply(ctx: Context, config: Schemastery.TypeT<typeof Config>) {
  ctx.logger.info(formatKbLog(resolveKbDir(config.kbDir, defaultProbe())));

  // TODO(T11): register the agent/error and tools/result listeners here.
  // TODO(T13): register tools/post-execute, agent/pre-step and
  // agent/session-start, and publish the system-prompt section here.
  // TODO(T16): register the agent/request-error listener here.
}
