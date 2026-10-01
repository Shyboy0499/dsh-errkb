// dsh-errkb plugin entry. This file declares the plugin's surface and wires it:
// apply() resolves the knowledge base, binds a recorder to it and registers the
// two capture listeners (T11). The pipeline itself lives in src/plugin.ts;
// every registration point still to come is marked with the task that owns it.
import type { Context } from "@deepseek-ai/cordis";
import z from "@deepseek-ai/schemastery";
import { defaultProbe, formatKbLog, resolveKbDir } from "./paths";
import { createRecorder, registerListeners } from "./plugin";
import type { RecorderOptions } from "./plugin";

// The plugin's runtime name, matching `id` in cordis.patch.yml. The installed
// harness plugins follow the same rule: dsh-spill-policy exports "spill-policy",
// dsh-persona exports "persona", and the third-party dsh-pr-watch exports
// "pr-watch" next to `id: pr-watch` in its own patch.
export const name = "err-kb";

// Real service names, copied from installed plugins rather than invented: the
// `tools` service backs the five tools registered in T15, and `systemPrompt`
// backs the standing section registered in T13.
export const inject = ["tools", "systemPrompt"];

// Every default mirrors the Settings table in README.md item for item. T11
// reads the capture and store settings (see recorderOptions); captureFix,
// inject, sessionDigest, systemPromptHint, providers and exportDir are still
// only declared. The documented value sets (captureFix: prompt-once|off,
// inject: hit-only|always|off, sessionDigest: off|counts|index) are not
// enforced yet; validation tightens when those settings are read in T12-T16.
// `share` is read, and anything but "private" counts as "public", the safer
// of the two. `labels` is the exception: it is new, so it starts out as the
// union it documents (§17 Q2). The store reads both label sets whatever this
// says; it only decides the language of blocks it writes.
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

/** The plugin's settings, as apply() receives them. */
export type Config = Schemastery.TypeT<typeof Config>;

/**
 * The recorder settings a configuration selects.
 *
 * @param config - the plugin's settings.
 * @returns capture, matching and store settings for createRecorder().
 */
export function recorderOptions(config: Config): RecorderOptions {
  return {
    capture: config.capture,
    captureExitCodes: config.captureExitCodes,
    transientThreshold: config.transientThreshold,
    fuzzyThreshold: config.fuzzyThreshold,
    share: config.share === "private" ? "private" : "public",
    maxEntries: config.maxEntries,
    maxSampleChars: config.maxSampleChars,
    labels: config.labels,
    idPrefix: config.idPrefix,
    idWidth: config.idWidth,
  };
}

// One startup line with the resolved directory and its tier (T05), then the
// two capture listeners (T11). Nothing is injected into the model's context
// yet.
export function apply(ctx: Context, config: Config) {
  const kb = resolveKbDir(config.kbDir, defaultProbe());
  ctx.logger.info(formatKbLog(kb));
  const recorder = createRecorder({
    files: kb.files,
    logger: ctx.logger,
    options: recorderOptions(config),
  });
  registerListeners(ctx, recorder);

  // TODO(T13): register tools/post-execute, agent/pre-step and
  // agent/session-start, and publish the system-prompt section here.
  // TODO(T16): register the agent/request-error listener here.
}
