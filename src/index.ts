// dsh-errkb plugin entry. This file declares the plugin's surface and wires it:
// apply() resolves the knowledge base, binds a recorder to it, and registers the
// two capture listeners (T11) together with the four injection points (T13).
// The pipeline itself lives in src/plugin.ts; every registration point still
// to come is marked with the task that owns it.
import type { Context } from "@deepseek-ai/cordis";
import z from "@deepseek-ai/schemastery";
import { defaultProbe, formatKbLog, resolveKbDir } from "./paths";
import { INJECT_MODES, SESSION_DIGEST_MODES } from "./inject";
import type { InjectMode, SessionDigestMode } from "./inject";
import {
  DEFAULT_INJECTION_OPTIONS,
  createInjection,
  createRecorder,
  registerInjection,
} from "./plugin";
import type { InjectionOptions, RecorderOptions } from "./plugin";

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
// reads the capture and store settings (see recorderOptions), T13 reads
// inject, sessionDigest and systemPromptHint (see injectionOptions);
// captureFix, providers and exportDir are still only declared. inject and
// sessionDigest stay strings so an existing profile with a typo still loads:
// a value outside the documented set (inject: hit-only|always|off,
// sessionDigest: off|counts|index) falls back to the default. captureFix
// (prompt-once|off) is not validated yet.
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

/** `value` when it is one of `allowed`, else `fallback`. */
function oneOf<T extends string>(
  value: string,
  allowed: readonly T[],
  fallback: T,
): T {
  return (allowed as readonly string[]).includes(value)
    ? (value as T)
    : fallback;
}

/**
 * The injection settings a configuration selects.
 *
 * @param config - the plugin's settings.
 * @returns the settings for createInjection(); an unknown `inject` or
 *   `sessionDigest` value takes its default.
 */
export function injectionOptions(config: Config): InjectionOptions {
  return {
    inject: oneOf<InjectMode>(
      config.inject,
      INJECT_MODES,
      DEFAULT_INJECTION_OPTIONS.inject,
    ),
    sessionDigest: oneOf<SessionDigestMode>(
      config.sessionDigest,
      SESSION_DIGEST_MODES,
      DEFAULT_INJECTION_OPTIONS.sessionDigest,
    ),
    systemPromptHint: config.systemPromptHint,
  };
}

// One startup line with the resolved directory and its tier (T05), then the
// two capture listeners (T11) and the four injection points (T13).
export function apply(ctx: Context, config: Config) {
  const kb = resolveKbDir(config.kbDir, defaultProbe());
  ctx.logger.info(formatKbLog(kb));
  const recorder = createRecorder({
    files: kb.files,
    logger: ctx.logger,
    options: recorderOptions(config),
  });
  const injection = createInjection({
    recorder,
    options: injectionOptions(config),
  });
  registerInjection(ctx, injection, recorder);

  // TODO(T16): register the agent/request-error listener here.
}
