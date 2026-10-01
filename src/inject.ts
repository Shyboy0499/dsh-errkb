// Notices: what the model is told about a captured error, and how often (§7).
//
// Pure text plus three small pieces of per-session state, and no hooks: T13
// wires `tools/post-execute`, `agent/pre-step`, `agent/session-start` and the
// system-prompt section on top of this.
//
// Wording (English; the plugin's own UI text is English):
//
//   hit       [errkb] E-0007 known (5 hits) | cause: … | fix: … Known fix: try
//             this first, before re-diagnosing or researching.
//   near hit  … | fix: … Approximate match, verify first.
//   doubted   … | fix: … This fix failed here last time; verify before applying.
//   no fix    [errkb] E-0007 seen before (5 hits), no fix recorded yet.
//   miss      [errkb] recorded as E-0011 (no fix yet).
//
// The hit wording answers docs/discussions.md §1.2: it orders the work ("try
// this first") instead of forbidding any ("do not re-diagnose"), so it does not
// contradict a research nudge from another plugin on the same step. A hit
// without a fix still speaks, briefly: the model learns the error is a repeat
// and that its fix is worth recording, for about 15 tokens.
//
// Caps (§7). The body is at most 400 characters and at most 120 tokens by
// estimateTokens(); cause and fix are clipped to fit, the fix last, so the
// closing instruction is never cut. The message source is the real
// `MessageSourceMap['plugin']` shape from @deepseek-ai/dsh-llm
// (lib/types/message.d.ts) with `form: 'notice'`, and its summary goes through
// that package's own `boundContextSummary` (≤ `CONTEXT_SUMMARY_MAX_CHARS`, 120).
// CapTracker allows 1 notice per step, 3 per turn and 2 per ID per session, and
// a `fixed` entry 1 per session.
//
// Fix trust (docs/discussions.md §4). FixTrust counts, per ID, how often a fix
// was injected and how often the same entry was captured again later in the
// same turn. One recurrence turns the wording into "This fix failed here last
// time"; two with no recorded success stop the ID from being injected on this
// machine. Editing the entry's fix starts its count again. The state is a plain
// serializable object behind TrustStore, held in memory for now; it is
// machine-local and never written into ERRORS.md.
import { createHash } from "node:crypto";
import {
  CONTEXT_SUMMARY_MAX_CHARS,
  boundContextSummary,
} from "@deepseek-ai/dsh-llm";
import type { MessageSourceMap } from "@deepseek-ai/dsh-llm";
import type { Hit } from "./match";

/** The plugin name every notice source carries; matches `name` in index.ts. */
export const PLUGIN_NAME = "err-kb";

/** A notice body never exceeds this many characters (§7). */
export const NOTICE_MAX_CHARS = 400;

/** A notice body never exceeds this many tokens by estimateTokens() (§7). */
export const NOTICE_MAX_TOKENS = 120;

/** The summary bound, re-exported from dsh-llm so callers need one import. */
export const SUMMARY_MAX_CHARS = CONTEXT_SUMMARY_MAX_CHARS;

/** The cause is clipped to this many characters before anything else. */
export const CAUSE_MAX_CHARS = 100;

/** When a notice is too long, the cause gives way down to this, then the fix. */
export const CAUSE_MIN_CHARS = 40;

/** Marks clipped text. */
export const ELLIPSIS = "…";

/** Values of the `inject` setting. */
export const INJECT_MODES = ["hit-only", "always", "off"] as const;

/** `hit-only` keeps misses silent; `always` announces them; `off` says nothing. */
export type InjectMode = (typeof INJECT_MODES)[number];

/** The closing sentences, exactly as the model reads them. */
export const WORDING = {
  hit: "Known fix: try this first, before re-diagnosing or researching.",
  approximate: "Approximate match, verify first.",
  doubted: "This fix failed here last time; verify before applying.",
} as const;

/** How far a fix is trusted on this machine. */
export type TrustLevel = "trusted" | "doubted" | "suppressed";

/** What a notice says, which also names its wording. */
export type NoticeKind = "hit" | "near" | "doubted" | "no-fix" | "miss";

/** The message source of a notice: dsh-llm's plugin source, `notice` form. */
export type NoticeSource = MessageSourceMap["plugin"] & {
  readonly form: "notice";
  readonly summary: string;
};

/** One notice, ready to inject. */
export interface Notice {
  id: string;
  kind: NoticeKind;
  /** The model-facing body: one line, within both caps. */
  text: string;
  source: NoticeSource;
}

/** What happened to a captured error, as the injector sees it. */
export type NoticeEvent =
  | { kind: "hit"; hit: Hit }
  /** No entry matched and the store recorded a new one under `id`. */
  | { kind: "miss"; id: string };

// ---------------------------------------------------------------------------
// Text

// Code points outside printable ASCII and its whitespace. Each counts as a
// whole token: CJK characters usually are one, and accented or symbol
// characters are at most one in practice. (Bodies are one line, so other
// ASCII control characters never reach this.)
const NON_ASCII = /[^ -~\t\n\r]/gu;

/** ASCII characters per token in estimateTokens(); English averages about 4. */
export const ASCII_CHARS_PER_TOKEN = 3;

/**
 * A conservative token estimate that needs no tokenizer: every non-ASCII code
 * point counts as one token, ASCII as one per {@link ASCII_CHARS_PER_TOKEN}
 * characters, rounded up. It overestimates English by about a quarter and
 * Chinese by a little more, so a notice inside the cap is inside it for real.
 *
 * @param text - any text.
 * @returns the estimated token count.
 */
export function estimateTokens(text: string): number {
  const wide = text.match(NON_ASCII)?.length ?? 0;
  const ascii = text.length - wide - surrogateUnits(text);
  return Math.ceil(ascii / ASCII_CHARS_PER_TOKEN) + wide;
}

// A non-BMP code point is two UTF-16 units but one match above; take the
// second unit back out of the ASCII count.
function surrogateUnits(text: string): number {
  return text.length - Array.from(text).length;
}

/**
 * Whether a body is inside both caps.
 *
 * @param text - a notice body.
 */
export function withinCaps(text: string): boolean {
  return (
    Array.from(text).length <= NOTICE_MAX_CHARS &&
    estimateTokens(text) <= NOTICE_MAX_TOKENS
  );
}

/** Collapse runs of whitespace, newlines included, to one space. */
export function oneLine(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

/**
 * Clip text to at most `max` code points, the last of them an ellipsis when
 * anything was cut.
 *
 * @param text - the text.
 * @param max - the limit in code points; 0 or less gives the empty string.
 */
export function clip(text: string, max: number): string {
  const chars = Array.from(text);
  if (chars.length <= max) return text;
  if (max <= 0) return "";
  return `${chars
    .slice(0, max - 1)
    .join("")
    .trimEnd()}${ELLIPSIS}`;
}

// The largest n in [low, high] for which ok(n) holds, assuming ok is
// monotone; undefined when ok(low) does not.
function largest(
  low: number,
  high: number,
  ok: (n: number) => boolean,
): number | undefined {
  if (!ok(low)) return undefined;
  while (low < high) {
    const mid = Math.ceil((low + high) / 2);
    if (ok(mid)) low = mid;
    else high = mid - 1;
  }
  return low;
}

/**
 * Fit a cause and a fix into a template within both caps. The cause is first
 * clipped to {@link CAUSE_MAX_CHARS}; if the result is still too long the cause
 * shrinks towards {@link CAUSE_MIN_CHARS}, and only then does the fix shrink.
 * Whatever the template adds stays whole. A template too long on its own (an
 * absurd ID) is clipped as a last resort.
 *
 * @param render - builds the body from a clipped cause and fix.
 * @param cause - the cause, one line.
 * @param fix - the fix, one line.
 * @returns a body inside both caps.
 */
export function fitNotice(
  render: (cause: string, fix: string) => string,
  cause: string,
  fix: string,
): string {
  const causeChars = Math.min(Array.from(cause).length, CAUSE_MAX_CHARS);
  const fixChars = Array.from(fix).length;
  const fits = (c: number, f: number) =>
    withinCaps(render(clip(cause, c), clip(fix, f)));

  if (fits(causeChars, fixChars)) return render(clip(cause, causeChars), fix);
  const floor = Math.min(causeChars, CAUSE_MIN_CHARS);
  const c = largest(floor, causeChars, (n) => fits(n, fixChars));
  if (c !== undefined) return render(clip(cause, c), fix);
  const f = largest(0, fixChars, (n) => fits(floor, n));
  if (f !== undefined) return render(clip(cause, floor), clip(fix, f));
  return hardClip(render(clip(cause, floor), ""));
}

// Clip a whole body until it is inside both caps.
function hardClip(text: string): string {
  const n = largest(0, Array.from(text).length, (m) =>
    withinCaps(clip(text, m)),
  ) as number;
  return clip(text, n);
}

const hitsText = (hits: number) => `${hits} ${hits === 1 ? "hit" : "hits"}`;

/**
 * The body of a notice. Pure; caps are applied here, counts are not.
 *
 * @param event - the hit or miss.
 * @param trust - the fix's trust level; ignored for a miss or an entry
 *   without a fix.
 * @returns the kind and the body.
 */
export function noticeText(
  event: NoticeEvent,
  trust: TrustLevel = "trusted",
): { kind: NoticeKind; text: string } {
  if (event.kind === "miss")
    return {
      kind: "miss",
      text: hardClip(`[errkb] recorded as ${event.id} (no fix yet).`),
    };

  const { hit } = event;
  const { id, entry } = hit;
  const fix = oneLine(entry.fix);
  const near = hit.approximate ? ", approximate match" : "";
  if (fix === "")
    return {
      kind: "no-fix",
      text: hardClip(
        `[errkb] ${id} seen before (${hitsText(entry.hits)}${near}), no fix recorded yet.`,
      ),
    };

  const doubted = trust !== "trusted";
  const closing: string[] = [
    ...(hit.approximate ? [WORDING.approximate] : []),
    ...(doubted ? [WORDING.doubted] : []),
  ];
  if (closing.length === 0) closing.push(WORDING.hit);
  const kind: NoticeKind = doubted
    ? "doubted"
    : hit.approximate
      ? "near"
      : "hit";

  const cause = oneLine(entry.trigger);
  const head = `[errkb] ${id} known (${hitsText(entry.hits)})`;
  const tail = closing.join(" ");
  const render = (c: string, f: string) =>
    `${head}${c === "" ? "" : ` | cause: ${c}`} | fix: ${f} ${tail}`;
  return { kind, text: fitNotice(render, cause, fix) };
}

/**
 * The message source for a notice body: `{kind: 'plugin', plugin: 'err-kb',
 * form: 'notice', summary}`, the summary bounded by dsh-llm's
 * `boundContextSummary`.
 *
 * @param text - the notice body.
 */
export function noticeSource(text: string): NoticeSource {
  return {
    kind: "plugin",
    plugin: PLUGIN_NAME,
    form: "notice",
    summary: boundContextSummary(text),
  };
}

// ---------------------------------------------------------------------------
// Caps

/** The notice budget (§7). Every limit can only be lowered. */
export interface CapLimits {
  perStep: number;
  perTurn: number;
  perIdPerSession: number;
  /** For an entry whose status is `fixed`. */
  fixedPerSession: number;
}

export const DEFAULT_CAP_LIMITS: CapLimits = {
  perStep: 1,
  perTurn: 3,
  perIdPerSession: 2,
  fixedPerSession: 1,
};

/**
 * Counts one session's notices. Create one per session; call beginTurn() and
 * beginStep() at those boundaries, and tryEmit() before each notice.
 */
export class CapTracker {
  readonly limits: CapLimits;
  private step = 0;
  private turn = 0;
  private readonly perId = new Map<string, number>();

  /** @param limits - lower limits; a value above the default is ignored. */
  constructor(limits: Partial<CapLimits> = {}) {
    const l = { ...DEFAULT_CAP_LIMITS };
    for (const key of Object.keys(l) as (keyof CapLimits)[])
      l[key] = Math.min(l[key], limits[key] ?? l[key]);
    this.limits = l;
  }

  /** A new turn: the turn and step budgets start again. */
  beginTurn(): void {
    this.turn = 0;
    this.step = 0;
  }

  /** A new step within the turn: the step budget starts again. */
  beginStep(): void {
    this.step = 0;
  }

  /**
   * Take one notice from every budget, if every budget has one left.
   *
   * @param id - the entry the notice is about.
   * @param fixed - whether the entry's status is `fixed`.
   * @returns whether the notice may be emitted; nothing is taken when not.
   */
  tryEmit(id: string, fixed = false): boolean {
    const used = this.perId.get(id) ?? 0;
    const perId = fixed
      ? this.limits.fixedPerSession
      : this.limits.perIdPerSession;
    if (
      this.step >= this.limits.perStep ||
      this.turn >= this.limits.perTurn ||
      used >= perId
    )
      return false;
    this.step++;
    this.turn++;
    this.perId.set(id, used + 1);
    return true;
  }
}

// ---------------------------------------------------------------------------
// Fix trust

/** One entry's trust counters. */
export interface TrustRecord {
  /** Notices that carried this entry's fix. */
  injected: number;
  /** Times the entry was captured again later in a turn that injected it. */
  recurredAfterInject: number;
  /** Times the fix was confirmed to work (T14). */
  succeeded: number;
  /** Hash of the fix text the counts are about. */
  fixSig: string;
}

/** Everything FixTrust keeps: plain data, safe to JSON.stringify. */
export interface TrustState {
  entries: Record<string, TrustRecord>;
}

/** Where trust state lives; state.json later, memory for now. */
export interface TrustStore {
  load(): TrustState;
  save(state: TrustState): void;
}

/**
 * A TrustStore in memory. It saves a deep copy, so later changes to the saved
 * object do not leak in.
 *
 * @param initial - the state to start from.
 */
export function memoryTrustStore(
  initial: TrustState = { entries: {} },
): TrustStore {
  let state: TrustState = structuredClone(initial);
  return {
    load: () => structuredClone(state),
    save: (next) => {
      state = structuredClone(next);
    },
  };
}

/** Recurrences after which the wording turns to "failed here last time". */
export const DOUBT_AFTER = 1;

/** Recurrences, with no success, after which the ID is not injected. */
export const SUPPRESS_AFTER = 2;

/**
 * Hash a fix text, so an edited fix starts trusted again.
 *
 * @param fix - the entry's fix field.
 */
export function fixSig(fix: string): string {
  return createHash("sha256").update(oneLine(fix)).digest("hex").slice(0, 12);
}

/**
 * The trust level a record gives.
 *
 * @param record - the record, if there is one.
 */
export function trustLevel(record: TrustRecord | undefined): TrustLevel {
  if (record === undefined) return "trusted";
  if (record.recurredAfterInject >= SUPPRESS_AFTER && record.succeeded === 0)
    return "suppressed";
  return record.recurredAfterInject >= DOUBT_AFTER ? "doubted" : "trusted";
}

/**
 * Per-machine fix trust. Call beginTurn() at each turn, seen() whenever a
 * captured error matches an entry, and injected() when a notice carried that
 * entry's fix.
 */
export class FixTrust {
  private state: TrustState;
  private readonly injectedThisTurn = new Set<string>();

  constructor(private readonly store: TrustStore = memoryTrustStore()) {
    this.state = store.load();
  }

  /** The record for an entry, or undefined when none applies to this fix. */
  record(id: string, fix: string): TrustRecord | undefined {
    const record = this.state.entries[id];
    return record?.fixSig === fixSig(fix) ? record : undefined;
  }

  /** The trust level of an entry's current fix. */
  level(id: string, fix: string): TrustLevel {
    return trustLevel(this.record(id, fix));
  }

  /** A new turn: recurrence is counted within one turn. */
  beginTurn(): void {
    this.injectedThisTurn.clear();
  }

  /**
   * A captured error matched `id`. If its fix was injected earlier in this
   * turn, the fix did not hold: count a recurrence.
   */
  seen(id: string, fix: string): void {
    const record = this.record(id, fix);
    if (record === undefined || !this.injectedThisTurn.has(id)) return;
    record.recurredAfterInject++;
    this.injectedThisTurn.delete(id);
    this.store.save(this.state);
  }

  /** A notice carried `id`'s fix. */
  injected(id: string, fix: string): void {
    const record = this.record(id, fix) ?? {
      injected: 0,
      recurredAfterInject: 0,
      succeeded: 0,
      fixSig: fixSig(fix),
    };
    record.injected++;
    this.state.entries[id] = record;
    this.injectedThisTurn.add(id);
    this.store.save(this.state);
  }

  /** The fix for `id` was confirmed to work (resolution detection, T14). */
  succeeded(id: string, fix: string): void {
    const record = this.record(id, fix);
    if (record === undefined) return;
    record.succeeded++;
    this.store.save(this.state);
  }

  /** A copy of the whole state, for persistence or `err_stats`. */
  snapshot(): TrustState {
    return structuredClone(this.state);
  }
}

// ---------------------------------------------------------------------------
// The injector

/** Everything an injector needs; all of it is per session except trust. */
export interface InjectorDeps {
  mode?: InjectMode;
  caps?: CapTracker;
  trust?: FixTrust;
}

/**
 * Decides, per session, which captured errors become notices: the `inject`
 * setting, non-injectable entries, fix trust and the caps, in that order.
 */
export class Injector {
  readonly mode: InjectMode;
  readonly caps: CapTracker;
  readonly trust: FixTrust;

  constructor(deps: InjectorDeps = {}) {
    this.mode = deps.mode ?? "hit-only";
    this.caps = deps.caps ?? new CapTracker();
    this.trust = deps.trust ?? new FixTrust();
  }

  beginTurn(): void {
    this.caps.beginTurn();
    this.trust.beginTurn();
  }

  beginStep(): void {
    this.caps.beginStep();
  }

  /**
   * Offer a captured error.
   *
   * @param event - the hit or the miss.
   * @returns the notice to inject, or undefined to stay silent.
   */
  offer(event: NoticeEvent): Notice | undefined {
    if (this.mode === "off") return undefined;
    if (event.kind === "miss") {
      if (this.mode !== "always" || !this.caps.tryEmit(event.id))
        return undefined;
      return notice(event.id, noticeText(event));
    }

    const { id, entry, injectable } = event.hit;
    this.trust.seen(id, entry.fix);
    if (!injectable) return undefined;
    const carriesFix = oneLine(entry.fix) !== "";
    const level = carriesFix ? this.trust.level(id, entry.fix) : "trusted";
    if (level === "suppressed") return undefined;
    if (!this.caps.tryEmit(id, entry.status === "fixed")) return undefined;
    if (carriesFix) this.trust.injected(id, entry.fix);
    return notice(id, noticeText(event, level));
  }
}

function notice(
  id: string,
  { kind, text }: { kind: NoticeKind; text: string },
): Notice {
  return { id, kind, text, source: noticeSource(text) };
}
