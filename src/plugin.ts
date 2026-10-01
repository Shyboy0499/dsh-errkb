// The capture pipeline behind the plugin's listeners (T11): a hook payload goes
// in, an entry in ERRORS.md comes out - or nothing, and never a throw.
//
// Two hooks feed it, both `emit` events whose listener return value is ignored
// (design document §2, §6). Their payload types are the real ones, read off the
// installed packages rather than guessed:
//
// - `agent/error` - `Events['agent/error']` in @deepseek-ai/dsh-agent
//   (lib/types/runtime-types.d.ts): `{ agent: Agent; turn; step; error:
//   unknown }`. `Agent.id` is the session id (`SessionId`, lib/types/types.d.ts).
// - `tools/result` - `Events['tools/result']` in @deepseek-ai/dsh-tools
//   (lib/types/index.d.ts): `(exec: Readonly<ToolExecution>, result:
//   Readonly<ToolExecutionResult>)`. A failure is `ToolExecutionFailure`, with
//   `error: ToolFailure` (`message`, `info?: ToolErrorInfo` with `code`); a
//   success carries `content: ContentBlock[]` (dsh-llm), whose `TextBlock`s the
//   command sniffing reads. `exec.agent` is optional: a call made outside an
//   agent has none.
//
// Each captured error goes through classify() (T10), then, off the turn, through
// match() (T09) against the store's current entries and into the store (T08):
//
// - `record`: a hit bumps the entry's hits and last-seen; a miss appends a new
//   entry under the next ID.
// - `count-only`: an entry with the same signature - a transient error promoted
//   earlier - still has its hits and last-seen bumped; without one, nothing is
//   written. This closes the T10 gap where every repeat after the promotion
//   came back `count-only` and went uncounted.
//
// Safety (§13). Listeners only classify and enqueue: they never await, and a
// throw anywhere is caught, counted and logged at most once a minute through
// the logger. Writes are chained per knowledge base so they run one at a time
// in arrival order, without blocking the turn. Each write gets 500 ms in total
// and up to 3 retries; the store's lock wait is bounded by what is left, and a
// lock that is still busy when the time is up skips that write silently. A
// write that has already taken the lock is not cut off - the store has no way
// to abandon it half done - so the bound is on waiting, not on the write.
//
// Transient errors are counted per session: one TransientCounter per
// `Agent.id`, from `agent/error`'s payload or `exec.agent` on `tools/result`.
// A payload without an agent counts under one plugin-wide counter. Only the 64
// most recently active sessions keep a counter, so a long-lived host does not
// grow without bound.
import type { Context } from "@deepseek-ai/cordis";
import type {
  ToolExecution,
  ToolExecutionResult,
} from "@deepseek-ai/dsh-tools";
import {
  DEFAULT_CAPTURE_OPTIONS,
  TransientCounter,
  classify,
  safeErrorText,
} from "./capture";
import type { CaptureInput, CaptureOptions, Classified } from "./capture";
import { DEFAULT_MATCH_OPTIONS, indexEntries, match } from "./match";
import type { IndexedEntry, MatchOptions } from "./match";
import type { KbFiles } from "./paths";
import {
  DEFAULT_STORE_OPTIONS,
  LockTimeoutError,
  ParseError,
  StoreCorruptError,
  createStore,
  formatSeen,
  nodeStoreFs,
  systemClock,
} from "./store";
import type {
  Entry,
  ErrorStore,
  StoreClock,
  StoreFs,
  StoreOptions,
} from "./store";

/** Retries after a failed write, within {@link WRITE_TIMEOUT_MS} (§13). */
export const WRITE_RETRIES = 3;

/** The time one write may spend, retries and lock waits included (§13). */
export const WRITE_TIMEOUT_MS = 500;

/** Pause before a retry. */
export const RETRY_DELAY_MS = 25;

/** At most one failure is logged per this many milliseconds. */
export const FAILURE_LOG_INTERVAL_MS = 60_000;

/** Sessions that keep a transient counter; the least recently active goes first. */
export const MAX_SESSIONS = 64;

/** The counter key for payloads that name no agent. */
export const NO_SESSION = "";

/** Recorder settings; every one has a default. */
export interface RecorderOptions
  extends
    CaptureOptions,
    MatchOptions,
    Pick<
      StoreOptions,
      | "idPrefix"
      | "idWidth"
      | "maxEntries"
      | "labels"
      | "share"
      | "maxSampleChars"
    > {}

export const DEFAULT_RECORDER_OPTIONS: RecorderOptions = {
  ...DEFAULT_CAPTURE_OPTIONS,
  ...DEFAULT_MATCH_OPTIONS,
  idPrefix: DEFAULT_STORE_OPTIONS.idPrefix,
  idWidth: DEFAULT_STORE_OPTIONS.idWidth,
  maxEntries: DEFAULT_STORE_OPTIONS.maxEntries,
  labels: DEFAULT_STORE_OPTIONS.labels,
  share: DEFAULT_STORE_OPTIONS.share,
  maxSampleChars: DEFAULT_STORE_OPTIONS.maxSampleChars,
};

/** The one logger method the recorder uses; `ctx.logger` fits. */
export interface RecorderLogger {
  warn(format: string, ...param: unknown[]): void;
}

/** Everything a recorder needs; the filesystem and clock are injectable. */
export interface RecorderDeps {
  files: Pick<KbFiles, "errors" | "archive" | "lock">;
  logger: RecorderLogger;
  options?: Partial<RecorderOptions>;
  fs?: StoreFs;
  clock?: StoreClock;
}

/** What one captured error led to. */
export type RecordOutcome =
  | { kind: "appended"; id: string }
  | { kind: "hit"; id: string; hits: number }
  /** `count-only` with no entry to count against: nothing written. */
  | { kind: "counted" }
  /** The lock stayed busy for the whole write budget: skipped silently. */
  | { kind: "timeout" }
  /** The write failed; the failure was counted and maybe logged. */
  | { kind: "failed" };

/** An outcome of a write that did not fail. */
type Written = Exclude<RecordOutcome, { kind: "failed" }>;

/** Running totals since the recorder was created. */
export interface RecorderStats {
  appended: number;
  hits: number;
  counted: number;
  timeouts: number;
  failures: number;
}

/** The `agent/error` payload fields the recorder reads. */
export interface AgentErrorPayload {
  agent?: { readonly id: string };
  error: unknown;
}

/** Listener bodies and their bookkeeping. */
export interface Recorder {
  /** Classify one error now and queue its write; never throws. */
  capture(input: CaptureInput, session?: string): void;
  /** The `agent/error` listener body; never throws. */
  agentError(payload: AgentErrorPayload): void;
  /** The `tools/result` listener body; never throws. */
  toolResult(
    exec: Readonly<ToolExecution>,
    result: Readonly<ToolExecutionResult>,
  ): void;
  /** Settles once every write queued so far has finished. */
  idle(): Promise<void>;
  /** Outcomes in the order their writes finished. */
  readonly outcomes: readonly RecordOutcome[];
  readonly stats: Readonly<RecorderStats>;
}

// ---------------------------------------------------------------------------
// Per-knowledge-base write chains

const chains = new Map<string, Promise<void>>();

/**
 * Run `task` after every task queued before it for the same knowledge base.
 * The task must not reject; the recorder's never do.
 */
function enqueue(key: string, task: () => Promise<void>): void {
  const next = (chains.get(key) ?? Promise.resolve()).then(task);
  chains.set(key, next);
  void next.then(() => {
    if (chains.get(key) === next) chains.delete(key);
  });
}

/**
 * Settles once every write queued so far for the knowledge base whose
 * `ERRORS.md` is at `errorsPath` has finished, whichever recorder queued it.
 */
export function settled(errorsPath: string): Promise<void> {
  return chains.get(errorsPath) ?? Promise.resolve();
}

// ---------------------------------------------------------------------------
// Reading hook payloads

/** The text of a tool result: its text blocks, one per line. */
export function resultText(content: readonly { type: string }[]): string {
  return content
    .filter((block): block is { type: "text"; text: string } => {
      return (
        block.type === "text" &&
        typeof (block as { text?: unknown }).text === "string"
      );
    })
    .map((block) => block.text)
    .join("\n");
}

// The argument names shell tools use for the command line. The shell tool is
// not among the installed packages, so its schema could not be checked; these
// cover the common spellings, and a call without any of them is still
// recorded, only without the command.
const COMMAND_KEYS = ["command", "cmd", "script"] as const;

/**
 * The command a tool call ran, from its parsed arguments: a string argument
 * as is, or the first of `command`, `cmd` and `script` that is a string or a
 * list of strings (joined with spaces).
 *
 * @param args - `exec.arguments`, deep-frozen JSON.
 * @returns the command, or `undefined` when the arguments name none.
 */
export function commandFrom(args: unknown): string | undefined {
  if (typeof args === "string") return args;
  if (typeof args !== "object" || args === null) return undefined;
  for (const key of COMMAND_KEYS) {
    const value = (args as Record<string, unknown>)[key];
    if (typeof value === "string") return value;
    if (Array.isArray(value) && value.every((v) => typeof v === "string"))
      return value.join(" ");
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// The recorder

/**
 * True for a failure a retry cannot fix: a document that does not parse stays
 * unparseable until someone repairs it (§13 keeps it append-only meanwhile).
 */
function final(error: unknown): boolean {
  return error instanceof StoreCorruptError || error instanceof ParseError;
}

/**
 * Bind the capture pipeline to one knowledge base.
 *
 * @param deps - where to write, what to log through, settings, and optionally
 *   the filesystem and clock.
 * @returns listener bodies that never throw.
 */
export function createRecorder(deps: RecorderDeps): Recorder {
  const o: RecorderOptions = { ...DEFAULT_RECORDER_OPTIONS, ...deps.options };
  const fs = deps.fs ?? nodeStoreFs();
  const clock = deps.clock ?? systemClock();
  const key = deps.files.errors;
  const counters = new Map<string, TransientCounter>();
  const outcomes: RecordOutcome[] = [];
  const stats: RecorderStats = {
    appended: 0,
    hits: 0,
    counted: 0,
    timeouts: 0,
    failures: 0,
  };
  let cache: { mtime: number | undefined; index: IndexedEntry[] } | undefined;
  let lastLog: number | undefined;

  const now = () => clock.now().getTime();

  /** Count a failure; log it unless one was logged less than a minute ago. */
  function fail(error: unknown): void {
    stats.failures++;
    try {
      const at = now();
      if (lastLog !== undefined && at - lastLog < FAILURE_LOG_INTERVAL_MS)
        return;
      lastLog = at;
      deps.logger.warn(
        "err-kb: an error was not recorded (%d so far): %s",
        stats.failures,
        safeErrorText(error).message,
      );
    } catch {
      // A logger that throws must not reach the turn either.
    }
  }

  /** This session's counter, kept among the most recently active ones. */
  function counterFor(session: string): TransientCounter {
    const counter = counters.get(session) ?? new TransientCounter();
    counters.delete(session);
    counters.set(session, counter);
    if (counters.size > MAX_SESSIONS)
      counters.delete(counters.keys().next().value as string);
    return counter;
  }

  /** The indexed entries, re-read only when ERRORS.md changed on disk. */
  async function currentIndex(store: ErrorStore): Promise<IndexedEntry[]> {
    const mtime = await fs.mtimeMs(key);
    if (cache !== undefined && cache.mtime === mtime) return cache.index;
    const document = await store.read();
    const index = indexEntries(document.blocks.map((block) => block.entry));
    cache = { mtime, index };
    return index;
  }

  /** Bump an entry's hits and last-seen; undefined when it is gone. */
  async function bump(
    store: ErrorStore,
    entry: Entry,
  ): Promise<Written | undefined> {
    const updated = await store.update(entry.id, {
      hits: entry.hits + 1,
      lastSeen: formatSeen(clock.now()),
    });
    cache = undefined;
    if (updated === undefined) return undefined;
    return { kind: "hit", id: updated.id, hits: updated.hits };
  }

  /** One attempt at writing a classified error. */
  async function once(
    store: ErrorStore,
    { decision, record }: Classified,
  ): Promise<Written> {
    const index = await currentIndex(store);
    if (decision === "count-only") {
      const known = index.find((i) => i.sig === record.signature);
      const outcome =
        known === undefined ? undefined : await bump(store, known.entry);
      return outcome ?? { kind: "counted" };
    }
    const found = match(
      {
        category: record.category,
        message: record.message,
        ...(record.code === undefined ? {} : { code: record.code }),
      },
      index,
      o,
    );
    if (found.matched) {
      const outcome = await bump(store, found.entry);
      if (outcome !== undefined) return outcome;
    }
    const { id } = await store.append({
      title: record.title,
      signature: record.signature,
      category: record.displayCategory,
      meta: {
        cat: record.category,
        ...(record.code === undefined ? {} : { code: record.code }),
      },
      raw: record.raw,
    });
    cache = undefined;
    return { kind: "appended", id };
  }

  /** Write one classified error within the time budget, with retries. */
  async function persist(classified: Classified): Promise<Written> {
    const deadline = now() + WRITE_TIMEOUT_MS;
    for (let attempt = 0; ; attempt++) {
      const store = createStore(
        deps.files,
        { ...o, lockTimeoutMs: Math.max(0, deadline - now()) },
        fs,
        clock,
      );
      try {
        return await once(store, classified);
      } catch (error) {
        cache = undefined;
        if (error instanceof LockTimeoutError) return { kind: "timeout" };
        if (final(error) || attempt >= WRITE_RETRIES) throw error;
        await clock.sleep(RETRY_DELAY_MS);
        if (now() >= deadline) return { kind: "timeout" };
      }
    }
  }

  function tally(outcome: Written): void {
    outcomes.push(outcome);
    if (outcome.kind === "appended") stats.appended++;
    else if (outcome.kind === "hit") stats.hits++;
    else if (outcome.kind === "counted") stats.counted++;
    else stats.timeouts++;
  }

  function capture(input: CaptureInput, session = NO_SESSION): void {
    try {
      const classified = classify(input, counterFor(session), o);
      if (classified === undefined) return;
      enqueue(key, async () => {
        try {
          tally(await persist(classified));
        } catch (error) {
          fail(error);
          outcomes.push({ kind: "failed" });
        }
      });
    } catch (error) {
      fail(error);
    }
  }

  return {
    capture,

    agentError(payload) {
      try {
        capture({ kind: "agent", error: payload.error }, payload.agent?.id);
      } catch (error) {
        fail(error);
      }
    },

    toolResult(exec, result) {
      try {
        const session = exec.agent?.id;
        if (result.isError) {
          const code = result.error.info?.code;
          capture(
            {
              kind: "tool",
              toolName: exec.name,
              isError: true,
              message: result.error.message,
              ...(code === undefined ? {} : { code }),
            },
            session,
          );
          return;
        }
        const command = commandFrom(exec.arguments);
        capture(
          {
            kind: "command",
            toolName: exec.name,
            text: resultText(result.content),
            ...(command === undefined ? {} : { command }),
          },
          session,
        );
      } catch (error) {
        fail(error);
      }
    },

    idle: () => settled(key),
    outcomes,
    stats,
  };
}

// ---------------------------------------------------------------------------
// Registration

/** The part of a cordis Context the wiring uses. */
export type ListenerHost = Pick<Context, "on">;

/**
 * Register the two T11 listeners. `agent/request-error` (T16) and the
 * injection points (T13) are deliberately absent.
 *
 * @param ctx - the plugin's context.
 * @param recorder - from createRecorder().
 */
export function registerListeners(ctx: ListenerHost, recorder: Recorder): void {
  ctx.on("agent/error", (payload) => {
    recorder.agentError(payload);
  });
  ctx.on("tools/result", (exec, result) => {
    recorder.toolResult(exec, result);
    return undefined;
  });
}
