// The capture pipeline behind the plugin's listeners (T11): a hook payload goes
// in, an entry in ERRORS.md comes out - or nothing, and never a throw. The
// injection points built on it (T13) are described where they start, under
// "Injection (T13)" below.
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
// A bump never touches ERRORS.md (§4.3): it counts in this machine's
// state.json (src/state.ts), so the block stays byte-identical and two devices
// that hit the same error merge cleanly in git. Every entry the recorder hands
// out - to match(), the notices, the digest and the tools - is the effective
// one: the block's hits plus this machine's, through effectiveEntry().
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
import type { Context, Events } from "@deepseek-ai/cordis";
import { createUserMessage } from "@deepseek-ai/dsh-llm";
import type { UserMessage } from "@deepseek-ai/dsh-llm";
import type {
  PostToolDecision,
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
import {
  CapTracker,
  FixTrust,
  Injector,
  estimateTokens,
  SYSTEM_PROMPT_HINT,
  SYSTEM_PROMPT_SECTION,
  noticeSource,
  sessionDigestText,
} from "./inject";
import type {
  CapLimits,
  InjectMode,
  Notice,
  NoticeKind,
  NoticeSource,
  SessionDigestMode,
  TrustStore,
} from "./inject";
import { DEFAULT_MATCH_OPTIONS, indexEntries, match } from "./match";
import { ResolutionTracker, callOutcome } from "./resolve-detect";
import type { CaptureFixMode } from "./resolve-detect";
import type { Hit, IndexedEntry, MatchOptions, MatchResult } from "./match";
import type { KbFiles } from "./paths";
import { addHit, createStateFile, effectiveEntry } from "./state";
import type { MachineState, StateFile } from "./state";
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
  files: Pick<KbFiles, "errors" | "archive" | "lock" | "state">;
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

/** What one recordFix() led to. */
export type FixOutcome =
  /** The entry now carries the fix, redacted, and is `fixed`. */
  | { kind: "fixed"; entry: Entry }
  /** No entry has that ID. */
  | { kind: "unknown" }
  /** The lock stayed busy for the whole write budget: nothing written. */
  | { kind: "timeout" }
  /** The write failed; the failure was counted and maybe logged. */
  | { kind: "failed" };

/** What one write() led to. */
export type WriteOutcome<T> =
  /** The write ran; `value` is what it returned. */
  | { kind: "done"; value: T }
  /** The lock stayed busy for the whole write budget: nothing written. */
  | { kind: "timeout" }
  /** The write failed; the failure was counted and maybe logged. */
  | { kind: "failed"; error: unknown };

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

/** Settles with a write's outcome; never rejects. */
export type PendingWrite = Promise<RecordOutcome>;

/** Listener bodies and their bookkeeping. */
export interface Recorder {
  /**
   * Classify one error now and queue its write; never throws.
   *
   * @returns the queued write, or undefined when nothing was captured.
   */
  capture(input: CaptureInput, session?: string): PendingWrite | undefined;
  /** The `agent/error` listener body; never throws. */
  agentError(payload: AgentErrorPayload): PendingWrite | undefined;
  /** The `tools/result` listener body; never throws. */
  toolResult(
    exec: Readonly<ToolExecution>,
    result: Readonly<ToolExecutionResult>,
  ): PendingWrite | undefined;
  /**
   * Match one error against the knowledge base without writing anything (the
   * T13 hot path). The read is queued behind every write queued before it, so
   * an error recorded in an earlier step is already there to hit, and it runs
   * before any write queued after it, so an error never hits its own entry.
   * Classification is the pure part of capture: tool, command and turn errors
   * touch no counter, and an LLM error here does not count towards its
   * promotion.
   *
   * @returns the match, or undefined when the error is not captured at all or
   *   the read failed (counted like a failed write).
   */
  lookup(input: CaptureInput): Promise<MatchResult | undefined>;
  /** The indexed entries, read like lookup(); undefined when the read failed. */
  entries(): Promise<IndexedEntry[] | undefined>;
  /**
   * Write a fix into an entry and mark it `fixed` (T14), for `err_record`
   * (T15) to call. Off the turn and serialized with the other writes, under
   * the same budget and retries; the store redacts the fix. Never rejects.
   */
  recordFix(id: string, fix: string): Promise<FixOutcome>;
  /**
   * Run any store operation on the same path as recordFix(): queued behind
   * every earlier write, under the same budget and retries (T15's other
   * writes: status and notes, a new entry, an archived one). `state` is this
   * machine's state.json, under the same budget. Never rejects.
   */
  write<T>(
    task: (store: ErrorStore, state: StateFile) => Promise<T>,
  ): Promise<WriteOutcome<T>>;
  /**
   * This machine's state.json, read like entries(); a corrupt file reads as
   * empty. Undefined when the read failed.
   */
  machineState(): Promise<MachineState | undefined>;
  /** Count a failure and log it, at most once a minute; never throws. */
  fail(error: unknown): void;
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
 * Queue `task` like enqueue() and settle with its result. A rejection of
 * `task` settles the returned promise, so the chain itself never rejects and
 * the tasks after it still run.
 */
function queued<T>(key: string, task: () => Promise<T>): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    enqueue(key, () => task().then(resolve, reject));
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

/**
 * The capture input for one tool outcome: a failure as a tool error, anything
 * else as command output, which classify() keeps only when it reports a
 * non-zero exit code.
 *
 * @param exec - the call, for its tool name and arguments.
 * @param result - the outcome.
 */
export function toolInput(
  exec: Readonly<ToolExecution>,
  result: Readonly<ToolExecutionResult>,
): CaptureInput {
  if (result.isError) {
    const code = result.error.info?.code;
    return {
      kind: "tool",
      toolName: exec.name,
      isError: true,
      message: result.error.message,
      ...(code === undefined ? {} : { code }),
    };
  }
  const command = commandFrom(exec.arguments);
  return {
    kind: "command",
    toolName: exec.name,
    text: resultText(result.content),
    ...(command === undefined ? {} : { command }),
  };
}

// ---------------------------------------------------------------------------
// The recorder

/** The indexed effective entries, and the baselines they were built on. */
interface Snapshot {
  /** ERRORS.md's and state.json's mtimes when this was read. */
  mtime: number | undefined;
  stateMtime: number | undefined;
  index: IndexedEntry[];
  /** Each entry as written in ERRORS.md, by ID. */
  baseline: Map<string, Entry>;
}

/** What budgeted() returns when the write budget ran out. */
const TIMEOUT = Symbol("timeout");

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
  const stateFile = deps.files.state;
  const counters = new Map<string, TransientCounter>();
  const outcomes: RecordOutcome[] = [];
  const stats: RecorderStats = {
    appended: 0,
    hits: 0,
    counted: 0,
    timeouts: 0,
    failures: 0,
  };
  let cache: Snapshot | undefined;
  let lastLog: number | undefined;
  // A corrupt state.json is reported once, until a write saves it aside.
  let reportedCorrupt = false;

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

  /** This machine's state; a corrupt file reads as empty and is reported once. */
  async function readState(state: StateFile): Promise<MachineState> {
    const read = await state.read();
    if (read.corrupt && !reportedCorrupt) {
      reportedCorrupt = true;
      fail(
        new Error(
          "state.json is not valid; hit counts fall back to ERRORS.md until the next write saves it aside",
        ),
      );
    }
    return read.state;
  }

  /**
   * The indexed effective entries, re-read only when ERRORS.md or state.json
   * changed on disk.
   */
  async function currentIndex(
    store: ErrorStore,
    state: StateFile,
  ): Promise<Snapshot> {
    const mtime = await fs.mtimeMs(key);
    const stateMtime = await fs.mtimeMs(stateFile);
    if (
      cache !== undefined &&
      cache.mtime === mtime &&
      cache.stateMtime === stateMtime
    )
      return cache;
    const document = await store.read();
    const machine = await readState(state);
    const baseline = new Map<string, Entry>();
    const index = indexEntries(
      document.blocks.map(({ entry }) => {
        baseline.set(entry.id, entry);
        return effectiveEntry(entry, machine.entries[entry.id]);
      }),
    );
    cache = { mtime, stateMtime, index, baseline };
    return cache;
  }

  /** Match a classified record against the index. */
  function matchRecord(
    record: Classified["record"],
    index: readonly IndexedEntry[],
  ): MatchResult {
    return match(
      {
        category: record.category,
        message: record.message,
        ...(record.code === undefined ? {} : { code: record.code }),
      },
      index,
      o,
    );
  }

  /**
   * Bump an entry's hits and last-seen in state.json; ERRORS.md is not
   * touched (§4.3).
   *
   * @returns the hit, with the effective count after it.
   */
  async function bump(
    state: StateFile,
    snapshot: Snapshot,
    id: string,
  ): Promise<Written> {
    const at = formatSeen(clock.now());
    const written = await state.update((machine) => {
      addHit(machine, id, at);
    });
    if (written.savedAs !== undefined) reportedCorrupt = false;
    cache = undefined;
    const entry = effectiveEntry(
      snapshot.baseline.get(id) as Entry,
      written.state.entries[id],
    );
    return { kind: "hit", id, hits: entry.hits };
  }

  /** One attempt at writing a classified error. */
  async function once(
    store: ErrorStore,
    state: StateFile,
    { decision, record }: Classified,
  ): Promise<Written> {
    const snapshot = await currentIndex(store, state);
    const { index } = snapshot;
    if (decision === "count-only") {
      const known = index.find((i) => i.sig === record.signature);
      return known === undefined
        ? { kind: "counted" }
        : bump(state, snapshot, known.entry.id);
    }
    const found = matchRecord(record, index);
    if (found.matched) return bump(state, snapshot, found.id);
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

  /**
   * Run one write within the time budget, with retries.
   *
   * @returns what `write` returned, or `TIMEOUT` when the budget ran out.
   */
  async function budgeted<T>(
    write: (store: ErrorStore, state: StateFile) => Promise<T>,
  ): Promise<T | typeof TIMEOUT> {
    const deadline = now() + WRITE_TIMEOUT_MS;
    for (let attempt = 0; ; attempt++) {
      // One budget for both files: each lock wait gets what is left of it.
      const lockTimeoutMs = () => Math.max(0, deadline - now());
      const store = createStore(
        deps.files,
        { ...o, lockTimeoutMs: lockTimeoutMs() },
        fs,
        clock,
      );
      const state: StateFile = {
        read: () => createStateFile(deps.files, {}, fs, clock).read(),
        update: (mutate) =>
          createStateFile(
            deps.files,
            { lockTimeoutMs: lockTimeoutMs() },
            fs,
            clock,
          ).update(mutate),
      };
      try {
        return await write(store, state);
      } catch (error) {
        cache = undefined;
        if (error instanceof LockTimeoutError) return TIMEOUT;
        if (final(error) || attempt >= WRITE_RETRIES) throw error;
        await clock.sleep(RETRY_DELAY_MS);
        if (now() >= deadline) return TIMEOUT;
      }
    }
  }

  /** Write one classified error within the time budget, with retries. */
  async function persist(classified: Classified): Promise<Written> {
    const outcome = await budgeted((store, state) =>
      once(store, state, classified),
    );
    return outcome === TIMEOUT ? { kind: "timeout" } : outcome;
  }

  function tally(outcome: Written): void {
    outcomes.push(outcome);
    if (outcome.kind === "appended") stats.appended++;
    else if (outcome.kind === "hit") stats.hits++;
    else if (outcome.kind === "counted") stats.counted++;
    else stats.timeouts++;
  }

  function capture(
    input: CaptureInput,
    session = NO_SESSION,
  ): PendingWrite | undefined {
    try {
      const classified = classify(input, counterFor(session), o);
      if (classified === undefined) return undefined;
      return queued(key, async (): Promise<RecordOutcome> => {
        try {
          const outcome = await persist(classified);
          tally(outcome);
          return outcome;
        } catch (error) {
          fail(error);
          const outcome: RecordOutcome = { kind: "failed" };
          outcomes.push(outcome);
          return outcome;
        }
      });
    } catch (error) {
      fail(error);
      return undefined;
    }
  }

  /** Read the current index on the write chain; undefined on failure. */
  function read(): Promise<IndexedEntry[] | undefined> {
    return queued(key, async () => {
      try {
        const snapshot = await currentIndex(
          createStore(deps.files, o, fs, clock),
          createStateFile(deps.files, {}, fs, clock),
        );
        return snapshot.index;
      } catch (error) {
        cache = undefined;
        fail(error);
        return undefined;
      }
    });
  }

  function write<T>(
    task: (store: ErrorStore, state: StateFile) => Promise<T>,
  ): Promise<WriteOutcome<T>> {
    return queued(key, async (): Promise<WriteOutcome<T>> => {
      try {
        const value = await budgeted(task);
        cache = undefined;
        return value === TIMEOUT
          ? { kind: "timeout" }
          : { kind: "done", value };
      } catch (error) {
        fail(error);
        return { kind: "failed", error };
      }
    });
  }

  return {
    capture,

    agentError(payload) {
      try {
        return capture(
          { kind: "agent", error: payload.error },
          payload.agent?.id,
        );
      } catch (error) {
        fail(error);
        return undefined;
      }
    },

    toolResult(exec, result) {
      try {
        return capture(toolInput(exec, result), exec.agent?.id);
      } catch (error) {
        fail(error);
        return undefined;
      }
    },

    async lookup(input) {
      try {
        const classified = classify(input, new TransientCounter(), o);
        if (classified === undefined) return undefined;
        const index = await read();
        return index === undefined
          ? undefined
          : matchRecord(classified.record, index);
      } catch (error) {
        fail(error);
        return undefined;
      }
    },

    entries: read,

    machineState() {
      return queued(key, async () => {
        try {
          return await readState(createStateFile(deps.files, {}, fs, clock));
        } catch (error) {
          fail(error);
          return undefined;
        }
      });
    },

    write,

    async recordFix(id, fix) {
      const outcome = await write((store) =>
        store.update(id, { fix, status: "fixed" }),
      );
      if (outcome.kind !== "done") return { kind: outcome.kind };
      return outcome.value === undefined
        ? { kind: "unknown" }
        : { kind: "fixed", entry: outcome.value };
    },

    fail,
    idle: () => settled(key),
    outcomes,
    stats,
  };
}

// ---------------------------------------------------------------------------
// Injection (T13)
//
// Four points hand what the knowledge base knows back to the model (§7). Their
// types are read off the installed packages, through cordis `Events` as those
// packages augment it:
//
// - `tools/post-execute` - a waterfall in @deepseek-ai/dsh-tools
//   (lib/types/index.d.ts): `(exec: ToolExecution, result:
//   Readonly<ToolExecutionResult>, next: () => Promise<PostToolDecision>)`.
//   `PostToolDecision` is accept / accept-with-value / block, each with
//   `additionalContexts?: UserMessage[]`, which reach the next request.
// - `agent/pre-step` - a waterfall in @deepseek-ai/dsh-agent
//   (lib/types/runtime-types.d.ts): `({agent, messages, turn, step, signal},
//   next: () => Promise<PreStepDecision>)`, where `PreStepDecision` is
//   `{kind: 'reject'} | {kind: 'enter', messages: UserMessage[]}`.
// - `agent/session-start` - an emit in the same file: `({agent, source})`;
//   `Agent.inject(message: UserMessage)` queues context for the next pre-step.
// - `ctx.systemPrompt.section(section: PromptSection)` - `SystemPrompt` in
//   @deepseek-ai/dsh-system-prompt (lib/types/index.d.ts), `{name, order,
//   text}`; it throws on a duplicate name.
//
// Messages are built with dsh-llm's `createUserMessage` (lib/types/message.d.ts)
// from one `TextBlock` and the notice's `MessageSourceMap['plugin']` source.
//
// Both waterfalls `await next()` first and return what it returned, with only
// our message appended: `additionalContexts` or `messages` grow, nothing else
// changes, and with nothing to add the downstream object itself comes back. A
// failure of ours returns it unchanged; a rejection from `next()` is not ours
// and passes through.
//
// The hot path. A tool failure is matched inside `tools/post-execute`: the
// recorder's lookup() classifies it (pure for tool and command input) and reads
// the cached index, queued behind earlier writes so last step's new entry is
// there to hit. The write itself still happens later, off the turn, from
// `tools/result`. A miss under `inject: 'always'` therefore has no ID inside
// the hook; instead of guessing one or saying "pending", that notice waits for
// the write and rides the next `agent/pre-step`, when `recorded as E-00NN`
// can name the real ID. If the write has not settled by then it waits for the
// step after; if it fails, nothing is said.
//
// A turn error (`agent/error`) kills its turn, so its notice cannot ride
// `additionalContexts`. It is looked up as it is captured - the read queued
// ahead of its own write - and offered when the next step opens, appended to
// that step's entry messages (§7). LLM request failures reach this path only
// when they end the turn as an `agent/error`; `agent/request-error` is T16.
//
// Per-session state, keyed by `Agent.id` and bounded to the {@link
// MAX_SESSIONS} most recently active sessions like the transient counters:
// one Injector, so one CapTracker, and the pending pre-step notices. The
// boundaries come from `agent/pre-step`, the only hook that carries both
// numbers: a `turn` different from the last one seen calls beginTurn(), and
// every pre-step calls beginStep(). The tool calls of a step run after its
// pre-step, so their notices count against that step. `agent/session-start`
// starts a session's state afresh, since `clear` and `compact` begin a new
// lifecycle on the same agent. A tool call without `exec.agent` has nobody to
// read its context and is never injected.
//
// Fix trust is one FixTrust per plugin instance over an injectable TrustStore
// (state.json's `trust` by default, see stateTrustStore()), scoped per
// session, so a recurrence is counted only within the session and turn that
// injected the fix.
//
// Resolution detection (T14) rides `tools/result`. Each session keeps a
// ResolutionTracker (src/resolve-detect.ts, where the rule and the window are
// described), advanced by the same turn boundary as the caps. A failed call
// is watched once its write names the entry; a successful one is checked
// against the watches. Both are handled in arrival order on a per-session
// chain, so a success is never checked before the failure in front of it has
// been written. A resolved entry feeds FixTrust a success; one without a fix
// queues the one-shot `captureFix` prompt for the next pre-step, through the
// same pending list as a dead turn's notice. recordFix() on the recorder is
// the write `err_record` (T15) calls.

/** Pending pre-step notices kept per session; the oldest goes first. */
export const MAX_PENDING = 4;

/** Injection settings; every one has a default. */
export interface InjectionOptions {
  inject: InjectMode;
  captureFix: CaptureFixMode;
  sessionDigest: SessionDigestMode;
  systemPromptHint: boolean;
}

export const DEFAULT_INJECTION_OPTIONS: InjectionOptions = {
  inject: "hit-only",
  captureFix: "prompt-once",
  sessionDigest: "counts",
  systemPromptHint: true,
};

/** Everything the injection layer needs. */
export interface InjectionDeps {
  recorder: Recorder;
  options?: Partial<InjectionOptions>;
  /** Where fix trust lives; state.json by default. */
  trust?: TrustStore;
  /** Lower notice budgets, for every session. */
  caps?: Partial<CapLimits>;
}

/** `agent/pre-step`'s payload, as dsh-agent declares it. */
export type PreStepPayload = Parameters<Events["agent/pre-step"]>[0];

/** `PreStepDecision` from dsh-agent, through the event's return type. */
export type PreStepDecision = Awaited<ReturnType<Events["agent/pre-step"]>>;

/** `agent/session-start`'s payload, as dsh-agent declares it. */
export type SessionStartPayload = Parameters<Events["agent/session-start"]>[0];

/** `PromptSection` from dsh-system-prompt, through `SystemPrompt.section`. */
export type PromptSection = Parameters<Context["systemPrompt"]["section"]>[0];

/** The listener bodies of the four injection points. */
export interface Injection {
  readonly options: InjectionOptions;
  readonly trust: FixTrust;
  /** `agent/error`: record, and queue the notice for the next step. */
  agentError(payload: AgentErrorPayload): void;
  /**
   * `tools/result`: record, route a miss's new ID to the next step, and feed
   * resolution detection (T14).
   */
  toolResult(
    exec: Readonly<ToolExecution>,
    result: Readonly<ToolExecutionResult>,
  ): void;
  /** `tools/post-execute`: the downstream decision plus a notice. */
  postExecute(
    exec: Readonly<ToolExecution>,
    result: Readonly<ToolExecutionResult>,
    next: () => Promise<PostToolDecision>,
  ): Promise<PostToolDecision>;
  /** `agent/pre-step`: the downstream decision plus a pending notice. */
  preStep(
    payload: PreStepPayload,
    next: () => Promise<PreStepDecision>,
  ): Promise<PreStepDecision>;
  /** `agent/session-start`: fresh session state and the digest. */
  sessionStart(payload: SessionStartPayload): Promise<void>;
  /** The system-prompt section, or undefined under `systemPromptHint: false`. */
  section(): PromptSection | undefined;
  /**
   * Notices delivered so far (T15's `err_stats`): one session's since it
   * started, or every session's since the plugin started.
   */
  counts(session?: string): InjectionCounts;
}

/** Notices delivered, for `err_stats` (T15). */
export interface InjectionCounts {
  /** Every notice that reached the model. */
  notices: number;
  /** The notices that carried a recorded fix: `hit`, `near` or `doubted`. */
  fixNotices: number;
  /** Their bodies' size, by estimateTokens(). */
  noticeTokens: number;
}

const emptyCounts = (): InjectionCounts => ({
  notices: 0,
  fixNotices: 0,
  noticeTokens: 0,
});

const FIX_NOTICES: readonly NoticeKind[] = ["hit", "near", "doubted"];

/** One notice waiting for the next step. */
interface Pending {
  /** The lookup; a read, so the next step waits for it. */
  read: Promise<void>;
  hit?: Hit;
  /** The lookup missed and `inject: 'always'` wants the new ID announced. */
  miss: boolean;
  /** The write settled; `id` is set when it appended an entry. */
  written: boolean;
  id?: string;
  /** The fix prompt for this entry, which looks resolved (T14). */
  ask?: string;
  /** The resolver's turn the prompt was queued in; it lapses with the window. */
  turn?: number;
}

interface SessionState {
  injector: Injector;
  resolver: ResolutionTracker;
  turn: number | undefined;
  pending: Pending[];
  /**
   * Resolution events, handled one at a time in arrival order: a failure
   * waits for its write to name the entry before a later success is checked
   * against it. Never rejects.
   */
  resolving: Promise<void>;
  /** What this session was told, for `err_stats` (T15). */
  counts: InjectionCounts;
}

/** A user message carrying `text`, attributed to the plugin. */
function pluginMessage(text: string, source: NoticeSource): UserMessage {
  return createUserMessage({ content: [{ type: "text", text }], source });
}

/**
 * A hit whose hit count includes the occurrence being reported: the store is
 * about to add it, and "known (1 hit)" for the second sighting would undersell.
 */
function counted(hit: Hit): Hit {
  return { ...hit, entry: { ...hit.entry, hits: hit.entry.hits + 1 } };
}

/**
 * Fix trust kept in state.json under `trust` (§4.3, T12): machine-local, so
 * a fix suppressed here stays suppressed after a restart and on no other
 * device. The saved records are read once through the recorder and join the
 * in-memory state through FixTrust's `ready`. Every save queues a write of
 * this process's records over what is on disk, on the recorder's write chain
 * and under its lock and budget: records of IDs this process never touched -
 * another harness's on the same machine - are kept, and for an ID both
 * touched the later save wins. A failed or timed-out save is counted like any
 * write, and the next save writes the full set again.
 *
 * @param recorder - the recorder whose knowledge base holds state.json.
 */
export function stateTrustStore(recorder: Recorder): TrustStore {
  return {
    load: () => ({ entries: {} }),
    loaded: recorder
      .machineState()
      .then((state) =>
        state === undefined ? undefined : { entries: state.trust },
      ),
    save(state) {
      const entries = structuredClone(state.entries);
      void recorder.write((_store, file) =>
        file.update((machine) => {
          Object.assign(machine.trust, entries);
        }),
      );
    },
  };
}

/**
 * Bind the four injection points to a recorder.
 *
 * @param deps - the recorder, settings, trust store and caps.
 * @returns listener bodies that never throw into a turn.
 */
export function createInjection(deps: InjectionDeps): Injection {
  const o: InjectionOptions = { ...DEFAULT_INJECTION_OPTIONS, ...deps.options };
  const { recorder } = deps;
  const trust = new FixTrust(deps.trust ?? stateTrustStore(recorder));
  const sessions = new Map<string, SessionState>();
  // A tool miss whose write will name the ID, keyed by the execution object
  // that `tools/post-execute` and `tools/result` both receive (the registry
  // freezes it in place before `tools/result`; identity is kept).
  const awaitingId = new WeakMap<object, SessionState>();
  const injecting = o.inject !== "off";
  const totals = emptyCounts();

  /** The message for a notice about to reach the model, counted. */
  function deliver(state: SessionState, notice: Notice): UserMessage {
    const tokens = estimateTokens(notice.text);
    const fix = FIX_NOTICES.includes(notice.kind) ? 1 : 0;
    for (const counts of [state.counts, totals]) {
      counts.notices++;
      counts.fixNotices += fix;
      counts.noticeTokens += tokens;
    }
    return pluginMessage(notice.text, notice.source);
  }

  /** This session's state, kept among the most recently active ones. */
  function stateFor(session: string, fresh = false): SessionState {
    const state = (!fresh && sessions.get(session)) || {
      injector: new Injector({
        mode: o.inject,
        caps: new CapTracker(deps.caps),
        trust,
        scope: session,
      }),
      resolver: new ResolutionTracker(),
      turn: undefined,
      pending: [],
      resolving: Promise.resolve(),
      counts: emptyCounts(),
    };
    sessions.delete(session);
    sessions.set(session, state);
    if (sessions.size > MAX_SESSIONS)
      sessions.delete(sessions.keys().next().value as string);
    return state;
  }

  /**
   * `start()`, with a handler attached before anything else can run: a throw
   * or a rejection is counted and settles as undefined. Every promise the
   * hooks start and do not await at once goes through here, so none is ever
   * left to reject unhandled while `next()` runs (§13) - the Recorder
   * contract says lookup() and the writes never reject, but a recorder that
   * breaks it must not crash the host.
   */
  function guard<T>(start: () => Promise<T>): Promise<T | undefined> {
    try {
      return start().catch((error: unknown) => {
        recorder.fail(error);
        return undefined;
      });
    } catch (error) {
      recorder.fail(error);
      return Promise.resolve(undefined);
    }
  }

  /** Queue a notice for the next step; never rejects. */
  function track(
    state: SessionState,
    item: Pending,
    write: PendingWrite | undefined,
  ): void {
    if (write !== undefined)
      void guard(() => write).then((outcome) => {
        item.written = true;
        if (outcome?.kind === "appended") item.id = outcome.id;
      });
    else item.written = true;
    state.pending.push(item);
    if (state.pending.length > MAX_PENDING) state.pending.shift();
  }

  /**
   * Look a tool outcome up and offer it; never rejects. Reading the payload
   * is inside the try: a malformed result (`isError` with no `error`) throws
   * there, and this runs while `next()` is still pending.
   */
  async function toolNotice(
    exec: Readonly<ToolExecution>,
    result: Readonly<ToolExecutionResult>,
    state: SessionState,
  ): Promise<Notice | undefined> {
    try {
      const found = await recorder.lookup(toolInput(exec, result));
      if (found === undefined) return undefined;
      if (found.matched)
        return state.injector.offer({ kind: "hit", hit: counted(found) });
      if (o.inject === "always") awaitingId.set(exec, state);
      return undefined;
    } catch (error) {
      recorder.fail(error);
      return undefined;
    }
  }

  /** Queue a resolution step behind the session's earlier ones. */
  function resolving(state: SessionState, step: () => Promise<void>): void {
    state.resolving = state.resolving.then(() => guard(step)).then(() => {});
  }

  /**
   * Feed one tool outcome to resolution detection (T14). A failure is watched
   * once its write names the entry; a success resolves what it matches. The
   * turn is taken now, when the call happened.
   */
  function detect(
    state: SessionState,
    exec: Readonly<ToolExecution>,
    result: Readonly<ToolExecutionResult>,
    write: PendingWrite | undefined,
  ): void {
    const input = toolInput(exec, result);
    const outcome = callOutcome({
      toolName: exec.name,
      isError: input.kind === "tool",
      text: input.kind === "command" ? input.text : "",
      ...(input.kind === "command" && input.command !== undefined
        ? { command: input.command }
        : {}),
    });
    const turn = state.resolver.turn;
    if (!outcome.ok) {
      if (write === undefined) return;
      resolving(state, async () => {
        const written = await write;
        if (written.kind === "appended" || written.kind === "hit")
          state.resolver.occurred(written.id, outcome.key, turn);
      });
      return;
    }
    resolving(state, async () => {
      for (const id of state.resolver.succeeded(outcome.keys, turn))
        await resolved(state, id, turn);
    });
  }

  /**
   * `id` looks resolved: its fix earned a success, and an entry without one
   * gets the one-shot prompt on the next step, under `captureFix:
   * 'prompt-once'`.
   */
  async function resolved(
    state: SessionState,
    id: string,
    turn: number,
  ): Promise<void> {
    const index = await recorder.entries();
    const found = index?.find((indexed) => indexed.entry.id === id);
    if (found === undefined) return;
    const { entry, excluded } = found;
    trust.succeeded(id, entry.fix);
    if (
      !injecting ||
      o.captureFix !== "prompt-once" ||
      excluded !== undefined ||
      entry.fix.trim() !== "" ||
      !state.resolver.ask(id)
    )
      return;
    track(
      state,
      { read: Promise.resolve(), miss: false, written: true, ask: id, turn },
      undefined,
    );
  }

  /** The first pending notice the caps allow; the rest are dropped. */
  async function drain(state: SessionState): Promise<Notice | undefined> {
    const items = state.pending.slice();
    await Promise.all(items.map((item) => item.read));
    const kept: Pending[] = [];
    let notice: Notice | undefined;
    for (const item of items) {
      if (item.hit !== undefined) {
        notice ??= state.injector.offer({ kind: "hit", hit: item.hit }, true);
      } else if (item.miss && !item.written) {
        kept.push(item);
      } else if (item.id !== undefined) {
        notice ??= state.injector.offer({ kind: "miss", id: item.id });
      } else if (
        item.ask !== undefined &&
        state.resolver.open(item.turn as number)
      ) {
        // A prompt the caps refuse now waits for a later step, until the
        // window it was asked in closes; then it is dropped unsaid.
        const asked =
          notice === undefined ? state.injector.ask(item.ask) : undefined;
        if (asked !== undefined) notice = asked;
        else kept.push(item);
      }
    }
    state.pending = state.pending.filter(
      (item) => !items.includes(item) || kept.includes(item),
    );
    return notice;
  }

  return {
    options: o,
    trust,

    agentError(payload) {
      try {
        const session = payload.agent?.id;
        if (!injecting || session === undefined) {
          recorder.agentError(payload);
          return;
        }
        const state = stateFor(session);
        // Queued before the write, so the error cannot hit its own new entry.
        const found = guard(() =>
          recorder.lookup({ kind: "agent", error: payload.error }),
        );
        const write = recorder.agentError(payload);
        const item: Pending = {
          read: Promise.resolve(),
          miss: false,
          written: false,
        };
        item.read = found.then((result) => {
          try {
            if (result === undefined) return;
            if (!result.matched) {
              item.miss = o.inject === "always";
              return;
            }
            state.injector.observe(result);
            item.hit = counted(result);
          } catch (error) {
            recorder.fail(error);
          }
        });
        track(state, item, write);
      } catch (error) {
        recorder.fail(error);
      }
    },

    toolResult(exec, result) {
      try {
        const write = recorder.toolResult(exec, result);
        const state = awaitingId.get(exec);
        if (state !== undefined) {
          awaitingId.delete(exec);
          track(
            state,
            { read: Promise.resolve(), miss: true, written: false },
            write,
          );
        }
        const session = exec.agent?.id;
        if (session !== undefined)
          detect(stateFor(session), exec, result, write);
      } catch (error) {
        recorder.fail(error);
      }
    },

    async postExecute(exec, result, next) {
      let pending: Promise<Notice | undefined> | undefined;
      let state: SessionState | undefined;
      try {
        const session = exec.agent?.id;
        if (injecting && session !== undefined) {
          state = stateFor(session);
          const current = state;
          // Guarded as it is created, not when it is awaited after next().
          pending = guard(() => toolNotice(exec, result, current));
        }
      } catch (error) {
        recorder.fail(error);
      }
      const decision = await next();
      if (pending === undefined || state === undefined) return decision;
      try {
        const notice = await pending;
        if (notice === undefined) return decision;
        return {
          ...decision,
          additionalContexts: [
            ...(decision.additionalContexts ?? []),
            deliver(state, notice),
          ],
        };
      } catch (error) {
        recorder.fail(error);
        return decision;
      }
    },

    async preStep(payload, next) {
      let state: SessionState | undefined;
      try {
        state = stateFor(payload.agent.id);
        if (state.turn !== payload.turn) {
          state.turn = payload.turn;
          state.injector.beginTurn();
          state.resolver.beginTurn();
        }
        state.injector.beginStep();
      } catch (error) {
        recorder.fail(error);
      }
      const decision = await next();
      if (
        state === undefined ||
        decision.kind !== "enter" ||
        state.pending.length === 0
      )
        return decision;
      try {
        const notice = await drain(state);
        if (notice === undefined) return decision;
        return {
          ...decision,
          messages: [...decision.messages, deliver(state, notice)],
        };
      } catch (error) {
        recorder.fail(error);
        return decision;
      }
    },

    async sessionStart({ agent }) {
      try {
        stateFor(agent.id, true);
        if (o.sessionDigest === "off") return;
        const index = await recorder.entries();
        if (index === undefined) return;
        const text = sessionDigestText(
          index.map(({ entry, excluded }) => ({
            id: entry.id,
            title: entry.title,
            hits: entry.hits,
            injectable: excluded === undefined,
          })),
          o.sessionDigest,
        );
        if (text === undefined) return;
        const head = text.split("\n", 1)[0] as string;
        agent.inject(pluginMessage(text, noticeSource(head)));
      } catch (error) {
        recorder.fail(error);
      }
    },

    counts(session) {
      if (session === undefined) return { ...totals };
      const state = sessions.get(session);
      return state === undefined ? emptyCounts() : { ...state.counts };
    },

    section() {
      return o.systemPromptHint
        ? { ...SYSTEM_PROMPT_SECTION, text: SYSTEM_PROMPT_HINT }
        : undefined;
    },
  };
}

// ---------------------------------------------------------------------------
// Registration

/** The part of a cordis Context the wiring uses. */
export type ListenerHost = Pick<Context, "on">;

/** The part of a cordis Context the injection points use. */
export type InjectionHost = Pick<Context, "on" | "systemPrompt">;

/**
 * Register the two capture listeners (T11). `agent/request-error` (T16) is
 * deliberately absent.
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

/**
 * Register the capture listeners routed through the injection layer, the
 * three injection hooks and the system-prompt section (T11 + T13). A section
 * that cannot be registered (a duplicate name) is counted and logged; the
 * hooks still work without it.
 *
 * @param ctx - the plugin's context.
 * @param injection - from createInjection().
 * @param recorder - the recorder the injection layer was built on.
 */
export function registerInjection(
  ctx: InjectionHost,
  injection: Injection,
  recorder: Recorder,
): void {
  ctx.on("agent/error", (payload) => {
    injection.agentError(payload);
  });
  ctx.on("tools/result", (exec, result) => {
    injection.toolResult(exec, result);
    return undefined;
  });
  ctx.on("tools/post-execute", (exec, result, next) =>
    injection.postExecute(exec, result, next),
  );
  ctx.on("agent/pre-step", (payload, next) => injection.preStep(payload, next));
  // The promise goes back to the host: one that awaits its emit listeners
  // gets the digest queued before the first step; one that does not still
  // gets it at the next pre-step after the read. It never rejects.
  ctx.on("agent/session-start", (payload) => injection.sessionStart(payload));
  const section = injection.section();
  if (section === undefined) return;
  try {
    ctx.systemPrompt.section(section);
  } catch (error) {
    recorder.fail(error);
  }
}
