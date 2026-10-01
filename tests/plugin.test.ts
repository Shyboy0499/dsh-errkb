import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Context } from "@deepseek-ai/cordis";
import type {
  ToolExecution,
  ToolExecutionResult,
} from "@deepseek-ai/dsh-tools";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Config, apply, recorderOptions } from "../src/index";
import { filesIn } from "../src/paths";
import {
  FAILURE_LOG_INTERVAL_MS,
  MAX_SESSIONS,
  RETRY_DELAY_MS,
  WRITE_RETRIES,
  commandFrom,
  createRecorder,
  registerListeners,
  resultText,
  settled,
} from "../src/plugin";
import type { RecorderDeps, RecorderLogger } from "../src/plugin";
import { nodeStoreFs, parseDocument } from "../src/store";
import type { StoreClock, StoreFs } from "../src/store";

// ---------------------------------------------------------------------------
// Fixtures

let dir: string;

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "errkb-plugin-"));
});

afterEach(async () => {
  await rm(dir, { recursive: true, force: true });
});

type Listener = (...args: unknown[]) => unknown;

/** A context that records registrations and lets a test fire events. */
function fakeCtx() {
  const listeners = new Map<string, Listener[]>();
  const logs: Array<{ level: string; args: unknown[] }> = [];
  const log =
    (level: string) =>
    (...args: unknown[]) => {
      logs.push({ level, args });
    };
  const sections: unknown[] = [];
  const ctx = {
    on(name: string, listener: Listener) {
      listeners.set(name, [...(listeners.get(name) ?? []), listener]);
      return () => true;
    },
    tools: {
      register: () => () => undefined,
    },
    systemPrompt: {
      section(section: unknown) {
        sections.push(section);
        return () => undefined;
      },
    },
    logger: {
      info: log("info"),
      warn: log("warn"),
      error: log("error"),
      debug: log("debug"),
    },
  };
  return {
    ctx: ctx as unknown as Context,
    listeners,
    logs,
    fire(name: string, ...args: unknown[]) {
      for (const listener of listeners.get(name) ?? []) listener(...args);
    },
  };
}

/** A clock whose time only moves when the code sleeps, or a test says so. */
function fakeClock(start = Date.UTC(2026, 9, 1, 8, 0, 0)) {
  let t = start;
  const clock: StoreClock = {
    now: () => new Date(t),
    sleep: async (ms) => {
      t += ms;
    },
    random: () => 0.5,
  };
  return {
    clock,
    advance(ms: number) {
      t += ms;
    },
  };
}

function quietLogger() {
  const warnings: unknown[][] = [];
  const logger: RecorderLogger = {
    warn: (...args) => {
      warnings.push(args);
    },
  };
  return { logger, warnings };
}

const exec = (
  name: string,
  args: unknown = {},
  agent: { id: string } | undefined = { id: "s1" },
) =>
  ({
    name,
    arguments: args,
    ...(agent === undefined ? {} : { agent }),
  }) as unknown as Readonly<ToolExecution>;

const ok = (text: string) =>
  ({
    isError: false,
    value: null,
    content: [{ type: "text", text }],
  }) as unknown as Readonly<ToolExecutionResult>;

const failed = (message: string, code?: string) =>
  ({
    isError: true,
    error: {
      message,
      ...(code === undefined ? {} : { info: { name: "Error", code } }),
    },
    content: [{ type: "text", text: message }],
  }) as unknown as Readonly<ToolExecutionResult>;

const TSC_A =
  "/srv/app/src/a.ts:12:5 - error TS2307: Cannot find module 'missing-pkg'.\n[exit code: 1]";
const TSC_B =
  "/srv/other/src/a.ts:40:9 - error TS2307: Cannot find module 'missing-pkg'.\n[exit code: 1]";

async function readEntries(path: string) {
  return parseDocument(await readFile(path, "utf8")).blocks.map((b) => b.entry);
}

function recorder(over: Partial<RecorderDeps> = {}) {
  const { logger, warnings } = quietLogger();
  const files = filesIn(dir);
  const rec = createRecorder({ files, logger, ...over });
  return { rec, files, warnings };
}

// ---------------------------------------------------------------------------
// apply(): registration and end-to-end

describe("apply: registration", () => {
  it("registers the T11 listeners and the T13 hooks, not T16's, and logs the directory", () => {
    const fake = fakeCtx();
    apply(fake.ctx, Config({ kbDir: dir }));
    expect([...fake.listeners.keys()].sort()).toEqual([
      "agent/error",
      "agent/pre-step",
      "agent/session-start",
      "tools/post-execute",
      "tools/result",
    ]);
    expect(fake.listeners.has("agent/request-error")).toBe(false);
    expect(fake.logs).toHaveLength(1);
    expect(fake.logs[0]?.level).toBe("info");
    expect(String(fake.logs[0]?.args[0])).toContain(dir);
  });

  it("the tools/result listener returns undefined, as the emit event declares", () => {
    const fake = fakeCtx();
    apply(fake.ctx, Config({ kbDir: dir }));
    const listener = fake.listeners.get("tools/result")?.[0] as Listener;
    expect(listener(exec("read"), ok("fine"))).toBeUndefined();
  });
});

describe("apply: end to end on a temporary knowledge base", () => {
  it("a guaranteed-failing command produces E-0001 in ERRORS.md", async () => {
    const fake = fakeCtx();
    apply(fake.ctx, Config({ kbDir: dir }));
    fake.fire("tools/result", exec("bash", { command: "pnpm tsc" }), ok(TSC_A));
    await settled(filesIn(dir).errors);

    const text = await readFile(filesIn(dir).errors, "utf8");
    expect(text).toMatch(/^## E-0001 · \[command-exit:bash\] /m);
    const [entry] = await readEntries(filesIn(dir).errors);
    expect(entry).toMatchObject({
      id: "E-0001",
      category: "command-exit / bash",
      hits: 1,
      meta: { cat: "command-exit", code: "TS2307" },
    });
    // The store redacts: the absolute path never reaches the document.
    expect(text).not.toContain("/srv/app");
  });

  it("the same failure with another path and line is still E-0001, hits 2", async () => {
    const fake = fakeCtx();
    apply(fake.ctx, Config({ kbDir: dir }));
    fake.fire("tools/result", exec("bash", { command: "pnpm tsc" }), ok(TSC_A));
    fake.fire("tools/result", exec("bash", { command: "pnpm tsc" }), ok(TSC_B));
    await settled(filesIn(dir).errors);

    const entries = await readEntries(filesIn(dir).errors);
    expect(entries).toHaveLength(1);
    expect(entries[0]).toMatchObject({ id: "E-0001", hits: 2 });
  });

  it("two different silent commands take two IDs", async () => {
    const fake = fakeCtx();
    apply(fake.ctx, Config({ kbDir: dir }));
    fake.fire(
      "tools/result",
      exec("bash", { command: "pnpm test" }),
      ok("[exit code: 1]"),
    );
    fake.fire(
      "tools/result",
      exec("bash", { command: "pnpm lint" }),
      ok("[exit code: 1]"),
    );
    await settled(filesIn(dir).errors);

    const entries = await readEntries(filesIn(dir).errors);
    expect(entries.map((e) => [e.id, e.title])).toEqual([
      ["E-0001", "[command-exit:bash] pnpm test → exit code 1"],
      ["E-0002", "[command-exit:bash] pnpm lint → exit code 1"],
    ]);
  });

  it("tool failures and turn errors are recorded too", async () => {
    const fake = fakeCtx();
    apply(fake.ctx, Config({ kbDir: dir }));
    fake.fire(
      "tools/result",
      exec("write"),
      failed("EPERM: operation not permitted", "EPERM"),
    );
    fake.fire("agent/error", {
      agent: { id: "s1" },
      turn: 1,
      step: 2,
      error: new TypeError("x is not a function"),
    });
    await settled(filesIn(dir).errors);

    const entries = await readEntries(filesIn(dir).errors);
    expect(
      entries.map((e) => [e.id, e.title, e.meta.cat, e.meta.code]),
    ).toEqual([
      [
        "E-0001",
        "[tool:write] EPERM: operation not permitted",
        "tool",
        "EPERM",
      ],
      [
        "E-0002",
        "[agent] TypeError: x is not a function",
        "agent",
        "TypeError",
      ],
    ]);
  });

  it("a store that throws is swallowed: nothing reaches the turn, one warning", async () => {
    // ERRORS.md is a directory: every read fails with EISDIR.
    await mkdir(filesIn(dir).errors);
    const fake = fakeCtx();
    apply(fake.ctx, Config({ kbDir: dir }));
    expect(() => {
      fake.fire("tools/result", exec("bash"), ok("boom\n[exit code: 1]"));
      fake.fire("agent/error", { agent: { id: "s1" }, error: new Error("x") });
    }).not.toThrow();
    await settled(filesIn(dir).errors);

    const warnings = fake.logs.filter((l) => l.level === "warn");
    expect(warnings).toHaveLength(1);
    expect(String(warnings[0]?.args[0])).toContain("not recorded");
  });

  it("capture: [] writes nothing", async () => {
    const fake = fakeCtx();
    apply(fake.ctx, Config({ kbDir: dir, capture: [] }));
    fake.fire("tools/result", exec("bash"), ok("boom\n[exit code: 1]"));
    fake.fire("tools/result", exec("write"), failed("EPERM"));
    fake.fire("agent/error", { agent: { id: "s1" }, error: new Error("x") });
    await settled(filesIn(dir).errors);
    await expect(readFile(filesIn(dir).errors, "utf8")).rejects.toThrow();
  });

  it("settings reach the store: idPrefix, idWidth and labels", async () => {
    const fake = fakeCtx();
    apply(
      fake.ctx,
      Config({ kbDir: dir, idPrefix: "ERR-", idWidth: 2, labels: "zh" }),
    );
    fake.fire("tools/result", exec("bash"), ok("boom\n[exit code: 1]"));
    await settled(filesIn(dir).errors);
    const text = await readFile(filesIn(dir).errors, "utf8");
    expect(text).toMatch(/^## ERR-01 · /m);
    expect(text).toContain("- 指纹:");
  });
});

describe("recorderOptions", () => {
  it("maps every setting the recorder reads", () => {
    expect(recorderOptions(Config({}))).toEqual({
      capture: ["tool", "command", "llm", "agent"],
      captureExitCodes: true,
      transientThreshold: 5,
      fuzzyThreshold: 0.72,
      share: "public",
      maxEntries: 200,
      maxSampleChars: 500,
      labels: "en",
      idPrefix: "E-",
      idWidth: 4,
    });
  });

  it("share: only `private` is private; anything else is the safer public", () => {
    expect(recorderOptions(Config({ share: "private" })).share).toBe("private");
    expect(recorderOptions(Config({ share: "Private " })).share).toBe("public");
  });
});

// ---------------------------------------------------------------------------
// The recorder

describe("recorder: the record and count-only paths", () => {
  it("transient promotion end to end: counted, promoted once, then counted as hits", async () => {
    const { rec, files } = recorder({ options: { transientThreshold: 3 } });
    const rateLimit = {
      kind: "llm",
      code: "RATE_LIMIT",
      message: "429 Too Many Requests",
    } as const;
    for (let i = 0; i < 5; i++) rec.capture(rateLimit, "s1");
    await rec.idle();
    expect(rec.outcomes.map((o) => o.kind)).toEqual([
      "counted",
      "counted",
      "appended",
      "hit",
      "hit",
    ]);
    const [entry] = await readEntries(files.errors);
    expect(entry).toMatchObject({ id: "E-0001", hits: 3 });

    // Another session has its own count, but the entry exists: still a hit.
    rec.capture(rateLimit, "s2");
    await rec.idle();
    expect(rec.outcomes.at(-1)).toEqual({ kind: "hit", id: "E-0001", hits: 4 });
    expect(rec.stats).toEqual({
      appended: 1,
      hits: 3,
      counted: 2,
      timeouts: 0,
      failures: 0,
    });
  });

  it("a permanent LLM failure takes an ID at once", async () => {
    const { rec } = recorder();
    rec.capture({ kind: "llm", code: "AUTH", message: "invalid api key" });
    await rec.idle();
    expect(rec.outcomes).toEqual([{ kind: "appended", id: "E-0001" }]);
  });

  it("only the most recently active sessions keep a counter", async () => {
    const { rec } = recorder({ options: { transientThreshold: 2 } });
    const llm = { kind: "llm", code: "TIMEOUT", message: "timed out" } as const;
    rec.capture(llm, "first");
    for (let i = 0; i < MAX_SESSIONS; i++) rec.capture(llm, `other-${i}`);
    // "first" was evicted, so this is its first occurrence again, not its second.
    rec.capture(llm, "first");
    await rec.idle();
    expect(rec.stats.appended).toBe(0);

    // A session kept active is not evicted.
    const { rec: kept } = recorder({ options: { transientThreshold: 2 } });
    kept.capture(llm, "first");
    for (let i = 0; i < MAX_SESSIONS; i++) {
      kept.capture(llm, "first-keepalive");
      kept.capture(llm, `other-${i}`);
    }
    await kept.idle();
    expect(kept.stats.appended).toBe(1);
  });

  it("payloads without an agent share one plugin-wide counter", async () => {
    const { rec } = recorder({ options: { transientThreshold: 2 } });
    rec.capture({ kind: "llm", code: "SERVER", message: "502" });
    rec.capture({ kind: "llm", code: "SERVER", message: "502" });
    await rec.idle();
    expect(rec.outcomes.map((o) => o.kind)).toEqual(["counted", "appended"]);
  });

  it("an entry that vanished between match and update is appended again", async () => {
    // mtime stays put, so the cached index keeps an entry that, once `vanish`
    // is set, the store no longer finds when it re-reads under the lock.
    const files = filesIn(dir);
    const real = nodeStoreFs();
    let vanish = false;
    const fs: StoreFs = {
      ...real,
      mtimeMs: async (path) => (path === files.errors ? 1 : real.mtimeMs(path)),
      readFile: async (path) =>
        vanish && path === files.errors ? "# ERRORS\n" : real.readFile(path),
    };
    const { rec } = recorder({ fs, options: { transientThreshold: 2 } });
    const auth = { kind: "llm", code: "AUTH", message: "bad key" } as const;
    const slow = { kind: "llm", code: "RATE_LIMIT", message: "slow" } as const;
    const late = { kind: "llm", code: "TIMEOUT", message: "late" } as const;

    rec.capture(auth);
    rec.capture(slow, "s1");
    rec.capture(slow, "s1");
    // A count-only miss reads the index and writes nothing: the cache stays.
    rec.capture(late, "s1");
    await rec.idle();
    expect(rec.outcomes.map((o) => o.kind)).toEqual([
      "appended",
      "counted",
      "appended",
      "counted",
    ]);

    vanish = true;
    rec.capture(auth);
    await rec.idle();
    expect(rec.outcomes.at(-1)).toEqual({ kind: "appended", id: "E-0001" });
  });

  it("count-only against an entry that vanished counts nothing", async () => {
    const files = filesIn(dir);
    const real = nodeStoreFs();
    let vanish = false;
    let vanishedReads = 0;
    const fs: StoreFs = {
      ...real,
      mtimeMs: async (path) => (path === files.errors ? 1 : real.mtimeMs(path)),
      readFile: async (path) => {
        if (!vanish || path !== files.errors) return real.readFile(path);
        vanishedReads++;
        return "# ERRORS\n";
      },
    };
    const { rec } = recorder({ fs, options: { transientThreshold: 2 } });
    const slow = { kind: "llm", code: "RATE_LIMIT", message: "slow" } as const;
    rec.capture(slow, "s1");
    rec.capture(slow, "s1");
    // A first occurrence elsewhere reads the index and writes nothing.
    rec.capture({ kind: "llm", code: "TIMEOUT", message: "late" }, "s2");
    await rec.idle();
    expect(rec.outcomes.map((o) => o.kind)).toEqual([
      "counted",
      "appended",
      "counted",
    ]);

    vanish = true;
    rec.capture(slow, "s1");
    await rec.idle();
    // The cached index still had the entry, so the store was asked to update
    // it - and found nothing under the lock.
    expect(vanishedReads).toBe(1);
    expect(rec.outcomes.at(-1)).toEqual({ kind: "counted" });
  });
});

describe("recorder: listener bodies", () => {
  it("agentError without an agent still records", async () => {
    const { rec } = recorder();
    rec.agentError({ error: "plain string failure" });
    await rec.idle();
    expect(rec.outcomes).toEqual([{ kind: "appended", id: "E-0001" }]);
  });

  it("toolResult: a failure without info has no code", async () => {
    const { rec, files } = recorder();
    rec.toolResult(exec("fetch", {}, undefined), failed("socket hang up"));
    await rec.idle();
    const [entry] = await readEntries(files.errors);
    expect(entry?.meta.code).toBeUndefined();
    expect(entry?.title).toBe("[tool:fetch] socket hang up");
  });

  it("toolResult: a success without an exit marker writes nothing", async () => {
    const { rec } = recorder();
    rec.toolResult(exec("read"), ok("all good"));
    rec.toolResult(exec("bash"), ok("built\n[exit code: 0]"));
    await rec.idle();
    expect(rec.outcomes).toEqual([]);
  });

  it("a payload whose getters throw is swallowed and counted", () => {
    const { rec, warnings } = recorder();
    const hostile = new Proxy(
      {},
      {
        get() {
          throw new Error("hostile payload");
        },
      },
    );
    expect(() => rec.agentError(hostile as never)).not.toThrow();
    expect(() => rec.toolResult(hostile as never, ok("x"))).not.toThrow();
    expect(() => rec.capture(hostile as never)).not.toThrow();
    expect(rec.stats.failures).toBe(3);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]?.[2]).toBe("hostile payload");
  });

  it("a logger that throws does not reach the caller", () => {
    const { rec } = recorder({
      logger: {
        warn() {
          throw new Error("logger down");
        },
      },
    });
    expect(() => rec.agentError(null as never)).not.toThrow();
    expect(rec.stats.failures).toBe(1);
  });

  it("registerListeners forwards both events to the recorder", async () => {
    const fake = fakeCtx();
    const { rec } = recorder();
    registerListeners(fake.ctx, rec);
    fake.fire("agent/error", { agent: { id: "s9" }, error: new Error("boom") });
    fake.fire("tools/result", exec("write"), failed("EACCES: denied"));
    await rec.idle();
    expect(rec.stats.appended).toBe(2);
  });
});

describe("recorder: §13 write budget, retries and throttled logging", () => {
  it("a transient write failure is retried and then succeeds", async () => {
    const { clock } = fakeClock();
    const real = nodeStoreFs();
    let renames = 0;
    const fs: StoreFs = {
      ...real,
      rename: async (from, to) => {
        if (renames++ === 0)
          throw Object.assign(new Error("EBUSY: busy"), { code: "EBUSY" });
        return real.rename(from, to);
      },
    };
    const { rec, warnings } = recorder({ fs, clock });
    rec.capture({ kind: "agent", error: new Error("boom") });
    await rec.idle();
    expect(rec.outcomes).toEqual([{ kind: "appended", id: "E-0001" }]);
    expect(renames).toBe(2);
    expect(warnings).toEqual([]);
  });

  it(`gives up after ${WRITE_RETRIES} retries, counts and logs the failure`, async () => {
    const { clock } = fakeClock();
    let attempts = 0;
    const fs: StoreFs = {
      ...nodeStoreFs(),
      mtimeMs: async () => {
        attempts++;
        throw new Error("disk gone");
      },
    };
    const { rec, warnings } = recorder({ fs, clock });
    rec.capture({ kind: "agent", error: new Error("boom") });
    await rec.idle();
    expect(attempts).toBe(WRITE_RETRIES + 1);
    expect(rec.outcomes).toEqual([{ kind: "failed" }]);
    expect(rec.stats.failures).toBe(1);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]?.slice(1)).toEqual([1, "disk gone"]);
  });

  it("logs at most one failure per interval, with the running count", async () => {
    const { clock, advance } = fakeClock();
    const fs: StoreFs = {
      ...nodeStoreFs(),
      mtimeMs: async () => {
        throw new Error("disk gone");
      },
    };
    const { rec, warnings } = recorder({ fs, clock });
    rec.capture({ kind: "agent", error: new Error("one") });
    rec.capture({ kind: "agent", error: new Error("two") });
    await rec.idle();
    expect(warnings).toHaveLength(1);
    advance(FAILURE_LOG_INTERVAL_MS);
    rec.capture({ kind: "agent", error: new Error("three") });
    await rec.idle();
    expect(warnings).toHaveLength(2);
    expect(warnings[1]?.[1]).toBe(3);
  });

  it("a document that does not parse is not retried", async () => {
    const { clock } = fakeClock();
    const files = filesIn(dir);
    await writeFile(files.errors, "# ERRORS\n\n## E-0001 · broken\n");
    let reads = 0;
    const real = nodeStoreFs();
    const fs: StoreFs = {
      ...real,
      readFile: async (path) => {
        if (path === files.errors) reads++;
        return real.readFile(path);
      },
    };
    const { rec, warnings } = recorder({ fs, clock });
    rec.capture({ kind: "agent", error: new Error("boom") });
    await rec.idle();
    expect(reads).toBe(1);
    expect(rec.outcomes).toEqual([{ kind: "failed" }]);
    expect(warnings).toHaveLength(1);
  });

  it("a lock held for the whole budget skips the write silently", async () => {
    const { clock } = fakeClock();
    const files = filesIn(dir);
    const real = nodeStoreFs();
    const fs: StoreFs = {
      ...real,
      createExclusive: async () => false,
      readFile: async (path) =>
        path === files.lock ? "someone-else" : real.readFile(path),
      mtimeMs: async (path) =>
        path === files.lock ? clock.now().getTime() : real.mtimeMs(path),
    };
    const { rec, warnings } = recorder({ fs, clock });
    rec.capture({ kind: "agent", error: new Error("boom") });
    await rec.idle();
    expect(rec.outcomes).toEqual([{ kind: "timeout" }]);
    expect(rec.stats).toMatchObject({ timeouts: 1, failures: 0 });
    expect(warnings).toEqual([]);
  });

  it("a retry that would start past the budget is skipped as a timeout", async () => {
    const { clock, advance } = fakeClock();
    const fs: StoreFs = {
      ...nodeStoreFs(),
      mtimeMs: async () => {
        advance(500);
        throw new Error("slow disk");
      },
    };
    const { rec, warnings } = recorder({ fs, clock });
    rec.capture({ kind: "agent", error: new Error("boom") });
    await rec.idle();
    expect(rec.outcomes).toEqual([{ kind: "timeout" }]);
    expect(warnings).toEqual([]);
    expect(RETRY_DELAY_MS).toBeLessThan(500);
  });

  it("writes to one knowledge base run one at a time, in order", async () => {
    const { rec: a } = recorder();
    const { rec: b } = recorder();
    for (let i = 0; i < 5; i++) {
      a.capture({ kind: "agent", error: new Error(`a${i} unique failure`) });
      b.capture({ kind: "agent", error: new Error(`b${i} different one`) });
    }
    await settled(filesIn(dir).errors);
    const entries = await readEntries(filesIn(dir).errors);
    expect(entries.map((e) => e.id)).toEqual(
      Array.from(
        { length: 10 },
        (_, i) => `E-${String(i + 1).padStart(4, "0")}`,
      ),
    );
    expect(entries[0]?.title).toContain("a0");
    expect(entries[1]?.title).toContain("b0");
  });

  it("settled() of a knowledge base with nothing queued resolves at once", async () => {
    await expect(settled(join(dir, "nothing", "ERRORS.md"))).resolves.toBe(
      undefined,
    );
  });
});

// ---------------------------------------------------------------------------
// Payload readers

describe("recorder: recordFix (T14)", () => {
  it("writes the fix, redacted, and marks the entry fixed", async () => {
    const { rec, files } = recorder();
    rec.capture({ kind: "agent", error: new Error("boom") });
    const secret = "sk-" + "a".repeat(24);
    const outcome = await rec.recordFix(
      "E-0001",
      `export OPENAI_KEY=${secret} and retry`,
    );
    expect(outcome).toMatchObject({
      kind: "fixed",
      entry: { id: "E-0001", status: "fixed" },
    });
    const [entry] = await readEntries(files.errors);
    expect(entry?.status).toBe("fixed");
    expect(entry?.fix).toContain("and retry");
    expect(entry?.fix).not.toContain(secret);
  });

  it("is serialized behind the writes queued before it", async () => {
    const { rec } = recorder();
    rec.capture({ kind: "agent", error: new Error("boom") });
    // Queued right after the append, which has not run yet: it still finds
    // the entry.
    expect((await rec.recordFix("E-0001", "f")).kind).toBe("fixed");
  });

  it("an unknown ID writes nothing", async () => {
    const { rec } = recorder();
    expect(await rec.recordFix("E-0042", "f")).toEqual({ kind: "unknown" });
  });

  it("a lock held for the whole budget is a timeout, not a failure", async () => {
    const { clock } = fakeClock();
    const files = filesIn(dir);
    const real = nodeStoreFs();
    const fs: StoreFs = {
      ...real,
      createExclusive: async () => false,
      readFile: async (path) =>
        path === files.lock ? "someone-else" : real.readFile(path),
      mtimeMs: async (path) =>
        path === files.lock ? clock.now().getTime() : real.mtimeMs(path),
    };
    const { rec, warnings } = recorder({ fs, clock });
    expect(await rec.recordFix("E-0001", "f")).toEqual({ kind: "timeout" });
    expect(warnings).toEqual([]);
  });

  it("a document that does not parse is counted as a failure", async () => {
    const { rec, files, warnings } = recorder();
    await writeFile(files.errors, "# ERRORS\n\n## E-0001 · broken\n");
    expect(await rec.recordFix("E-0001", "f")).toEqual({ kind: "failed" });
    expect(rec.stats.failures).toBe(1);
    expect(warnings).toHaveLength(1);
  });
});

describe("commandFrom", () => {
  it("reads the common argument spellings", () => {
    expect(commandFrom("ls -la")).toBe("ls -la");
    expect(commandFrom({ command: "pnpm test" })).toBe("pnpm test");
    expect(commandFrom({ cmd: ["git", "status"] })).toBe("git status");
    expect(commandFrom({ script: "make", command: 3 })).toBe("make");
  });

  it("names no command when there is none", () => {
    expect(commandFrom(undefined)).toBeUndefined();
    expect(commandFrom(null)).toBeUndefined();
    expect(commandFrom(42)).toBeUndefined();
    expect(commandFrom({ path: "a.ts" })).toBeUndefined();
    expect(commandFrom({ command: ["git", 1] })).toBeUndefined();
  });
});

describe("resultText", () => {
  it("joins the text blocks and skips everything else", () => {
    expect(
      resultText([
        { type: "text", text: "one" },
        { type: "image" },
        { type: "text" },
        { type: "text", text: "two" },
      ] as never),
    ).toBe("one\ntwo");
    expect(resultText([])).toBe("");
  });
});
