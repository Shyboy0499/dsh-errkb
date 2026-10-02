import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Context } from "@deepseek-ai/cordis";
import type {
  PostToolDecision,
  ToolExecution,
  ToolExecutionResult,
} from "@deepseek-ai/dsh-tools";
import type { UserMessage } from "@deepseek-ai/dsh-llm";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Config, apply, injectionOptions } from "../src/index";
import {
  NOTICE_MAX_CHARS,
  SYSTEM_PROMPT_HINT,
  WORDING,
  fixSig,
  memoryTrustStore,
  withinCaps,
} from "../src/inject";
import type { CapLimits, TrustStore } from "../src/inject";
import { filesIn } from "../src/paths";
import { askFixText } from "../src/resolve-detect";
import {
  DEFAULT_INJECTION_OPTIONS,
  MAX_PENDING,
  MAX_SESSIONS,
  createInjection,
  createRecorder,
  registerInjection,
  settled,
} from "../src/plugin";
import type {
  InjectionOptions,
  PreStepDecision,
  Recorder,
} from "../src/plugin";
import { createStore } from "../src/store";
import type { EntryPatch } from "../src/store";

// ---------------------------------------------------------------------------
// Fixtures

let dir: string;

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "errkb-inject-"));
});

afterEach(async () => {
  await settled(filesIn(dir).errors);
  await rm(dir, { recursive: true, force: true });
});

type Listener = (...args: unknown[]) => unknown;

/** A fake agent: an id, and the messages agent.inject() received. */
function agent(id = "s1") {
  const injected: UserMessage[] = [];
  return {
    id,
    injected,
    inject(message: UserMessage) {
      injected.push(message);
    },
  };
}

type FakeAgent = ReturnType<typeof agent>;

/**
 * A context that records registrations and plays the host: emits call every
 * listener, waterfalls call the plugin's listener with a `next` that resolves
 * to whatever the test says the later listeners returned.
 */
function fakeCtx(options: { sectionThrows?: boolean } = {}) {
  const listeners = new Map<string, Listener[]>();
  const sections: Array<{ name: string; order: number; text: string }> = [];
  const warnings: unknown[][] = [];
  const ctx = {
    on(name: string, listener: Listener) {
      listeners.set(name, [...(listeners.get(name) ?? []), listener]);
      return () => true;
    },
    tools: {
      register: () => () => undefined,
    },
    systemPrompt: {
      section(section: { name: string; order: number; text: string }) {
        if (options.sectionThrows) throw new Error("duplicate section");
        sections.push(section);
        return () => undefined;
      },
    },
    logger: {
      info: () => undefined,
      warn: (...args: unknown[]) => {
        warnings.push(args);
      },
      error: () => undefined,
      debug: () => undefined,
    },
  };
  const only = (name: string) => {
    const found = listeners.get(name) ?? [];
    expect(found).toHaveLength(1);
    return found[0] as Listener;
  };
  return {
    ctx: ctx as unknown as Context,
    listeners,
    sections,
    warnings,
    emit(name: string, ...args: unknown[]) {
      for (const listener of listeners.get(name) ?? []) listener(...args);
    },
    postExecute(
      e: Readonly<ToolExecution>,
      r: Readonly<ToolExecutionResult>,
      downstream: PostToolDecision = { kind: "accept" },
    ) {
      return only("tools/post-execute")(
        e,
        r,
        async () => downstream,
      ) as Promise<PostToolDecision>;
    },
    preStep(
      a: FakeAgent,
      turn: number,
      step: number,
      downstream: PreStepDecision = { kind: "enter", messages: [] },
    ) {
      return only("agent/pre-step")(
        {
          agent: a,
          messages: [],
          turn,
          step,
          signal: new AbortController().signal,
        },
        async () => downstream,
      ) as Promise<PreStepDecision>;
    },
    sessionStart(a: FakeAgent, source = "startup") {
      return only("agent/session-start")({ agent: a, source });
    },
  };
}

type Fake = ReturnType<typeof fakeCtx>;

const exec = (
  name: string,
  args: unknown = {},
  a: { id: string } | null = { id: "s1" },
) =>
  ({
    name,
    arguments: args,
    ...(a === null ? {} : { agent: a }),
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
const FIX = "run pnpm add missing-pkg, then re-run tsc";
const CAUSE = "the dependency was never installed";

const textOf = (message: UserMessage | undefined) =>
  (message?.content?.[0] as { text: string } | undefined)?.text;

/** The notices a post-execute decision carries. */
const contexts = (decision: PostToolDecision) =>
  (decision.additionalContexts ?? []).map(textOf);

/** The messages an entered pre-step carries. */
const entered = (decision: PreStepDecision) =>
  decision.kind === "enter" ? decision.messages.map(textOf) : [];

/** Apply the plugin to a temporary knowledge base. */
function applied(config: Partial<Config> = {}) {
  const fake = fakeCtx();
  apply(fake.ctx, Config({ kbDir: dir, ...config }));
  return fake;
}

/** Edit an entry the way a person (or a later err_record) would. */
async function edit(id: string, patch: EntryPatch) {
  await settled(filesIn(dir).errors);
  await createStore(filesIn(dir)).update(id, patch);
}

/** One failing tool call, through both pipeline hooks, as the registry runs them. */
async function call(
  fake: Fake,
  e: Readonly<ToolExecution>,
  r: Readonly<ToolExecutionResult>,
  downstream?: PostToolDecision,
) {
  const decision = await fake.postExecute(e, r, downstream);
  fake.emit("tools/result", e, r);
  await settled(filesIn(dir).errors);
  return decision;
}

/** Each entry's effective hits: its ERRORS.md baseline plus state.json. */
async function effectiveHits() {
  const index = await createRecorder({
    files: filesIn(dir),
    logger: { warn: () => undefined },
  }).entries();
  return index?.map(({ entry }) => [entry.id, entry.hits]);
}

/** A recorder and injection wired by hand, for the tests that need the parts. */
function wired(
  options: Partial<InjectionOptions> = {},
  over: {
    trust?: TrustStore;
    caps?: Partial<CapLimits>;
    recorder?: Recorder;
  } = {},
) {
  const fake = fakeCtx();
  const recorder =
    over.recorder ??
    createRecorder({
      files: filesIn(dir),
      logger: { warn: (...args) => fake.warnings.push(args) },
    });
  const injection = createInjection({ recorder, options, ...over });
  registerInjection(fake.ctx, injection, recorder);
  return { fake, recorder, injection };
}

// ---------------------------------------------------------------------------
// Registration and settings

describe("registration", () => {
  it("publishes the standing section at plugin:errkb / 10400", () => {
    const fake = applied();
    expect(fake.sections).toEqual([
      { name: "plugin:errkb", order: 10400, text: SYSTEM_PROMPT_HINT },
    ]);
  });

  it("systemPromptHint: false publishes no section", () => {
    const fake = applied({ systemPromptHint: false });
    expect(fake.sections).toEqual([]);
    expect(fake.listeners.has("tools/post-execute")).toBe(true);
  });

  it("a section that cannot be registered is logged; the hooks still register", () => {
    const fake = fakeCtx({ sectionThrows: true });
    apply(fake.ctx, Config({ kbDir: dir }));
    expect(fake.listeners.has("agent/pre-step")).toBe(true);
    expect(fake.warnings).toHaveLength(1);
    expect(String(fake.warnings[0]?.[1])).toBe("1");
  });

  it("the emit listeners return undefined, as their events declare", async () => {
    const fake = applied();
    const result = fake.listeners.get("tools/result")?.[0] as Listener;
    const error = fake.listeners.get("agent/error")?.[0] as Listener;
    const start = fake.listeners.get("agent/session-start")?.[0] as Listener;
    expect(result(exec("read"), ok("fine"))).toBeUndefined();
    expect(error({ error: new Error("x") })).toBeUndefined();
    // The digest's promise goes back to the host; it settles to undefined.
    expect(
      await (start({ agent: agent(), source: "startup" }) as Promise<void>),
    ).toBeUndefined();
  });
});

describe("injectionOptions", () => {
  it("maps the four settings, defaults first", () => {
    expect(injectionOptions(Config({}))).toEqual(DEFAULT_INJECTION_OPTIONS);
    expect(
      injectionOptions(
        Config({
          inject: "always",
          captureFix: "off",
          sessionDigest: "index",
          systemPromptHint: false,
        }),
      ),
    ).toEqual({
      inject: "always",
      captureFix: "off",
      sessionDigest: "index",
      systemPromptHint: false,
    });
  });

  it("a value outside the documented set falls back to the default", () => {
    expect(
      injectionOptions(
        Config({ inject: "loud", captureFix: "always", sessionDigest: "all" }),
      ),
    ).toMatchObject({
      inject: "hit-only",
      captureFix: "prompt-once",
      sessionDigest: "counts",
    });
  });
});

// ---------------------------------------------------------------------------
// tools/post-execute

describe("tools/post-execute: the T13 acceptance path", () => {
  it("a repeated failure gets the known fix in additionalContexts, capped", async () => {
    const fake = applied();
    const a = agent();
    const shell = (args = { command: "pnpm tsc" }) => exec("shell", args, a);

    // Turn 1, step 1: the first failure is a miss - silent under hit-only -
    // and becomes E-0001.
    await fake.preStep(a, 1, 1);
    expect(contexts(await call(fake, shell(), ok(TSC_A)))).toEqual([]);
    await edit("E-0001", { fix: FIX, trigger: CAUSE });

    // Step 2: the same failure, another path and line: the known fix.
    await fake.preStep(a, 1, 2);
    const hit = await call(fake, shell(), ok(TSC_B));
    const [notice] = contexts(hit);
    expect(notice).toBe(
      `[errkb] E-0001 known (2 hits) | cause: ${CAUSE} | fix: ${FIX} ${WORDING.hit}`,
    );
    expect(withinCaps(notice as string)).toBe(true);
    expect(hit.additionalContexts?.[0]).toMatchObject({
      role: "user",
      source: { kind: "plugin", plugin: "err-kb", form: "notice" },
    });

    // Same step again: one notice per step.
    expect(contexts(await call(fake, shell(), ok(TSC_A)))).toEqual([]);

    // Turn 2: the second notice for E-0001 this session, then none - two per
    // ID per session.
    await fake.preStep(a, 2, 1);
    expect(contexts(await call(fake, shell(), ok(TSC_A)))).toHaveLength(1);
    await fake.preStep(a, 2, 2);
    expect(contexts(await call(fake, shell(), ok(TSC_A)))).toEqual([]);

    // Capture kept recording throughout.
    expect(await effectiveHits()).toEqual([["E-0001", 5]]);
  });

  it("at most three notices per turn, across entries", async () => {
    const { fake } = wired();
    const a = agent();
    for (let i = 0; i < 4; i++) {
      await call(
        fake,
        exec("t", {}, a),
        failed(`distinct failure ${i} zzz${i}`),
      );
      await edit(`E-000${i + 1}`, { fix: `fix ${i}` });
    }
    await fake.preStep(a, 1, 1);
    const counts: number[] = [];
    for (let step = 0; step < 4; step++) {
      await fake.preStep(a, 1, step + 2);
      const decision = await call(
        fake,
        exec("t", {}, a),
        failed(`distinct failure ${step} zzz${step}`),
      );
      counts.push(contexts(decision).length);
    }
    expect(counts).toEqual([1, 1, 1, 0]);
  });

  it("keeps what later listeners returned, and appends after it", async () => {
    const { fake } = wired();
    const a = agent();
    await call(fake, exec("t", {}, a), failed("EPERM: rename", "EPERM"));
    await edit("E-0001", { fix: "close the editor" });
    const theirs = { id: "m0" } as unknown as UserMessage;
    const downstream: PostToolDecision = {
      kind: "block",
      feedback: [{ type: "text", text: "blocked" }],
      additionalContexts: [theirs],
    };
    const decision = await call(
      fake,
      exec("t", {}, a),
      failed("EPERM: rename", "EPERM"),
      downstream,
    );
    expect(decision).not.toBe(downstream);
    expect(decision).toMatchObject({
      kind: "block",
      feedback: downstream.feedback,
    });
    expect(decision.additionalContexts?.[0]).toBe(theirs);
    expect(decision.additionalContexts).toHaveLength(2);
    expect(downstream.additionalContexts).toEqual([theirs]);
  });

  it("returns the downstream decision itself when there is nothing to add", async () => {
    const { fake } = wired();
    const downstream: PostToolDecision = { kind: "accept" };
    // A success, a first-time failure, and a call without an agent.
    expect(await fake.postExecute(exec("read"), ok("fine"), downstream)).toBe(
      downstream,
    );
    expect(await call(fake, exec("t"), failed("new"), downstream)).toBe(
      downstream,
    );
    await edit("E-0001", { fix: "f" });
    expect(
      await call(fake, exec("t", {}, null), failed("new"), downstream),
    ).toBe(downstream);
  });

  it("an execution whose agent getter throws is counted; the decision passes through", async () => {
    const { fake, recorder } = wired();
    const hostile = {
      name: "t",
      arguments: {},
      get agent(): never {
        throw new Error("getter");
      },
    } as unknown as Readonly<ToolExecution>;
    const downstream: PostToolDecision = { kind: "accept" };
    expect(await fake.postExecute(hostile, failed("x"), downstream)).toBe(
      downstream,
    );
    expect(recorder.stats.failures).toBe(1);
  });

  it("a lookup whose classification throws is counted and finds nothing", async () => {
    const recorder = createRecorder({
      files: filesIn(dir),
      logger: { warn: () => undefined },
    });
    const input = {
      kind: "tool",
      toolName: "t",
      isError: true,
      get message(): never {
        throw new Error("getter");
      },
    };
    expect(await recorder.lookup(input as never)).toBeUndefined();
    expect(recorder.stats.failures).toBe(1);
  });

  it("a rejection from next() is not ours and passes through", async () => {
    const { fake } = wired();
    const listener = fake.listeners.get("tools/post-execute")?.[0] as Listener;
    await expect(
      listener(exec("t"), failed("x"), async () => {
        throw new Error("downstream");
      }),
    ).rejects.toThrow("downstream");
  });

  it("internals that throw leave the downstream decision untouched", async () => {
    const real = createRecorder({
      files: filesIn(dir),
      logger: { warn: () => undefined },
    });
    let failures = 0;
    const recorder: Recorder = {
      ...real,
      lookup: () => {
        throw new Error("lookup broke");
      },
      fail: () => {
        failures++;
      },
    };
    const { fake } = wired({}, { recorder });
    const downstream: PostToolDecision = { kind: "accept" };
    expect(await fake.postExecute(exec("t"), failed("x"), downstream)).toBe(
      downstream,
    );
    expect(failures).toBe(1);
  });

  it("a malformed result is counted, rejects nothing while next() runs, and passes the decision through", async () => {
    const { fake, recorder } = wired();
    const listener = fake.listeners.get("tools/post-execute")?.[0] as Listener;
    const unhandled: unknown[] = [];
    const spy = (reason: unknown) => unhandled.push(reason);
    process.on("unhandledRejection", spy);
    try {
      // `isError` with no `error`: reading the payload throws.
      const malformed = {
        isError: true,
        content: [],
      } as unknown as Readonly<ToolExecutionResult>;
      const downstream: PostToolDecision = { kind: "accept" };
      // A slow next(): the lookup settles while it is still pending, and Node
      // reports an unhandled rejection at the end of a macrotask, so the
      // window is real.
      const slow = () =>
        new Promise<PostToolDecision>((resolve) => {
          setTimeout(() => resolve(downstream), 20);
        });
      expect(await listener(exec("t"), malformed, slow)).toBe(downstream);
      await new Promise((resolve) => setTimeout(resolve, 10));
      expect(unhandled).toEqual([]);
      expect(recorder.stats.failures).toBe(1);
    } finally {
      process.off("unhandledRejection", spy);
    }
  });

  it("a lookup that rejects is counted, even when next() rejects first", async () => {
    const real = createRecorder({
      files: filesIn(dir),
      logger: { warn: () => undefined },
    });
    let failures = 0;
    const recorder: Recorder = {
      ...real,
      lookup: () => Promise.reject(new Error("lookup broke")),
      fail: () => {
        failures++;
      },
    };
    const { fake } = wired({}, { recorder });
    const listener = fake.listeners.get("tools/post-execute")?.[0] as Listener;
    const unhandled: unknown[] = [];
    const spy = (reason: unknown) => unhandled.push(reason);
    process.on("unhandledRejection", spy);
    try {
      await expect(
        listener(exec("t"), failed("x"), async () => {
          throw new Error("downstream");
        }),
      ).rejects.toThrow("downstream");
      await new Promise((resolve) => setTimeout(resolve, 10));
      expect(unhandled).toEqual([]);
      expect(failures).toBe(1);
    } finally {
      process.off("unhandledRejection", spy);
    }
  });

  it("a notice that cannot be merged leaves the downstream decision untouched", async () => {
    const { fake, recorder } = wired();
    await call(fake, exec("t"), failed("EPERM: rename", "EPERM"));
    await edit("E-0001", { fix: "f" });
    const hostile = {
      kind: "accept",
      get additionalContexts(): never {
        throw new Error("getter");
      },
    } as unknown as PostToolDecision;
    expect(
      await call(fake, exec("t"), failed("EPERM: rename", "EPERM"), hostile),
    ).toBe(hostile);
    expect(recorder.stats.failures).toBe(1);
  });

  it("an offer that throws is counted, and nothing is added", async () => {
    const { fake, recorder, injection } = wired();
    await call(fake, exec("t"), failed("EPERM: rename", "EPERM"));
    await edit("E-0001", { fix: "f" });
    injection.trust.seen = () => {
      throw new Error("trust broke");
    };
    const downstream: PostToolDecision = { kind: "accept" };
    expect(
      await call(fake, exec("t"), failed("EPERM: rename", "EPERM"), downstream),
    ).toBe(downstream);
    expect(recorder.stats.failures).toBe(1);
  });
});

describe("inject: 'off'", () => {
  it("says nothing while capture still records", async () => {
    const fake = applied({ inject: "off" });
    const a = agent();
    await fake.preStep(a, 1, 1);
    await call(fake, exec("shell", { command: "pnpm tsc" }, a), ok(TSC_A));
    await edit("E-0001", { fix: FIX });
    await fake.preStep(a, 1, 2);
    const downstream: PostToolDecision = { kind: "accept" };
    expect(
      await call(
        fake,
        exec("shell", { command: "pnpm tsc" }, a),
        ok(TSC_B),
        downstream,
      ),
    ).toBe(downstream);
    fake.emit("agent/error", { agent: a, error: new Error("turn died") });
    await settled(filesIn(dir).errors);
    const next: PreStepDecision = { kind: "enter", messages: [] };
    expect(await fake.preStep(a, 2, 1, next)).toBe(next);
    expect(await effectiveHits()).toEqual([
      ["E-0001", 2],
      ["E-0002", 1],
    ]);
  });
});

describe("inject: 'always'", () => {
  it("a tool miss names its new ID on the next step, once it is written", async () => {
    const { fake } = wired({ inject: "always" });
    const a = agent();
    await fake.preStep(a, 1, 1);
    const decision = await call(
      fake,
      exec("t", {}, a),
      failed("brand new failure"),
    );
    expect(contexts(decision)).toEqual([]);
    expect(entered(await fake.preStep(a, 1, 2))).toEqual([
      "[errkb] recorded as E-0001 (no fix yet).",
    ]);
    // Announced once.
    expect(entered(await fake.preStep(a, 1, 3))).toEqual([]);
  });

  it("a miss whose write has not settled waits for a later step", async () => {
    const { fake } = wired({ inject: "always" });
    const a = agent();
    const e = exec("t", {}, a);
    await fake.postExecute(e, failed("brand new failure"));
    fake.emit("tools/result", e, failed("brand new failure"));
    // The write is queued, not done: nothing yet, and the notice is kept.
    expect(entered(await fake.preStep(a, 1, 2))).toEqual([]);
    await settled(filesIn(dir).errors);
    expect(entered(await fake.preStep(a, 1, 3))).toEqual([
      "[errkb] recorded as E-0001 (no fix yet).",
    ]);
  });

  it("a miss whose write did not append says nothing", async () => {
    const { fake } = wired({ inject: "always" });
    const a = agent();
    const e = exec("t", {}, a);
    // The pending miss, but tools/result sees a result that is not captured.
    await fake.postExecute(e, failed("brand new failure"));
    fake.emit("tools/result", e, ok("fine after all"));
    await settled(filesIn(dir).errors);
    expect(entered(await fake.preStep(a, 1, 2))).toEqual([]);
  });

  it("a turn error that misses is announced on the next step too", async () => {
    const { fake } = wired({ inject: "always" });
    const a = agent();
    fake.emit("agent/error", {
      agent: a,
      turn: 1,
      step: 1,
      error: new Error("turn died"),
    });
    await settled(filesIn(dir).errors);
    expect(entered(await fake.preStep(a, 2, 1))).toEqual([
      "[errkb] recorded as E-0001 (no fix yet).",
    ]);
  });
});

// ---------------------------------------------------------------------------
// agent/pre-step

describe("agent/pre-step: a dead turn's notice rides the next step", () => {
  async function knownTurnError() {
    const ctx = wired();
    fakeTurnError(ctx.fake);
    await settled(filesIn(dir).errors);
    await edit("E-0001", {
      fix: "lower maxTokens",
      trigger: "context too long",
    });
    return ctx;
  }

  const a = agent();
  const fakeTurnError = (fake: Fake) =>
    fake.emit("agent/error", {
      agent: a,
      turn: 1,
      step: 3,
      error: Object.assign(new Error("context window exceeded"), {
        code: "CONTEXT_OVERFLOW",
      }),
    });

  it("appends the entry's notice after the step's own messages", async () => {
    const { fake } = await knownTurnError();
    fakeTurnError(fake);
    const theirs = { id: "m1" } as unknown as UserMessage;
    const downstream: PreStepDecision = { kind: "enter", messages: [theirs] };
    const decision = await fake.preStep(a, 2, 1, downstream);
    expect(decision).not.toBe(downstream);
    expect(decision.kind === "enter" && decision.messages[0]).toBe(theirs);
    expect(entered(decision)[1]).toMatch(
      /^\[errkb\] E-0001 known \(2 hits\) \| cause: context too long \| fix: lower maxTokens /,
    );
    expect(downstream.kind === "enter" && downstream.messages).toEqual([
      theirs,
    ]);
    // Delivered once.
    const next: PreStepDecision = { kind: "enter", messages: [] };
    expect(await fake.preStep(a, 2, 2, next)).toBe(next);
  });

  it("keeps the notice when the step is rejected, for the next one", async () => {
    const { fake } = await knownTurnError();
    fakeTurnError(fake);
    const reject: PreStepDecision = { kind: "reject" };
    expect(await fake.preStep(a, 2, 1, reject)).toBe(reject);
    expect(entered(await fake.preStep(a, 2, 2))).toHaveLength(1);
  });

  it("several pending errors still give one notice per step", async () => {
    const { fake } = await knownTurnError();
    fakeTurnError(fake);
    fakeTurnError(fake);
    expect(entered(await fake.preStep(a, 2, 1))).toHaveLength(1);
    expect(entered(await fake.preStep(a, 2, 2))).toEqual([]);
  });

  it(`more than ${MAX_PENDING} pending errors: all observed, one notice, nothing left over`, async () => {
    const { fake, injection } = await knownTurnError();
    let offered = 0;
    const offer = injection.trust.seen.bind(injection.trust);
    injection.trust.seen = (...args) => {
      offered++;
      offer(...args);
    };
    for (let i = 0; i < MAX_PENDING + 3; i++) fakeTurnError(fake);
    await settled(filesIn(dir).errors);
    await fake.preStep(a, 2, 1);
    expect(offered).toBe(MAX_PENDING + 3);
    // Observed as captured, but only the last MAX_PENDING were still queued:
    // none is left over for the next step.
    expect(entered(await fake.preStep(a, 2, 2))).toEqual([]);
  });

  it("a notice that cannot be merged leaves the decision untouched", async () => {
    const { fake, recorder } = await knownTurnError();
    fakeTurnError(fake);
    const hostile = {
      kind: "enter",
      get messages(): never {
        throw new Error("getter");
      },
    } as unknown as PreStepDecision;
    expect(await fake.preStep(a, 2, 1, hostile)).toBe(hostile);
    expect(recorder.stats.failures).toBe(1);
  });

  it("a payload whose agent throws still returns the downstream decision", async () => {
    const { fake, recorder } = wired();
    const listener = fake.listeners.get("agent/pre-step")?.[0] as Listener;
    const downstream: PreStepDecision = { kind: "enter", messages: [] };
    const payload = {
      get agent(): never {
        throw new Error("getter");
      },
    };
    expect(await listener(payload, async () => downstream)).toBe(downstream);
    expect(recorder.stats.failures).toBe(1);
  });

  it("a turn error without an agent is recorded and never injected", async () => {
    const { fake } = wired({ inject: "always" });
    fake.emit("agent/error", { error: new Error("no agent here") });
    await settled(filesIn(dir).errors);
    expect(entered(await fake.preStep(a, 1, 1))).toEqual([]);
    const document = await createStore(filesIn(dir)).read();
    expect(document.blocks).toHaveLength(1);
  });

  it("an agent/error whose lookup throws is counted, and still nothing throws", async () => {
    const real = createRecorder({
      files: filesIn(dir),
      logger: { warn: () => undefined },
    });
    let failures = 0;
    const recorder: Recorder = {
      ...real,
      lookup: () => {
        throw new Error("lookup broke");
      },
      fail: () => {
        failures++;
      },
    };
    const { fake } = wired({}, { recorder });
    fake.emit("agent/error", { agent: a, error: new Error("x") });
    expect(failures).toBe(1);
  });

  it("an agent/error payload whose agent getter throws is counted", () => {
    const { fake, recorder } = wired();
    const hostile = {
      error: new Error("x"),
      get agent(): never {
        throw new Error("getter");
      },
    };
    fake.emit("agent/error", hostile);
    expect(recorder.stats.failures).toBe(1);
  });

  it("a lookup or a write that rejects is counted and never left unhandled", async () => {
    const real = createRecorder({
      files: filesIn(dir),
      logger: { warn: () => undefined },
    });
    let failures = 0;
    const recorder: Recorder = {
      ...real,
      lookup: () => Promise.reject(new Error("lookup broke")),
      agentError: () => Promise.reject(new Error("write broke")),
      fail: () => {
        failures++;
      },
    };
    const { fake } = wired({ inject: "always" }, { recorder });
    const unhandled: unknown[] = [];
    const spy = (reason: unknown) => unhandled.push(reason);
    process.on("unhandledRejection", spy);
    try {
      fake.emit("agent/error", { agent: a, error: new Error("x") });
      await new Promise((resolve) => setTimeout(resolve, 10));
      expect(unhandled).toEqual([]);
      expect(failures).toBe(2);
      expect(entered(await fake.preStep(a, 1, 1))).toEqual([]);
    } finally {
      process.off("unhandledRejection", spy);
    }
  });

  it("an observe that throws is counted, and the step still enters", async () => {
    const { fake, recorder, injection } = await knownTurnError();
    injection.trust.seen = () => {
      throw new Error("trust broke");
    };
    fakeTurnError(fake);
    expect(entered(await fake.preStep(a, 2, 1))).toEqual([]);
    expect(recorder.stats.failures).toBe(1);
  });

  it("a tools/result whose recorder throws is counted", () => {
    const real = createRecorder({
      files: filesIn(dir),
      logger: { warn: () => undefined },
    });
    let failures = 0;
    const recorder: Recorder = {
      ...real,
      toolResult: () => {
        throw new Error("recorder broke");
      },
      fail: () => {
        failures++;
      },
    };
    const { fake } = wired({}, { recorder });
    fake.emit("tools/result", exec("t"), failed("x"));
    expect(failures).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Fix trust

describe("fix trust through the hooks", () => {
  it("a fix captured again later in the turn that injected it counts a recurrence", async () => {
    const { fake, injection } = wired();
    const a = agent();
    await call(fake, exec("t", {}, a), failed("EPERM: rename", "EPERM"));
    await edit("E-0001", { fix: "close the editor" });

    await fake.preStep(a, 1, 1);
    const first = contexts(
      await call(fake, exec("t", {}, a), failed("EPERM: rename", "EPERM")),
    );
    expect(first[0]).toContain(WORDING.hit);
    await fake.preStep(a, 1, 2);
    const second = contexts(
      await call(fake, exec("t", {}, a), failed("EPERM: rename", "EPERM")),
    );
    expect(second[0]).toContain(WORDING.doubted);
    expect(injection.trust.snapshot().entries["E-0001"]).toMatchObject({
      injected: 2,
      recurredAfterInject: 1,
    });
  });

  it("a recurrence in a later turn, or another session, does not count", async () => {
    const store = memoryTrustStore();
    const { fake, injection } = wired({}, { trust: store });
    const a = agent("a");
    const b = agent("b");
    await call(fake, exec("t", {}, a), failed("EPERM: rename", "EPERM"));
    await edit("E-0001", { fix: "close the editor" });

    await fake.preStep(a, 1, 1);
    await call(fake, exec("t", {}, a), failed("EPERM: rename", "EPERM"));
    // Session b, mid-turn of a: b's new turn must not erase a's injection,
    // and b's own capture is not a's recurrence.
    await fake.preStep(b, 1, 1);
    await call(fake, exec("t", {}, b), failed("EPERM: rename", "EPERM"));
    expect(
      injection.trust.snapshot().entries["E-0001"]?.recurredAfterInject,
    ).toBe(0);
    // a's next turn: no longer the turn that injected.
    await fake.preStep(a, 2, 1);
    await call(fake, exec("t", {}, a), failed("EPERM: rename", "EPERM"));
    expect(store.load().entries["E-0001"]).toMatchObject({
      injected: 3,
      recurredAfterInject: 0,
    });
  });
});

// ---------------------------------------------------------------------------
// state.json

describe("state.json: counters stay out of ERRORS.md (§4.3)", () => {
  const readState = async () =>
    JSON.parse(await readFile(filesIn(dir).state, "utf8")) as {
      entries: Record<string, { hits: number; lastSeen: string }>;
      trust: Record<string, unknown>;
    };

  it("a repeated failure leaves ERRORS.md byte-identical; hits count in state.json and the notice adds them up", async () => {
    const fake = applied({ sessionDigest: "index" });
    const a = agent();
    const shell = () => exec("shell", { command: "pnpm tsc" }, a);

    await fake.preStep(a, 1, 1);
    await call(fake, shell(), ok(TSC_A));
    // A baseline a person typed: the block says 5.
    await edit("E-0001", { fix: FIX, hits: 5 });
    const before = await readFile(filesIn(dir).errors, "utf8");
    await expect(readFile(filesIn(dir).state, "utf8")).rejects.toThrow();

    // Repeat 1: the notice counts this occurrence too, 5 + 0 + 1.
    await fake.preStep(a, 1, 2);
    expect(contexts(await call(fake, shell(), ok(TSC_B)))[0]).toMatch(
      /^\[errkb\] E-0001 known \(6 hits\)/,
    );
    expect(await readFile(filesIn(dir).errors, "utf8")).toBe(before);
    expect((await readState()).entries["E-0001"]?.hits).toBe(1);

    // Repeat 2: 5 + 1 + 1.
    await fake.preStep(a, 2, 1);
    expect(contexts(await call(fake, shell(), ok(TSC_A)))[0]).toMatch(
      /^\[errkb\] E-0001 known \(7 hits\)/,
    );
    expect(await readFile(filesIn(dir).errors, "utf8")).toBe(before);
    expect((await readState()).entries["E-0001"]).toEqual({
      hits: 2,
      lastSeen: expect.stringMatching(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$/),
    });
    expect(await effectiveHits()).toEqual([["E-0001", 7]]);

    // The digest's index shows the effective count as well.
    const b = agent("s2");
    await fake.sessionStart(b);
    expect(textOf(b.injected[0])).toContain("E-0001 ");
    expect(textOf(b.injected[0])).toContain("(7 hits)");
  });

  it("deleting state.json is harmless: the counts fall back to the baselines", async () => {
    const fake = applied();
    const shell = () => exec("shell", { command: "pnpm tsc" });
    await call(fake, shell(), ok(TSC_A));
    await call(fake, shell(), ok(TSC_B));
    await call(fake, shell(), ok(TSC_A));
    expect(await effectiveHits()).toEqual([["E-0001", 3]]);
    await rm(filesIn(dir).state);
    expect(await effectiveHits()).toEqual([["E-0001", 1]]);
    // The next repeat starts a fresh delta on top of the baseline.
    await call(fake, shell(), ok(TSC_B));
    expect(await effectiveHits()).toEqual([["E-0001", 2]]);
  });

  it("fix trust lives in state.json: a suppressed fix stays suppressed after a restart", async () => {
    // Another harness on this machine already trusts E-0009; it is kept.
    const other = {
      injected: 1,
      recurredAfterInject: 0,
      succeeded: 1,
      fixSig: fixSig("other"),
    };
    await writeFile(
      filesIn(dir).state,
      JSON.stringify({ version: 1, entries: {}, trust: { "E-0009": other } }),
    );
    const first = wired();
    await first.injection.trust.ready;
    const a = agent();
    const fail = () => exec("t", {}, a);
    await call(first.fake, fail(), failed("EPERM: rename", "EPERM"));
    await edit("E-0001", { fix: "close the editor" });

    // Injected, then captured again in the same turn, twice.
    await first.fake.preStep(a, 1, 1);
    await call(first.fake, fail(), failed("EPERM: rename", "EPERM"));
    await first.fake.preStep(a, 1, 2);
    await call(first.fake, fail(), failed("EPERM: rename", "EPERM"));
    await first.fake.preStep(a, 1, 3);
    await call(first.fake, fail(), failed("EPERM: rename", "EPERM"));
    expect(first.injection.trust.level("E-0001", "close the editor")).toBe(
      "suppressed",
    );
    await settled(filesIn(dir).errors);
    const state = await readState();
    expect(state.trust).toEqual({
      "E-0009": other,
      "E-0001": {
        injected: 2,
        recurredAfterInject: 2,
        succeeded: 0,
        fixSig: fixSig("close the editor"),
      },
    });
    // Machine-local: nothing of it reaches ERRORS.md.
    expect(await readFile(filesIn(dir).errors, "utf8")).not.toMatch(
      /recurred|injected/,
    );

    // A new recorder and injection layer: the restart.
    const second = wired();
    await second.injection.trust.ready;
    expect(second.injection.trust.level("E-0001", "close the editor")).toBe(
      "suppressed",
    );
    const b = agent("b");
    await second.fake.preStep(b, 1, 1);
    expect(
      contexts(
        await call(
          second.fake,
          exec("t", {}, b),
          failed("EPERM: rename", "EPERM"),
        ),
      ),
    ).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// agent/session-start

describe("agent/session-start: the digest", () => {
  async function seed(n: number) {
    const store = createStore(filesIn(dir));
    for (let i = 1; i <= n; i++)
      await store.append({
        title: `failure number ${i}`,
        signature: `sig${i}`,
        category: "tool / t",
        meta: { cat: "tool" },
        raw: `failure number ${i}`,
      });
    return store;
  }

  it("counts (the default): one line", async () => {
    await seed(37);
    const fake = applied();
    const a = agent();
    await fake.sessionStart(a);
    expect(a.injected.map(textOf)).toEqual([
      "[errkb] 37 known errors; known fixes are shown when an error repeats.",
    ]);
    expect(a.injected[0]?.source).toMatchObject({
      kind: "plugin",
      plugin: "err-kb",
      form: "notice",
    });
  });

  it("index: the most-hit titles, at most ten, wontfix left out", async () => {
    const store = await seed(12);
    await store.update("E-0012", { hits: 9 });
    await store.update("E-0003", { hits: 4 });
    await store.update("E-0001", { status: "wontfix", hits: 50 });
    const fake = applied({ sessionDigest: "index" });
    const a = agent();
    await fake.sessionStart(a);
    const lines = (textOf(a.injected[0]) as string).split("\n");
    expect(lines[0]).toBe(
      "[errkb] 12 known errors; known fixes are shown when an error repeats. Most frequent:",
    );
    expect(lines.slice(1, 4)).toEqual([
      "E-0012 failure number 12 (9 hits)",
      "E-0003 failure number 3 (4 hits)",
      "E-0002 failure number 2 (1 hit)",
    ]);
    expect(lines).toHaveLength(11);
    expect(lines.join("\n")).not.toContain("E-0001");
    // The summary is the head line alone: a one-line account.
    expect(a.injected[0]?.source).toMatchObject({ summary: lines[0] });
  });

  it("off: nothing", async () => {
    await seed(3);
    const fake = applied({ sessionDigest: "off" });
    const a = agent();
    await fake.sessionStart(a);
    expect(a.injected).toEqual([]);
  });

  it("an empty knowledge base: nothing", async () => {
    const fake = applied();
    const a = agent();
    await fake.sessionStart(a);
    expect(a.injected).toEqual([]);
  });

  it("an unreadable knowledge base, or an agent that throws, is counted and swallowed", async () => {
    const { fake, recorder } = wired();
    const { writeFile } = await import("node:fs/promises");
    await writeFile(filesIn(dir).errors, "# ERRORS\n\n## E-0001 · broken\n");
    const a = agent();
    await fake.sessionStart(a);
    expect(a.injected).toEqual([]);
    expect(recorder.stats.failures).toBe(1);

    await rm(filesIn(dir).errors);
    await seed(1);
    const hostile = {
      id: "h",
      inject: () => {
        throw new Error("inject broke");
      },
    };
    await fake.sessionStart(hostile as unknown as FakeAgent);
    expect(recorder.stats.failures).toBe(2);
  });

  it("a new lifecycle starts the session's caps afresh", async () => {
    const { fake } = wired();
    const a = agent();
    await call(fake, exec("t", {}, a), failed("EPERM: rename", "EPERM"));
    await edit("E-0001", { fix: "close the editor", status: "fixed" });
    await fake.preStep(a, 1, 1);
    expect(
      contexts(
        await call(fake, exec("t", {}, a), failed("EPERM: rename", "EPERM")),
      ),
    ).toHaveLength(1);
    // A fixed entry speaks once per session...
    await fake.preStep(a, 2, 1);
    expect(
      contexts(
        await call(fake, exec("t", {}, a), failed("EPERM: rename", "EPERM")),
      ),
    ).toEqual([]);
    // ...and a cleared session is a new one.
    await fake.sessionStart(a, "clear");
    await fake.preStep(a, 1, 1);
    expect(
      contexts(
        await call(fake, exec("t", {}, a), failed("EPERM: rename", "EPERM")),
      ),
    ).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// Sessions

describe("per-session state", () => {
  it(`only the ${MAX_SESSIONS} most recently active sessions keep their caps`, async () => {
    const { fake } = wired();
    const first = agent("first");
    await call(fake, exec("t", {}, first), failed("EPERM: rename", "EPERM"));
    await edit("E-0001", { fix: "close the editor", status: "fixed" });
    await fake.preStep(first, 1, 1);
    expect(
      contexts(
        await call(
          fake,
          exec("t", {}, first),
          failed("EPERM: rename", "EPERM"),
        ),
      ),
    ).toHaveLength(1);
    for (let i = 0; i < MAX_SESSIONS; i++)
      await fake.preStep(agent(`s${i}`), 1, 1);
    // `first` was evicted: its fixed-entry budget starts again.
    await fake.preStep(first, 2, 1);
    expect(
      contexts(
        await call(
          fake,
          exec("t", {}, first),
          failed("EPERM: rename", "EPERM"),
        ),
      ),
    ).toHaveLength(1);
  });

  it("notices never exceed the caps, however long the entry", async () => {
    const { fake } = wired();
    const a = agent();
    await call(fake, exec("t", {}, a), failed("EPERM: rename", "EPERM"));
    await edit("E-0001", { fix: "x".repeat(2000), trigger: "y".repeat(2000) });
    const [notice] = contexts(
      await call(fake, exec("t", {}, a), failed("EPERM: rename", "EPERM")),
    );
    expect(Array.from(notice as string).length).toBeLessThanOrEqual(
      NOTICE_MAX_CHARS,
    );
    expect(withinCaps(notice as string)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Resolution detection (T14)

describe("resolution detection (T14)", () => {
  const a = agent();
  const shell = (command = "pnpm tsc") => exec("shell", { command }, a);
  const passed = ok("built\n[exit code: 0]");

  /** Let queued resolution steps, and the reads they queue, finish. */
  async function flush() {
    for (let i = 0; i < 4; i++) {
      await settled(filesIn(dir).errors);
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  /** A tool call that succeeds, through tools/result only, as the host emits it. */
  async function succeed(fake: Fake, e: Readonly<ToolExecution>) {
    await call(fake, e, passed);
    await flush();
  }

  it("a new entry that resolves is asked for its fix once, on the next step", async () => {
    const fake = applied();
    await fake.preStep(a, 1, 1);
    await call(fake, shell(), ok(TSC_A));
    await succeed(fake, shell());
    expect(entered(await fake.preStep(a, 1, 2))).toEqual([
      askFixText("E-0001"),
    ]);
    expect(askFixText("E-0001")).toBe(
      "[errkb] E-0001 looks resolved. Record the fix with err_record in one sentence so it can be reused.",
    );
    // It fails and resolves again: never asked twice in a session.
    await call(fake, shell(), ok(TSC_B));
    await succeed(fake, shell());
    expect(entered(await fake.preStep(a, 1, 3))).toEqual([]);
    expect(entered(await fake.preStep(a, 2, 1))).toEqual([]);
  });

  it("the prompt is a plugin notice, inside the caps", async () => {
    const fake = applied();
    await call(fake, shell(), ok(TSC_A));
    await succeed(fake, shell());
    const decision = await fake.preStep(a, 1, 1);
    expect(decision.kind === "enter" && decision.messages[0]).toMatchObject({
      role: "user",
      source: { kind: "plugin", plugin: "err-kb", form: "notice" },
    });
    expect(withinCaps(entered(decision)[0] as string)).toBe(true);
  });

  it("an entry that has a fix is not asked", async () => {
    const fake = applied();
    await call(fake, shell(), ok(TSC_A));
    await edit("E-0001", { fix: FIX });
    await call(fake, shell(), ok(TSC_B));
    await succeed(fake, shell());
    expect(entered(await fake.preStep(a, 1, 1))).toEqual([]);
  });

  it("captureFix: 'off' never asks", async () => {
    const fake = applied({ captureFix: "off" });
    await call(fake, shell(), ok(TSC_A));
    await succeed(fake, shell());
    expect(entered(await fake.preStep(a, 1, 1))).toEqual([]);
  });

  it("inject: 'off' never asks either", async () => {
    const fake = applied({ inject: "off" });
    await call(fake, shell(), ok(TSC_A));
    await succeed(fake, shell());
    expect(entered(await fake.preStep(a, 1, 1))).toEqual([]);
  });

  it("a wontfix entry is not asked", async () => {
    const fake = applied();
    await call(fake, shell(), ok(TSC_A));
    await edit("E-0001", { status: "wontfix" });
    await call(fake, shell(), ok(TSC_B));
    await succeed(fake, shell());
    expect(entered(await fake.preStep(a, 1, 1))).toEqual([]);
  });

  it("another command, or a call without an agent, resolves nothing", async () => {
    const fake = applied();
    await call(fake, shell(), ok(TSC_A));
    await succeed(fake, shell("pnpm test"));
    await succeed(fake, exec("shell", { command: "pnpm tsc" }, null));
    expect(entered(await fake.preStep(a, 1, 1))).toEqual([]);
    await succeed(fake, shell());
    expect(entered(await fake.preStep(a, 1, 2))).toEqual([
      askFixText("E-0001"),
    ]);
  });

  it("a tool failure resolves on the tool's next success", async () => {
    const fake = applied();
    await call(fake, exec("fetch", {}, a), failed("ECONNREFUSED 10.0.0.1"));
    await succeed(fake, exec("fetch", {}, a));
    expect(entered(await fake.preStep(a, 1, 1))).toEqual([
      askFixText("E-0001"),
    ]);
  });

  describe("the window: the rest of the turn plus the next one", () => {
    it("a success in the next turn resolves", async () => {
      const fake = applied();
      await fake.preStep(a, 1, 1);
      await call(fake, shell(), ok(TSC_A));
      await fake.preStep(a, 2, 1);
      await succeed(fake, shell());
      expect(entered(await fake.preStep(a, 2, 2))).toEqual([
        askFixText("E-0001"),
      ]);
    });

    it("a success two turns later does not", async () => {
      const fake = applied();
      await fake.preStep(a, 1, 1);
      await call(fake, shell(), ok(TSC_A));
      await fake.preStep(a, 2, 1);
      await fake.preStep(a, 3, 1);
      await succeed(fake, shell());
      expect(entered(await fake.preStep(a, 3, 2))).toEqual([]);
    });

    it("a new session lifecycle forgets the watches", async () => {
      const fake = applied();
      await call(fake, shell(), ok(TSC_A));
      await fake.sessionStart(a, "clear");
      await succeed(fake, shell());
      expect(entered(await fake.preStep(a, 1, 1))).toEqual([]);
    });
  });

  it("a recurrence cancels the resolution its success would have made", async () => {
    const fake = applied();
    // The same entry from two commands: TS2307 names its code, so the
    // command does not split it.
    await call(fake, shell("pnpm tsc"), ok(TSC_A));
    await call(fake, shell("npx tsc"), ok(TSC_B));
    expect(await effectiveHits()).toEqual([["E-0001", 2]]);
    await succeed(fake, shell("pnpm tsc"));
    expect(entered(await fake.preStep(a, 1, 1))).toEqual([]);
    await succeed(fake, shell("npx tsc"));
    expect(entered(await fake.preStep(a, 1, 2))).toEqual([
      askFixText("E-0001"),
    ]);
  });

  it("a prompt the step cap refuses rides a later step", async () => {
    const { fake } = wired();
    // A known turn error, and a tool entry that resolves: two notices for
    // one step.
    const turnError = () =>
      fake.emit("agent/error", {
        agent: a,
        error: Object.assign(new Error("context window exceeded"), {
          code: "CONTEXT_OVERFLOW",
        }),
      });
    turnError();
    await settled(filesIn(dir).errors);
    await edit("E-0001", { fix: "lower maxTokens" });
    turnError();
    await call(fake, shell(), ok(TSC_A));
    await succeed(fake, shell());
    const first = entered(await fake.preStep(a, 1, 1));
    expect(first).toHaveLength(1);
    expect(first[0]).toMatch(/^\[errkb\] E-0001 known/);
    expect(entered(await fake.preStep(a, 1, 2))).toEqual([
      askFixText("E-0002"),
    ]);
  });

  it("a prompt still waiting when its window closes is dropped unsaid", async () => {
    const { fake } = wired({}, { caps: { perTurn: 0 } });
    await fake.preStep(a, 1, 1);
    await call(fake, shell(), ok(TSC_A));
    await succeed(fake, shell());
    // Refused by the caps: kept, while the window is open.
    expect(entered(await fake.preStep(a, 1, 2))).toEqual([]);
    expect(entered(await fake.preStep(a, 2, 1))).toEqual([]);
    // Two turns on, it is dropped rather than offered again.
    expect(entered(await fake.preStep(a, 3, 1))).toEqual([]);
  });

  it("a resolved fix counts as a success, and lifts its suppression", async () => {
    const trust = memoryTrustStore({
      entries: {
        "E-0001": {
          injected: 2,
          recurredAfterInject: 2,
          succeeded: 0,
          fixSig: fixSig(FIX),
        },
      },
    });
    const { fake, injection } = wired({}, { trust });
    await call(fake, shell(), ok(TSC_A));
    await edit("E-0001", { fix: FIX });
    expect(injection.trust.level("E-0001", FIX)).toBe("suppressed");
    // Suppressed: the hit is silent, but still recorded and watched.
    await fake.preStep(a, 1, 1);
    expect(contexts(await call(fake, shell(), ok(TSC_B)))).toEqual([]);
    await succeed(fake, shell());
    expect(injection.trust.snapshot().entries["E-0001"]?.succeeded).toBe(1);
    expect(injection.trust.level("E-0001", FIX)).toBe("doubted");
    // An entry with a fix is not asked; the next hit speaks again.
    expect(entered(await fake.preStep(a, 1, 2))).toEqual([]);
    const [notice] = contexts(await call(fake, shell(), ok(TSC_A)));
    expect(notice).toContain(WORDING.doubted);
  });

  it("recordFix: the entry is fixed, speaks once, then stays silent", async () => {
    const { fake, recorder } = wired();
    await fake.preStep(a, 1, 1);
    await call(fake, shell(), ok(TSC_A));
    expect(await recorder.recordFix("E-0001", FIX)).toMatchObject({
      kind: "fixed",
      entry: { id: "E-0001", status: "fixed", fix: FIX },
    });
    await fake.preStep(a, 1, 2);
    const [notice] = contexts(await call(fake, shell(), ok(TSC_B)));
    expect(notice).toContain(`fix: ${FIX}`);
    await fake.preStep(a, 2, 1);
    expect(contexts(await call(fake, shell(), ok(TSC_A)))).toEqual([]);
    await fake.preStep(a, 3, 1);
    expect(contexts(await call(fake, shell(), ok(TSC_A)))).toEqual([]);
  });

  describe("guarded: nothing throws", () => {
    it("a malformed result is counted, and the session keeps working", async () => {
      const { fake, recorder } = wired();
      const malformed = {
        isError: true,
      } as unknown as Readonly<ToolExecutionResult>;
      expect(() =>
        fake.emit("tools/result", exec("t", {}, a), malformed),
      ).not.toThrow();
      expect(recorder.stats.failures).toBeGreaterThan(0);
      await call(fake, shell(), ok(TSC_A));
      await succeed(fake, shell());
      expect(entered(await fake.preStep(a, 1, 1))).toEqual([
        askFixText("E-0001"),
      ]);
    });

    it("a read that fails while resolving is counted, and asks nothing", async () => {
      const real = createRecorder({
        files: filesIn(dir),
        logger: { warn: () => undefined },
      });
      let failures = 0;
      const recorder: Recorder = {
        ...real,
        entries: () => Promise.reject(new Error("read broke")),
        fail: () => {
          failures++;
        },
      };
      const { fake } = wired({}, { recorder });
      await call(fake, shell(), ok(TSC_A));
      await succeed(fake, shell());
      expect(failures).toBe(1);
      expect(entered(await fake.preStep(a, 1, 1))).toEqual([]);
    });

    it("an unreadable knowledge base or a vanished entry asks nothing", async () => {
      const real = createRecorder({
        files: filesIn(dir),
        logger: { warn: () => undefined },
      });
      let reads = 0;
      const recorder: Recorder = {
        ...real,
        entries: async () => (reads++ === 0 ? undefined : []),
      };
      const { fake } = wired({}, { recorder });
      await call(fake, shell(), ok(TSC_A));
      await succeed(fake, shell());
      await call(fake, shell(), ok(TSC_B));
      await succeed(fake, shell());
      expect(reads).toBe(2);
      expect(entered(await fake.preStep(a, 1, 1))).toEqual([]);
    });

    it("a write that does not name an entry watches nothing", async () => {
      const real = createRecorder({
        files: filesIn(dir),
        logger: { warn: () => undefined },
      });
      const recorder: Recorder = {
        ...real,
        toolResult: () => Promise.resolve({ kind: "timeout" }),
      };
      const { fake } = wired({}, { recorder });
      await call(fake, shell(), ok(TSC_A));
      await succeed(fake, shell());
      expect(entered(await fake.preStep(a, 1, 1))).toEqual([]);
    });
  });
});
