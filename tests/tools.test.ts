import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Context } from "@deepseek-ai/cordis";
import type { ContentBlock } from "@deepseek-ai/dsh-llm";
import { ToolArgsError, validateJsonSchemaValue } from "@deepseek-ai/dsh-tools";
import type {
  PostToolDecision,
  ToolDefinition,
  ToolExecution,
  ToolExecutionResult,
  ToolRunContext,
} from "@deepseek-ai/dsh-tools";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Config, apply } from "../src/index";
import { fixSig, memoryTrustStore } from "../src/inject";
import { filesIn } from "../src/paths";
import {
  createInjection,
  createRecorder,
  registerInjection,
  settled,
} from "../src/plugin";
import type { Injection, Recorder } from "../src/plugin";
import { signature } from "../src/signature";
import { createStore, nodeStoreFs, parseDocument } from "../src/store";
import type { StoreFs } from "../src/store";
import {
  ASSUMED_DIAGNOSIS_TOKENS,
  DEFAULT_LIST_LIMIT,
  MAX_LIST_LIMIT,
  TOOL_NAMES,
  createTools,
  estimateSaved,
  registerTools,
} from "../src/tools";
import type { ToolsDeps } from "../src/tools";

// ---------------------------------------------------------------------------
// Fixtures

let dir: string;

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "errkb-tools-"));
});

afterEach(async () => {
  await settled(filesIn(dir).errors);
  await rm(dir, { recursive: true, force: true });
});

type Listener = (...args: unknown[]) => unknown;

/** A context that records hooks and tools, and plays the host for both. */
function fakeCtx(options: { registerThrows?: string } = {}) {
  const listeners = new Map<string, Listener[]>();
  const tools = new Map<string, ToolDefinition>();
  const warnings: unknown[][] = [];
  const ctx = {
    on(name: string, listener: Listener) {
      listeners.set(name, [...(listeners.get(name) ?? []), listener]);
      return () => true;
    },
    tools: {
      register(tool: ToolDefinition) {
        if (tool.name === options.registerThrows)
          throw new Error(`duplicate tool ${tool.name}`);
        tools.set(tool.name, tool);
        return () => undefined;
      },
    },
    systemPrompt: { section: () => () => undefined },
    logger: {
      info: () => undefined,
      warn: (...args: unknown[]) => {
        warnings.push(args);
      },
      error: () => undefined,
      debug: () => undefined,
    },
  };
  return {
    ctx: ctx as unknown as Context,
    listeners,
    tools,
    warnings,
    emit(name: string, ...args: unknown[]) {
      for (const listener of listeners.get(name) ?? []) listener(...args);
    },
    postExecute(
      e: Readonly<ToolExecution>,
      r: Readonly<ToolExecutionResult>,
    ): Promise<PostToolDecision> {
      const [listener] = listeners.get("tools/post-execute") ?? [];
      return (listener as Listener)(e, r, async () => ({
        kind: "accept",
      })) as Promise<PostToolDecision>;
    },
  };
}

type Fake = ReturnType<typeof fakeCtx>;

const exec = (name: string, args: unknown = {}, session = "s1") =>
  ({
    name,
    arguments: args,
    agent: { id: session, inject: () => undefined },
  }) as unknown as Readonly<ToolExecution>;

const failed = (message: string) =>
  ({
    isError: true,
    error: { message },
    content: [{ type: "text", text: message }],
  }) as unknown as Readonly<ToolExecutionResult>;

const runContext = (session: string | null = "s1") =>
  (session === null
    ? {}
    : { agent: { id: session } }) as unknown as ToolRunContext;

interface Called {
  value: Record<string, unknown>;
  text: string;
}

/**
 * Call a tool the way the registry does: execute, check the value against the
 * declared output schema, render it.
 */
async function callTool(
  tools: ReadonlyMap<string, ToolDefinition> | readonly ToolDefinition[],
  name: string,
  args: Record<string, unknown>,
  session: string | null = "s1",
): Promise<Called> {
  const tool =
    tools instanceof Map
      ? tools.get(name)
      : (tools as readonly ToolDefinition[]).find((t) => t.name === name);
  if (tool === undefined) throw new Error(`no tool ${name}`);
  const value = (await tool.execute(args, runContext(session))) as Record<
    string,
    unknown
  >;
  expect(validateJsonSchemaValue(tool.output.schema, value, "")).toEqual([]);
  const content = tool.output.render(args, value as never) as ContentBlock[];
  expect(content).toHaveLength(1);
  return { value, text: (content[0] as { text: string }).text };
}

async function entries() {
  const text = await readFile(filesIn(dir).errors, "utf8");
  return parseDocument(text).blocks.map((b) => b.entry);
}

/** One failing tool call, through both pipeline hooks. */
async function fail(fake: Fake, message: string, tool = "bash") {
  const e = exec(tool);
  const r = failed(message);
  const decision = await fake.postExecute(e, r);
  fake.emit("tools/result", e, r);
  await settled(filesIn(dir).errors);
  return decision;
}

const contextText = (decision: PostToolDecision) =>
  (decision.additionalContexts ?? [])
    .flatMap((m) => m.content as Array<{ text: string }>)
    .map((b) => b.text)
    .join("\n");

/** A recorder, an injection layer and the tools, wired by hand. */
function wired(
  options: {
    fs?: StoreFs;
    tools?: Partial<ToolsDeps["options"]>;
    recorder?: Recorder;
    injection?: ToolsDeps["injection"];
  } = {},
) {
  const warnings: unknown[][] = [];
  const recorder =
    options.recorder ??
    createRecorder({
      files: filesIn(dir),
      logger: { warn: (...args) => warnings.push(args) },
      ...(options.fs === undefined ? {} : { fs: options.fs }),
    });
  const injection = createInjection({ recorder });
  const tools = createTools({
    recorder,
    injection: options.injection ?? injection,
    kbDir: dir,
    ...(options.tools === undefined ? {} : { options: options.tools }),
  });
  return { recorder, injection, tools, warnings };
}

/** Seed entries straight through the store. */
async function seed(
  ...inputs: Array<
    Partial<Parameters<ReturnType<typeof createStore>["append"]>[0]>
  >
) {
  const store = createStore(filesIn(dir));
  for (const [n, input] of inputs.entries())
    await store.append({
      title: `[tool:bash] failure ${n + 1}`,
      signature: `00000000000${n + 1}`,
      category: "tool / bash",
      meta: { cat: "tool" },
      raw: `failure ${n + 1}`,
      ...input,
    });
}

const PNPM_EPERM =
  "EPERM: operation not permitted, rename '/srv/app/node_modules/.pnpm/x'";

// ---------------------------------------------------------------------------
// Acceptance (T15)

describe("acceptance: the model records a fix and the next failure is told it", () => {
  it("err_record → ERRORS.md fixed; err_lookup by message; repeat → known-fix notice", async () => {
    const fake = fakeCtx();
    apply(fake.ctx, Config({ kbDir: dir }));
    expect([...fake.tools.keys()]).toEqual([...TOOL_NAMES]);

    // The failure is captured as E-0001, without a fix.
    await fail(fake, PNPM_EPERM);
    expect((await entries())[0]).toMatchObject({ id: "E-0001", fix: "" });

    // The model records the fix.
    const FIX =
      "Close the editor holding node_modules, then rerun pnpm install.";
    const recorded = await callTool(fake.tools, "err_record", {
      id: "E-0001",
      fix: FIX,
    });
    expect(recorded.value).toEqual({
      id: "E-0001",
      created: false,
      status: "fixed",
      hasFix: true,
    });
    expect(recorded.text).toBe("Updated E-0001 (status fixed).");
    const text = await readFile(filesIn(dir).errors, "utf8");
    expect(text).toContain(`- Fix:\n  ${FIX}`);
    expect(text).toContain("- Status: `fixed`");

    // Looking the original message up finds it, fix included.
    const looked = await callTool(fake.tools, "err_lookup", {
      query: PNPM_EPERM,
    });
    expect(looked.value).toMatchObject({
      via: "exact",
      entry: { id: "E-0001", status: "fixed", fix: FIX },
    });
    expect(looked.text).toContain(`fix: ${FIX}`);

    // The same failure again: the notice carries the fix.
    const decision = await fail(fake, PNPM_EPERM);
    const notice = contextText(decision);
    expect(notice).toContain("[errkb] E-0001 known");
    expect(notice).toContain(FIX);

    // err_stats counts that notice.
    const stats = await callTool(fake.tools, "err_stats", { scope: "session" });
    expect(stats.value).toMatchObject({
      entries: 1,
      hits: 2,
      notices: 1,
      fixNotices: 1,
      kbDir: dir,
      suppressed: [],
    });
  });
});

// ---------------------------------------------------------------------------
// Registration

describe("registerTools", () => {
  it("registers all five; a refused one is counted and the rest still register", () => {
    const fake = fakeCtx({ registerThrows: "err_list" });
    const { recorder, injection } = wired();
    registerTools(fake.ctx, { recorder, injection, kbDir: dir });
    expect([...fake.tools.keys()]).toEqual(
      TOOL_NAMES.filter((n) => n !== "err_list"),
    );
    expect(recorder.stats.failures).toBe(1);
  });

  it("declares a description and only the documented parameters", () => {
    const { tools } = wired();
    const params = Object.fromEntries(
      tools.map((t) => [
        t.name,
        Object.keys((t.parameters as { properties: object }).properties),
      ]),
    );
    expect(params).toEqual({
      err_lookup: ["query", "full"],
      err_record: ["id", "message", "fix", "status", "note", "category"],
      err_list: ["cat", "status", "limit"],
      err_forget: ["id", "reason"],
      err_stats: ["scope"],
    });
    for (const tool of tools)
      expect(tool.description.length).toBeGreaterThan(20);
  });

  it("read-only tools may run in parallel; writes may not", () => {
    const { tools } = wired();
    const args = { query: "x", id: "E-0001" };
    const safe = tools.filter((t) => t.isConcurrencySafe?.(args) === true);
    expect(safe.map((t) => t.name)).toEqual([
      "err_lookup",
      "err_list",
      "err_stats",
    ]);
  });

  it("arguments of the wrong type are refused before the body runs", async () => {
    const { tools } = wired();
    const record = tools.find((t) => t.name === "err_record") as ToolDefinition;
    await expect(
      record.execute({ id: "E-0001", status: "done" }, runContext()),
    ).rejects.toBeInstanceOf(ToolArgsError);
  });
});

// ---------------------------------------------------------------------------
// err_lookup

describe("err_lookup", () => {
  it("finds by ID in any width, by fingerprint, and by text", async () => {
    await seed(
      { fix: "do one", raw: PNPM_EPERM, title: "[tool:bash] EPERM" },
      { signature: "abcdef012345" },
    );
    const { tools } = wired();
    const byId = await callTool(tools, "err_lookup", { query: "e-1" });
    expect(byId.value).toMatchObject({ via: "id", entry: { id: "E-0001" } });
    expect(byId.value.entry).not.toHaveProperty("raw");
    expect(byId.text).toBe(
      [
        "E-0001 · [tool:bash] EPERM",
        "category: tool / bash · hits: 1 · status: open · matched by id",
        "fix: do one",
      ].join("\n"),
    );
    const bySig = await callTool(tools, "err_lookup", {
      query: "ABCDEF012345",
    });
    expect(bySig.value).toMatchObject({
      via: "fingerprint",
      entry: { id: "E-0002", fix: "" },
    });
    expect(bySig.text).toContain("fix: (none recorded)");
    // A 12-hex query that is no fingerprint is matched as text.
    const hex = await callTool(tools, "err_lookup", { query: "0123456789ab" });
    expect(hex.value).toMatchObject({ via: "none", entry: null });
  });

  it("text without a category tries every category; an exact hit beats an earlier fuzzy one", async () => {
    await seed(
      { raw: "cannot open file config yaml now" },
      {
        raw: "cannot open file config yaml",
        category: "agent",
        meta: { cat: "agent" },
        signature: signature("agent", "cannot open file config yaml"),
      },
    );
    const { tools } = wired();
    expect(
      (
        await callTool(tools, "err_lookup", {
          query: "cannot open file config yaml",
        })
      ).value,
    ).toMatchObject({ via: "exact", entry: { id: "E-0002" } });
  });

  it("between two fuzzy hits in different categories the closer one wins", async () => {
    await seed(
      { raw: "one two three four five six seven extra" },
      {
        raw: "one two three four five six seven eight other",
        category: "agent",
        meta: { cat: "agent" },
      },
    );
    const { tools } = wired();
    expect(
      (
        await callTool(tools, "err_lookup", {
          query: "one two three four five six seven eight",
        })
      ).value,
    ).toMatchObject({ via: "fuzzy", entry: { id: "E-0002" } });
  });

  it("full: true adds the redacted raw sample", async () => {
    await seed({ raw: PNPM_EPERM });
    const { tools } = wired();
    const found = await callTool(tools, "err_lookup", {
      query: "E-0001",
      full: true,
    });
    const raw = (found.value.entry as { raw: string }).raw;
    expect(raw).toContain("EPERM");
    expect(raw).not.toContain("/srv/app");
    expect(found.text).toContain(`raw:\n${raw}`);
  });

  it("a miss returns null and the three closest by Jaccard", async () => {
    await seed(
      { raw: "cannot find module alpha beta" },
      { raw: "cannot find module alpha" },
      { raw: "nothing in common here" },
      { raw: "cannot find" },
      { raw: "cannot" },
    );
    const { tools } = wired();
    const miss = await callTool(tools, "err_lookup", {
      query: "cannot find module alpha gamma delta",
    });
    expect(miss.value).toMatchObject({ via: "none", entry: null });
    const closest = miss.value.closest as Array<{
      id: string;
      similarity: number;
    }>;
    expect(closest.map((c) => c.id)).toEqual(["E-0002", "E-0001", "E-0004"]);
    expect(closest[0]?.similarity).toBe(0.67);
    expect(miss.text.split("\n")[0]).toBe(
      'No entry matches "cannot find module alpha gamma delta". Closest:',
    );
    expect(miss.text).toContain(
      "E-0002 [tool:bash] failure 2 (similarity 0.67)",
    );
  });

  it("a miss with nothing close says so", async () => {
    const { tools } = wired();
    const miss = await callTool(tools, "err_lookup", { query: "zzz" });
    expect(miss.value).toEqual({
      query: "zzz",
      via: "none",
      entry: null,
      closest: [],
    });
    expect(miss.text).toBe('No entry matches "zzz".');
  });

  it("an unknown ID, an empty query and an unreadable file are errors", async () => {
    await seed({});
    const { tools } = wired();
    expect(
      (await callTool(tools, "err_lookup", { query: "E-42" })).value,
    ).toEqual({ error: "no entry E-0042 (it may have been archived)" });
    const empty = await callTool(tools, "err_lookup", { query: "  " });
    expect(empty.text).toBe("err_lookup: query is empty");
    await writeFile(filesIn(dir).errors, "## E-0001 · broken\n");
    expect(
      (await callTool(tools, "err_lookup", { query: "x" })).value.error,
    ).toMatch(/could not read ERRORS\.md/);
  });

  it("respects idPrefix and idWidth", async () => {
    const store = createStore(filesIn(dir), { idPrefix: "BUG", idWidth: 2 });
    await store.append({
      title: "t",
      signature: "000000000001",
      category: "x",
    });
    const { tools } = wired({
      tools: { idPrefix: "BUG", idWidth: 2 },
      recorder: createRecorder({
        files: filesIn(dir),
        logger: { warn: () => undefined },
        options: { idPrefix: "BUG", idWidth: 2 },
      }),
    });
    expect(
      (await callTool(tools, "err_lookup", { query: "bug1" })).value,
    ).toMatchObject({ entry: { id: "BUG01" } });
    expect(
      (await callTool(tools, "err_lookup", { query: "BUG7" })).value.error,
    ).toBe("no entry BUG07 (it may have been archived)");
    const lookup = tools.find((t) => t.name === "err_lookup") as ToolDefinition;
    expect(JSON.stringify(lookup.parameters)).toContain("BUG07");
  });
});

// ---------------------------------------------------------------------------
// err_record

describe("err_record", () => {
  it("needs exactly one of id and message", async () => {
    const { tools } = wired();
    for (const args of [{}, { id: "E-1", message: "m" }, { id: " " }])
      expect(await callTool(tools, "err_record", args)).toEqual({
        value: { error: "give exactly one of id and message" },
        text: "err_record: give exactly one of id and message",
      });
  });

  it("with id: needs something to record, an existing entry, a non-empty fix", async () => {
    await seed({});
    const { tools } = wired();
    expect(
      (await callTool(tools, "err_record", { id: "E-0001" })).value.error,
    ).toBe("nothing to record: give fix, status or note");
    expect(
      (await callTool(tools, "err_record", { id: "E-9", fix: "x" })).value
        .error,
    ).toBe("no entry E-0009");
    expect(
      (await callTool(tools, "err_record", { id: "E-0001", fix: " " })).value
        .error,
    ).toBe("fix is empty");
  });

  it("with id: status and notes go through the write path, notes accumulate", async () => {
    await seed({ notes: "earlier" });
    const { tools } = wired();
    const first = await callTool(tools, "err_record", {
      id: "E-1",
      status: "wontfix",
      note: "upstream bug",
    });
    expect(first.value).toEqual({
      id: "E-0001",
      created: false,
      status: "wontfix",
      hasFix: false,
    });
    expect(first.text).toBe(
      "Updated E-0001 (status wontfix, no fix recorded).",
    );
    await callTool(tools, "err_record", { id: "E-0001", note: "still" });
    expect((await entries())[0]).toMatchObject({
      status: "wontfix",
      notes: "earlier\nupstream bug\nstill",
    });
  });

  it("with id: a fix sets fixed unless a status is given too, and is redacted", async () => {
    await seed({}, {});
    const { tools } = wired();
    const secret = "sk-" + "b".repeat(24);
    await callTool(tools, "err_record", {
      id: "E-0001",
      fix: `set OPENAI_KEY=${secret}`,
    });
    const both = await callTool(tools, "err_record", {
      id: "E-0002",
      fix: "pin the version",
      status: "open",
    });
    expect(both.value).toMatchObject({ status: "open", hasFix: true });
    const [one, two] = await entries();
    expect(one).toMatchObject({ status: "fixed" });
    expect(one?.fix).not.toContain(secret);
    expect(two).toMatchObject({ status: "open", fix: "pin the version" });
  });

  it("with message: an exact hit updates that entry, matched across categories", async () => {
    await seed({
      raw: PNPM_EPERM,
      title: "[tool:bash] EPERM",
      signature: signature("tool", PNPM_EPERM),
    });
    const { tools } = wired();
    const hit = await callTool(tools, "err_record", {
      message: PNPM_EPERM,
      fix: "close the editor",
    });
    expect(hit.value).toEqual({
      id: "E-0001",
      created: false,
      status: "fixed",
      hasFix: true,
    });
    expect(hit.text).toBe("Updated E-0001 (status fixed).");
    expect(await entries()).toHaveLength(1);
    expect((await entries())[0]).toMatchObject({
      hits: 1,
      fix: "close the editor",
      status: "fixed",
    });
  });

  it("with message: an exact hit with nothing else to record just names it", async () => {
    await seed({
      raw: PNPM_EPERM,
      signature: signature("tool", PNPM_EPERM),
      fix: "known",
    });
    const { tools } = wired();
    expect(
      (await callTool(tools, "err_record", { message: PNPM_EPERM })).value,
    ).toEqual({ id: "E-0001", created: false, status: "open", hasFix: true });
  });

  it("with message: a fuzzy hit writes nothing and names the candidate", async () => {
    // The seed's signature is made up, so this is only a fuzzy hit in `tool`.
    await seed({ raw: PNPM_EPERM, signature: "ffffffffffff" });
    const before = await readFile(filesIn(dir).errors, "utf8");
    const { tools } = wired();
    for (const args of [
      { message: PNPM_EPERM, fix: "close the editor" },
      { message: PNPM_EPERM, status: "wontfix", note: "n" },
      { message: PNPM_EPERM },
    ]) {
      const refused = await callTool(tools, "err_record", args);
      expect(refused.value).toEqual({
        error:
          'closest match is E-0001 (approximate, by fuzzy); nothing was written. Call err_record with id: "E-0001" to confirm, or reword message',
      });
      expect(refused.text).toBe(`err_record: ${refused.value.error}`);
    }
    expect(await readFile(filesIn(dir).errors, "utf8")).toBe(before);

    // Confirming the candidate by ID then records the fix.
    const confirmed = await callTool(tools, "err_record", {
      id: "E-0001",
      fix: "close the editor",
    });
    expect(confirmed.value).toEqual({
      id: "E-0001",
      created: false,
      status: "fixed",
      hasFix: true,
    });
    expect((await entries())[0]).toMatchObject({
      fix: "close the editor",
      status: "fixed",
    });
  });

  it("with message: a code-only hit writes nothing and names the candidate", async () => {
    await seed({
      raw: "EBUSY: resource busy",
      meta: { cat: "tool", code: "EBUSY" },
    });
    const before = await readFile(filesIn(dir).errors, "utf8");
    const { tools } = wired();
    const refused = await callTool(tools, "err_record", {
      message: "EBUSY: lock held by vite",
      category: "tool",
      fix: "install the adapter",
    });
    expect(refused.value.error).toBe(
      'closest match is E-0001 (approximate, by code); nothing was written. Call err_record with id: "E-0001" to confirm, or reword message',
    );
    expect(await readFile(filesIn(dir).errors, "utf8")).toBe(before);
    expect(await entries()).toHaveLength(1);
  });

  it("with message: a miss appends a new entry, redacted, under agent or the category given", async () => {
    await seed({});
    const { tools } = wired();
    const created = await callTool(tools, "err_record", {
      message:
        "Traceback (most recent call last):\n  ...\nKeyError: 'token' at /srv/kris/app.py",
      fix: "set TOKEN first",
      note: "seen once",
    });
    expect(created.value).toEqual({
      id: "E-0002",
      created: true,
      status: "fixed",
      hasFix: true,
    });
    expect(created.text).toBe("Created E-0002 (status fixed).");
    const open = await callTool(tools, "err_record", {
      message: "flaky network",
      category: "llm",
    });
    expect(open.value).toMatchObject({ id: "E-0003", status: "open" });
    const wontfix = await callTool(tools, "err_record", {
      message: "EACCES: permission denied",
      status: "wontfix",
    });
    expect(wontfix.value).toMatchObject({ status: "wontfix", hasFix: false });

    const [, two, three, four] = await entries();
    expect(two).toMatchObject({
      title: "[agent] KeyError: 'token' at <path>/app.py",
      category: "agent",
      meta: { cat: "agent" },
      fix: "set TOKEN first",
      notes: "seen once",
      status: "fixed",
    });
    expect(two?.raw).not.toContain("/srv/kris");
    expect(three).toMatchObject({
      category: "llm",
      title: "[llm] flaky network",
    });
    expect(four).toMatchObject({ meta: { cat: "agent", code: "EACCES" } });
    // Recording the same message again finds the new entry.
    expect(
      (
        await callTool(tools, "err_record", {
          message: "flaky network",
          category: "llm",
        })
      ).value,
    ).toMatchObject({ id: "E-0003", created: false });
  });

  it("reports a busy store and a failed write as errors", async () => {
    await seed({});
    const files = filesIn(dir);
    const real = nodeStoreFs();
    const busy: StoreFs = {
      ...real,
      createExclusive: async () => false,
      readFile: async (path) =>
        path === files.lock ? "someone-else" : real.readFile(path),
      mtimeMs: async (path) =>
        path === files.lock ? Date.now() : real.mtimeMs(path),
    };
    const { tools } = wired({ fs: busy });
    for (const args of [
      { id: "E-0001", fix: "x" },
      { id: "E-0001", note: "x" },
      { message: "new error" },
    ])
      expect((await callTool(tools, "err_record", args)).value).toEqual({
        error: "the knowledge base is busy; try again",
      });
  });

  it("a write that throws is an error value, not a throw", async () => {
    await seed({});
    const real = nodeStoreFs();
    const broken: StoreFs = {
      ...real,
      rename: async () => {
        throw Object.assign(new Error("disk full"), { code: "ENOSPC" });
      },
    };
    const { tools, recorder } = wired({ fs: broken });
    expect(
      (await callTool(tools, "err_record", { id: "E-0001", fix: "x" })).value,
    ).toEqual({ error: "could not write ERRORS.md (see the plugin log)" });
    expect(
      (await callTool(tools, "err_record", { id: "E-0001", note: "x" })).value
        .error,
    ).toBe("could not write ERRORS.md: disk full");
    expect(recorder.stats.failures).toBe(2);
  });

  it("an entry that disappears between the read and the write is unknown", async () => {
    await seed({});
    const { recorder } = wired();
    const vanishing: Recorder = {
      ...recorder,
      recordFix: async () => ({ kind: "unknown" }),
      write: async () => ({ kind: "done", value: undefined }) as never,
    };
    const { tools } = wired({ recorder: vanishing });
    expect(
      (await callTool(tools, "err_record", { id: "E-0001", fix: "x" })).value,
    ).toEqual({ error: "no entry E-0001" });
    expect(
      (await callTool(tools, "err_record", { id: "E-0001", note: "x" })).value,
    ).toEqual({ error: "no entry E-0001" });
    const noEntries: Recorder = {
      ...recorder,
      write: async () =>
        ({ kind: "done", value: { id: "E-0005", created: false } }) as never,
    };
    const second = wired({ recorder: noEntries });
    expect(
      (await callTool(second.tools, "err_record", { message: "m" })).value,
    ).toEqual({ error: "no entry E-0005" });
    const unreadable: Recorder = {
      ...noEntries,
      entries: async () => undefined,
    };
    const third = wired({ recorder: unreadable });
    expect(
      (await callTool(third.tools, "err_record", { message: "m" })).value.error,
    ).toMatch(/could not read/);
    expect(
      (await callTool(third.tools, "err_record", { id: "E-1", fix: "f" })).value
        .error,
    ).toMatch(/could not read/);
  });

  it("anything unexpected becomes an internal error value, counted", async () => {
    const { recorder } = wired();
    const throwing: Recorder = {
      ...recorder,
      write: () => {
        throw new TypeError("boom");
      },
    };
    const { tools } = wired({ recorder: throwing });
    expect(
      (await callTool(tools, "err_record", { message: "m" })).value,
    ).toEqual({ error: "internal error: boom" });
    expect(recorder.stats.failures).toBe(1);
  });

  it("two calls about the same new error append it once", async () => {
    // The matching write reads the document itself, so a hit and an append
    // can never race another err_record.
    const { tools } = wired();
    const calls = await Promise.all([
      callTool(tools, "err_record", { message: "same new error" }),
      callTool(tools, "err_record", { message: "same new error" }),
    ]);
    expect(calls.map((c) => c.value.created).sort()).toEqual([false, true]);
    expect(await entries()).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// err_list

describe("err_list", () => {
  it("lists ID, title and hits, filtered by category and status, limited", async () => {
    await seed(
      {},
      { category: "command-exit / bash", meta: { cat: "command-exit" } },
      { status: "fixed", fix: "x" },
    );
    const { tools } = wired();
    const all = await callTool(tools, "err_list", {});
    expect(all.value).toEqual({
      entries: [
        { id: "E-0001", title: "[tool:bash] failure 1", hits: 1 },
        { id: "E-0002", title: "[tool:bash] failure 2", hits: 1 },
        { id: "E-0003", title: "[tool:bash] failure 3", hits: 1 },
      ],
      total: 3,
    });
    expect(all.text).toBe(
      [
        "E-0001 (1) [tool:bash] failure 1",
        "E-0002 (1) [tool:bash] failure 2",
        "E-0003 (1) [tool:bash] failure 3",
        "3 of 3 shown.",
      ].join("\n"),
    );
    const ids = async (args: Record<string, unknown>) =>
      (
        (await callTool(tools, "err_list", args)).value.entries as Array<{
          id: string;
        }>
      ).map((e) => e.id);
    expect(await ids({ cat: "TOOL" })).toEqual(["E-0001", "E-0003"]);
    expect(await ids({ cat: "command-exit / bash" })).toEqual(["E-0002"]);
    expect(await ids({ status: "fixed" })).toEqual(["E-0003"]);
    expect(await ids({ cat: "tool", status: "open" })).toEqual(["E-0001"]);
    expect(await ids({ limit: 1 })).toEqual(["E-0001"]);
    expect((await callTool(tools, "err_list", { limit: 1 })).text).toContain(
      "1 of 3 shown.",
    );
  });

  it("defaults to 20 and never returns more than the maximum", async () => {
    await seed(...Array.from({ length: 25 }, () => ({})));
    const { tools } = wired();
    const listed = async (args: Record<string, unknown>) =>
      ((await callTool(tools, "err_list", args)).value.entries as unknown[])
        .length;
    expect(await listed({})).toBe(DEFAULT_LIST_LIMIT);
    expect(await listed({ limit: MAX_LIST_LIMIT + 1 })).toBe(25);
  });

  it("an empty knowledge base, a bad limit and an unreadable file", async () => {
    const { tools } = wired();
    expect((await callTool(tools, "err_list", {})).text).toBe("No entries.");
    expect((await callTool(tools, "err_list", { limit: 0 })).text).toBe(
      "err_list: limit must be a positive integer",
    );
    await writeFile(filesIn(dir).errors, "## E-0001 · broken\n");
    expect((await callTool(tools, "err_list", {})).value.error).toMatch(
      /could not read/,
    );
  });
});

// ---------------------------------------------------------------------------
// err_forget

describe("err_forget", () => {
  it("moves the entry to the archive with the reason", async () => {
    await seed({}, {});
    const { tools } = wired();
    const forgot = await callTool(tools, "err_forget", {
      id: "e-2",
      reason: "misjudged: not an error",
    });
    expect(forgot.value).toEqual({ id: "E-0002", archived: true });
    expect(forgot.text).toBe("Archived E-0002 to ERRORS.archive.md.");
    expect((await entries()).map((e) => e.id)).toEqual(["E-0001"]);
    const archive = await readFile(filesIn(dir).archive, "utf8");
    expect(archive).toContain("## E-0002 · ");
    expect(archive).toMatch(
      /Archived [0-9-]+ [0-9:]+: misjudged: not an error/,
    );
    // Gone from lookups, and its ID is not reused.
    expect(
      (await callTool(tools, "err_lookup", { query: "E-0002" })).value.error,
    ).toMatch(/archived/);
    expect(
      (await callTool(tools, "err_record", { message: "brand new" })).value,
    ).toMatchObject({ id: "E-0003", created: true });
  });

  it("an empty or unknown ID, a busy store and a vanished entry are errors", async () => {
    await seed({});
    const { tools, recorder } = wired();
    expect((await callTool(tools, "err_forget", { id: "" })).value.error).toBe(
      "id is empty",
    );
    expect((await callTool(tools, "err_forget", { id: "E-0009" })).text).toBe(
      "err_forget: no entry E-0009",
    );
    expect(
      (await callTool(tools, "err_forget", { id: "first one" })).value.error,
    ).toBe("no entry first one");
    const busy = wired({
      recorder: { ...recorder, write: async () => ({ kind: "timeout" }) },
    });
    expect(
      (await callTool(busy.tools, "err_forget", { id: "E-0001" })).value.error,
    ).toBe("the knowledge base is busy; try again");
    const vanished = wired({
      recorder: {
        ...recorder,
        write: async () => ({ kind: "done", value: undefined }) as never,
      },
    });
    expect(
      (await callTool(vanished.tools, "err_forget", { id: "E-0001" })).value
        .error,
    ).toBe("no entry E-0001");
    await writeFile(filesIn(dir).errors, "## E-0001 · broken\n");
    expect(
      (await callTool(tools, "err_forget", { id: "E-0001" })).value.error,
    ).toMatch(/could not read/);
  });
});

// ---------------------------------------------------------------------------
// err_stats

describe("err_stats", () => {
  it("counts entries, hits, open entries without a fix and the path", async () => {
    await seed({}, { fix: "x", status: "fixed" }, { status: "wontfix" });
    const { tools } = wired();
    const stats = await callTool(tools, "err_stats", {});
    expect(stats.value).toEqual({
      scope: "all",
      kbDir: dir,
      entries: 3,
      hits: 3,
      openWithoutFix: 1,
      notices: 0,
      fixNotices: 0,
      noticeTokens: 0,
      estimatedTokensSaved: 0,
      suppressed: [],
    });
    expect(stats.text).toBe(
      [
        `Knowledge base: ${dir}`,
        "Entries: 3 · hits: 3 · open without a fix: 1",
        "Notices (all sessions): 0, 0 with a fix, 0 tokens",
        `Estimated tokens saved: 0 (estimate: 0 fix notices × ${ASSUMED_DIAGNOSIS_TOKENS} − 0 notice tokens)`,
        "Distrusted fixes, not injected: none",
      ].join("\n"),
    );
  });

  it("separates this session's notices from everyone's", async () => {
    const fake = fakeCtx();
    apply(fake.ctx, Config({ kbDir: dir }));
    await fail(fake, PNPM_EPERM);
    await callTool(fake.tools, "err_record", { id: "E-0001", fix: "close it" });
    await fail(fake, PNPM_EPERM);
    const mine = await callTool(
      fake.tools,
      "err_stats",
      { scope: "session" },
      "s1",
    );
    const other = await callTool(
      fake.tools,
      "err_stats",
      { scope: "session" },
      "s2",
    );
    const all = await callTool(
      fake.tools,
      "err_stats",
      { scope: "all" },
      undefined,
    );
    expect(mine.value).toMatchObject({ notices: 1, fixNotices: 1 });
    expect(other.value).toMatchObject({ notices: 0, fixNotices: 0 });
    expect(all.value).toMatchObject({ notices: 1, fixNotices: 1 });
    const tokens = mine.value.noticeTokens as number;
    expect(tokens).toBeGreaterThan(0);
    expect(mine.value.estimatedTokensSaved).toBe(
      ASSUMED_DIAGNOSIS_TOKENS - tokens,
    );
    expect(mine.text).toContain("Notices (this session): 1, 1 with a fix");
  });

  it("names the IDs whose fix trust is suppressed", async () => {
    await seed({ fix: "bad fix", status: "fixed" }, { fix: "good" }, {});
    const trust = memoryTrustStore({
      entries: {
        "E-0001": {
          injected: 2,
          recurredAfterInject: 2,
          succeeded: 0,
          fixSig: fixSig("bad fix"),
        },
      },
    });
    const { recorder } = wired();
    const injection: Injection = createInjection({ recorder, trust });
    const { tools } = wired({ recorder, injection });
    const stats = await callTool(tools, "err_stats", {});
    expect(stats.value.suppressed).toEqual(["E-0001"]);
    expect(stats.text).toContain("Distrusted fixes, not injected: E-0001");
  });

  it("scope session without a session, and an unreadable file, are errors", async () => {
    const { tools } = wired();
    expect(
      (await callTool(tools, "err_stats", { scope: "session" }, null)).value
        .error,
    ).toBe("scope session needs a call from an agent session");
    await writeFile(filesIn(dir).errors, "## E-0001 · broken\n");
    expect((await callTool(tools, "err_stats", {})).value.error).toMatch(
      /could not read/,
    );
  });

  it("the estimate is fix notices × the assumed diagnosis − notice tokens", () => {
    expect(estimateSaved(0, 0)).toBe(0);
    expect(estimateSaved(3, 150)).toBe(3 * ASSUMED_DIAGNOSIS_TOKENS - 150);
    expect(estimateSaved(0, 40)).toBe(-40);
  });
});

// ---------------------------------------------------------------------------
// Injection counters

describe("injection counts", () => {
  it("an unknown session counts nothing; a pre-step notice is counted too", async () => {
    const fake = fakeCtx();
    const recorder = createRecorder({
      files: filesIn(dir),
      logger: { warn: () => undefined },
    });
    const injection = createInjection({
      recorder,
      options: { inject: "always" },
    });
    registerInjection(fake.ctx, injection, recorder);
    expect(injection.counts("nobody")).toEqual({
      notices: 0,
      fixNotices: 0,
      noticeTokens: 0,
    });
    // A miss under `always` is announced on the next pre-step.
    await fail(fake, "a brand new failure");
    const [preStep] = fake.listeners.get("agent/pre-step") ?? [];
    await (preStep as Listener)(
      { agent: { id: "s1" }, turn: 1, step: 1, messages: [] },
      async () => ({ kind: "enter", messages: [] }),
    );
    expect(injection.counts("s1")).toMatchObject({ notices: 1, fixNotices: 0 });
    expect(injection.counts()).toMatchObject({ notices: 1 });
  });
});
