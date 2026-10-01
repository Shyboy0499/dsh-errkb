// The five agent tools (T15, design document §9): err_lookup, err_record,
// err_list, err_forget and err_stats. The model and a person at the prompt
// call them the same way.
//
// The API is the real one in @deepseek-ai/dsh-tools (lib/types/schema.d.ts and
// lib/types/index.d.ts), read off the installed package:
//
// - `defineTool(options: DefineToolOptions<S, O>): ToolDefinition`. `S` is a
//   `ParameterSchemaSpec` (one `ParameterPropertySpec` per argument, `required:
//   true` where needed); `O` is a `ValueSchemaSpec`. `execute(args:
//   InferArgs<S>, exec: ToolRunContext)` returns the canonical value,
//   `InferValue<O>`. defineTool validates the arguments against `S` first and
//   throws `ToolArgsError` (a `HarnessError` the registry turns into a clear
//   error result) for a wrong type or an unknown `status`/`scope` value.
// - `output: ToolOutputDefinition`, `{ schema, render(args, value) }`. The
//   registry checks every value against `schema` and the model reads only what
//   `render` returns: plain `TextBlock`s (dsh-llm `ContentBlock`), as in
//   dsh-note (§9). The structured value stays with the host.
// - `ctx.tools.register(definition): () => void` on `ToolRuntime`, the `tools`
//   service index.ts already injects. `ToolRunContext.agent?.id` is the session.
//
// Errors. Everything defineTool's schema cannot say - exactly one of `id` and
// `message`, an ID that does not exist, an empty fix, a store that is busy or
// unreadable - comes back as a value with one `error` field, rendered as one
// line, never as a thrown error. Every body runs inside a catch-all that does
// the same with anything unexpected, after counting it like a failed write.
//
// Writes. Every one goes through the recorder: recordFix() for a fix (T14:
// the entry becomes `fixed`), write() for the rest. Both are queued behind the
// capture writes on the knowledge base's one chain, with the same 500 ms
// budget and retries, and the store redacts every text on the way in. So
// err_record is the only way a fix reaches ERRORS.md (§3 step 6), and a tool
// call never interleaves with a hook's write.
import { defineTool } from "@deepseek-ai/dsh-tools";
import type { ToolDefinition, ToolRunContext } from "@deepseek-ai/dsh-tools";
import type { Context } from "@deepseek-ai/cordis";
import type { ContentBlock } from "@deepseek-ai/dsh-llm";
import { extractHeadline, safeErrorText } from "./capture";
import { clip, oneLine } from "./inject";
import { indexEntries, jaccard, match, tokenize } from "./match";
import type { Hit, IndexedEntry, MatchOptions } from "./match";
import type { Injection, Recorder, WriteOutcome } from "./plugin";
import { normalize, signature } from "./signature";
import { ENTRY_STATUSES, formatId } from "./store";
import type { Entry, EntryPatch, EntryStatus } from "./store";

/** The tool names, as the model sees them. */
export const TOOL_NAMES = [
  "err_lookup",
  "err_record",
  "err_list",
  "err_forget",
  "err_stats",
] as const;

/** Entries err_list returns when `limit` is not given (§9). */
export const DEFAULT_LIST_LIMIT = 20;

/** The most entries err_list returns, whatever `limit` says. */
export const MAX_LIST_LIMIT = 200;

/** Closest entries err_lookup offers on a miss (§9). */
export const CLOSEST_COUNT = 3;

/** Category of an entry err_record creates from a message without one. */
export const DEFAULT_RECORD_CATEGORY = "agent";

/**
 * The diagnosis one injected fix is assumed to replace, in tokens. §7 puts a
 * re-diagnosis at 800-3000 tokens of thinking and trial; this takes the low
 * end, so the estimate errs towards too little.
 */
export const ASSUMED_DIAGNOSIS_TOKENS = 800;

/** Titles in err_list and err_lookup's closest list are clipped to this. */
export const TITLE_MAX_CHARS = 120;

/** A query quoted back in a miss is clipped to this. */
const QUERY_MAX_CHARS = 80;

const SIGNATURE = /^[0-9a-f]{12}$/i;

/** Settings the tools read; every one has a default. */
export interface ToolsOptions extends MatchOptions {
  idPrefix: string;
  idWidth: number;
}

export const DEFAULT_TOOLS_OPTIONS: ToolsOptions = {
  idPrefix: "E-",
  idWidth: 4,
  fuzzyThreshold: 0.72,
};

/** Everything the tools need. */
export interface ToolsDeps {
  recorder: Recorder;
  /** For err_stats: notices delivered and fix trust. */
  injection: Pick<Injection, "counts" | "trust">;
  /** The resolved knowledge base directory, as the startup line logs it. */
  kbDir: string;
  options?: Partial<ToolsOptions>;
}

/** The part of a cordis Context the tools use. */
export type ToolHost = Pick<Context, "tools">;

// ---------------------------------------------------------------------------
// Shared pieces

/** The error branch every tool's output has. */
const ERROR = {
  type: "object",
  properties: { error: { type: "string", required: true } },
  additionalProperties: false,
} as const;

interface ErrorValue {
  error: string;
}

const isError = (value: object): value is ErrorValue => "error" in value;

const text = (body: string): ContentBlock[] => [{ type: "text", text: body }];

/** Render an error value as one line naming the tool. */
const errorText = (tool: string, value: ErrorValue) =>
  text(`${tool}: ${value.error}`);

const escapeRegExp = (value: string) =>
  value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** One line, clipped. */
const short = (value: string, max: number) => clip(oneLine(value), max);

/** Similarity rounded for display and for the JSON value. */
const round = (n: number) => Math.round(n * 100) / 100;

/** How a match was found, best first. */
const VIA_RANK = { exact: 0, fuzzy: 1, code: 2 } as const;

/**
 * Bind the tools to a recorder.
 *
 * @param deps - the recorder, the injection layer, the KB path and settings.
 * @returns the five definitions, ready for `ctx.tools.register()`.
 */
export function createTools(deps: ToolsDeps): ToolDefinition[] {
  const o: ToolsOptions = { ...DEFAULT_TOOLS_OPTIONS, ...deps.options };
  const { recorder } = deps;
  const idPattern = new RegExp(`^${escapeRegExp(o.idPrefix)}(\\d+)$`, "i");

  /** The ID number in an ID-shaped string, or undefined. */
  function idNumber(value: string): number | undefined {
    const m = idPattern.exec(value.trim());
    return m === null ? undefined : Number(m[1]);
  }

  /** The entry an ID-shaped string names: `E-7` finds `E-0007`. */
  function byId(
    index: readonly IndexedEntry[],
    value: string,
  ): IndexedEntry | undefined {
    const n = idNumber(value);
    return n === undefined
      ? undefined
      : index.find((i) => idNumber(i.entry.id) === n);
  }

  /** The canonical spelling of an ID-shaped string, for messages. */
  function canonicalId(value: string): string {
    const n = idNumber(value);
    return n === undefined ? value.trim() : formatId(n, o.idPrefix, o.idWidth);
  }

  /**
   * Match free text the way capture does (§5.3): its headline, signed under a
   * category. With no category, every category in the knowledge base is
   * tried and the best hit wins: exact before fuzzy before code, then the
   * higher similarity, then the earlier category.
   */
  function matchText(
    raw: string,
    index: readonly IndexedEntry[],
    category?: string,
  ): Hit | undefined {
    const headline = extractHeadline(raw);
    const message = headline.line === "" ? raw.trim() : headline.line;
    const categories =
      category === undefined
        ? [...new Set(index.map((i) => i.category))]
        : [category];
    let winner: Hit | undefined;
    for (const cat of categories) {
      const found = match(
        {
          category: cat,
          message,
          ...(headline.code === undefined ? {} : { code: headline.code }),
        },
        index,
        o,
      );
      if (!found.matched) continue;
      if (
        winner === undefined ||
        VIA_RANK[found.via] < VIA_RANK[winner.via] ||
        (found.via === winner.via && found.similarity > winner.similarity)
      )
        winner = found;
    }
    return winner;
  }

  /** The indexed entries, or an error value when the read failed. */
  async function readIndex(): Promise<IndexedEntry[] | ErrorValue> {
    const index = await recorder.entries();
    return (
      index ?? {
        error:
          "could not read ERRORS.md (it may not parse; see the plugin log)",
      }
    );
  }

  /** A write outcome's error, or its value. */
  function written<T>(outcome: WriteOutcome<T>): T | ErrorValue {
    if (outcome.kind === "done") return outcome.value;
    if (outcome.kind === "timeout")
      return { error: "the knowledge base is busy; try again" };
    return {
      error: `could not write ERRORS.md: ${safeErrorText(outcome.error).message}`,
    };
  }

  /**
   * Run a tool body; anything it throws comes back as an error value, counted
   * like a failed write.
   */
  async function safely<T extends object>(
    body: () => Promise<T | ErrorValue>,
  ): Promise<T | ErrorValue> {
    try {
      return await body();
    } catch (error) {
      recorder.fail(error);
      return { error: `internal error: ${safeErrorText(error).message}` };
    }
  }

  /** A trimmed string argument, or undefined when it is absent or blank. */
  const given = (value: string | undefined) => {
    const trimmed = value?.trim();
    return trimmed === undefined || trimmed === "" ? undefined : trimmed;
  };

  // -------------------------------------------------------------------------
  // err_lookup

  const ENTRY_VIEW = {
    type: "object",
    properties: {
      id: { type: "string", required: true },
      title: { type: "string", required: true },
      category: { type: "string", required: true },
      hits: { type: "integer", required: true },
      status: { type: "string", enum: ENTRY_STATUSES, required: true },
      fix: { type: "string", required: true },
      raw: { type: "string" },
    },
    additionalProperties: false,
  } as const;

  const lookup = defineTool({
    name: "err_lookup",
    description:
      "Look up an error in the errkb knowledge base by entry ID, 12-hex fingerprint or the error text itself. Returns the entry with its recorded fix, or the closest entries when nothing matches.",
    parameters: {
      query: {
        type: "string",
        required: true,
        description: `An entry ID (${formatId(7, o.idPrefix, o.idWidth)}), a 12-hex fingerprint, or the error message.`,
      },
      full: {
        type: "boolean",
        description: "Also return the redacted raw sample. Default false.",
      },
    },
    output: {
      schema: {
        oneOf: [
          {
            type: "object",
            properties: {
              query: { type: "string", required: true },
              via: {
                type: "string",
                enum: ["id", "fingerprint", "exact", "fuzzy", "code", "none"],
                required: true,
              },
              entry: { oneOf: [ENTRY_VIEW, { type: "null" }], required: true },
              closest: {
                type: "array",
                items: {
                  type: "object",
                  properties: {
                    id: { type: "string", required: true },
                    title: { type: "string", required: true },
                    similarity: { type: "number", required: true },
                  },
                  additionalProperties: false,
                },
                required: true,
              },
            },
            additionalProperties: false,
          },
          ERROR,
        ],
      },
      render(_args, value) {
        if (isError(value)) return errorText("err_lookup", value);
        const { entry } = value;
        if (entry !== null) {
          const lines = [
            `${entry.id} · ${entry.title}`,
            `category: ${entry.category} · hits: ${entry.hits} · status: ${entry.status} · matched by ${value.via}`,
            entry.fix === "" ? "fix: (none recorded)" : `fix: ${entry.fix}`,
          ];
          if (entry.raw !== undefined) lines.push("raw:", entry.raw);
          return text(lines.join("\n"));
        }
        const head = `No entry matches "${short(value.query, QUERY_MAX_CHARS)}".`;
        if (value.closest.length === 0) return text(head);
        return text(
          [
            `${head} Closest:`,
            ...value.closest.map(
              (c) => `${c.id} ${c.title} (similarity ${c.similarity})`,
            ),
          ].join("\n"),
        );
      },
    },
    isConcurrencySafe: () => true,
    execute: ({ query, full }) =>
      safely(async () => {
        const q = given(query);
        if (q === undefined) return { error: "query is empty" };
        const index = await readIndex();
        if (!Array.isArray(index)) return index;

        const view = (found: IndexedEntry) => ({
          id: found.entry.id,
          title: found.entry.title,
          category: found.entry.category,
          hits: found.entry.hits,
          status: found.entry.status,
          fix: found.entry.fix,
          ...(full === true ? { raw: found.entry.raw } : {}),
        });
        const result = (
          via: "id" | "fingerprint" | "exact" | "fuzzy" | "code",
          found: IndexedEntry,
        ) => ({ query: q, via, entry: view(found), closest: [] });

        if (idNumber(q) !== undefined) {
          const found = byId(index, q);
          if (found !== undefined) return result("id", found);
          return {
            error: `no entry ${canonicalId(q)} (it may have been archived)`,
          };
        }
        if (SIGNATURE.test(q)) {
          const found = index.find((i) => i.sig === q.toLowerCase());
          if (found !== undefined) return result("fingerprint", found);
        }
        const hit = matchText(q, index);
        if (hit !== undefined) {
          const found = index.find(
            (i) => i.entry === hit.entry,
          ) as IndexedEntry;
          return result(hit.via, found);
        }

        const tokens = tokenize(normalize(q));
        const closest = index
          .map((i, order) => ({
            i,
            order,
            similarity: jaccard(tokens, i.tokens),
          }))
          .filter((c) => c.similarity > 0)
          .sort((a, b) => b.similarity - a.similarity || a.order - b.order)
          .slice(0, CLOSEST_COUNT)
          .map(({ i, similarity }) => ({
            id: i.entry.id,
            title: short(i.entry.title, TITLE_MAX_CHARS),
            similarity: round(similarity),
          }));
        return { query: q, via: "none" as const, entry: null, closest };
      }),
  });

  // -------------------------------------------------------------------------
  // err_record

  /** Notes with one more line. */
  const withNote = (notes: string, note: string) =>
    notes === "" ? note : `${notes}\n${note}`;

  /** Apply status and note to an entry through the write path. */
  async function patch(
    id: string,
    status: EntryStatus | undefined,
    note: string | undefined,
  ): Promise<Entry | ErrorValue> {
    const outcome = await recorder.write(async (store) => {
      const document = await store.read();
      const current = document.blocks.find((b) => b.entry.id === id)?.entry;
      if (current === undefined) return undefined;
      const changes: EntryPatch = {};
      if (status !== undefined) changes.status = status;
      if (note !== undefined) changes.notes = withNote(current.notes, note);
      return store.update(id, changes);
    });
    const entry = written(outcome);
    if (entry === undefined) return { error: `no entry ${id}` };
    return entry;
  }

  /** Write a fix through recordFix(): the entry becomes `fixed`. */
  async function fixEntry(
    id: string,
    fix: string,
  ): Promise<Entry | ErrorValue> {
    const outcome = await recorder.recordFix(id, fix);
    if (outcome.kind === "fixed") return outcome.entry;
    if (outcome.kind === "unknown") return { error: `no entry ${id}` };
    if (outcome.kind === "timeout")
      return { error: "the knowledge base is busy; try again" };
    return { error: "could not write ERRORS.md (see the plugin log)" };
  }

  /**
   * Find the entry a message describes, or append one. Matching and appending
   * run in one write, so two calls about the same new error cannot both
   * append it.
   *
   * Only an exact hit names the entry to update. A near hit (fuzzy or code,
   * §5.3) writes nothing and comes back as the candidate: a fix recorded on a
   * similar but different entry would later be injected as its known fix.
   */
  async function findOrAppend(
    message: string,
    category: string | undefined,
    fix: string | undefined,
    status: EntryStatus | undefined,
    note: string | undefined,
  ): Promise<{ id: string; created: boolean } | ErrorValue> {
    const outcome = await recorder.write(async (store) => {
      const document = await store.read();
      const index = indexEntries(document.blocks.map((b) => b.entry));
      const hit = matchText(message, index, category);
      if (hit !== undefined && hit.via !== "exact")
        return {
          error: `closest match is ${hit.id} (approximate, by ${hit.via}); nothing was written. Call err_record with id: "${hit.id}" to confirm, or reword message`,
        };
      if (hit !== undefined) return { id: hit.id, created: false };

      const cat = category ?? DEFAULT_RECORD_CATEGORY;
      const headline = extractHeadline(message);
      const line = headline.line === "" ? message : headline.line;
      const { id } = await store.append({
        title: `[${cat}] ${line}`,
        signature: signature(cat, line),
        category: cat,
        meta: {
          cat,
          ...(headline.code === undefined ? {} : { code: headline.code }),
        },
        raw: message,
        ...(fix === undefined ? {} : { fix }),
        status: status ?? (fix === undefined ? "open" : "fixed"),
        ...(note === undefined ? {} : { notes: note }),
      });
      return { id, created: true };
    });
    return written(outcome);
  }

  const record = defineTool({
    name: "err_record",
    description:
      "Record what you learned about an error in the errkb knowledge base. Give exactly one of id (an existing entry) or message (the error text). A message updates an entry only when it matches exactly; on an approximate match nothing is written and the closest entry's ID is returned, so confirm it with id. A new entry is created when nothing matches. A fix marks the entry fixed. This is the only way a fix is written.",
    parameters: {
      id: {
        type: "string",
        description: `An existing entry, e.g. ${formatId(7, o.idPrefix, o.idWidth)}.`,
      },
      message: {
        type: "string",
        description:
          "The error text, when you do not know the ID. Only an exact match updates an existing entry.",
      },
      fix: {
        type: "string",
        description: "What fixed it, in one or two sentences.",
      },
      status: {
        type: "string",
        enum: ENTRY_STATUSES,
        description:
          "fixed, wontfix (never inject it) or open. A fix alone sets fixed.",
      },
      note: {
        type: "string",
        description: "A line added to the entry's notes.",
      },
      category: {
        type: "string",
        description: `For message: the category to match and file under, e.g. tool or command-exit. Default: match any; file new entries under ${DEFAULT_RECORD_CATEGORY}.`,
      },
    },
    output: {
      schema: {
        oneOf: [
          {
            type: "object",
            properties: {
              id: { type: "string", required: true },
              created: { type: "boolean", required: true },
              status: { type: "string", enum: ENTRY_STATUSES, required: true },
              hasFix: { type: "boolean", required: true },
            },
            additionalProperties: false,
          },
          ERROR,
        ],
      },
      render(_args, value) {
        if (isError(value)) return errorText("err_record", value);
        const what = value.created ? "Created" : "Updated";
        const fix = value.hasFix ? "" : ", no fix recorded";
        return text(`${what} ${value.id} (status ${value.status}${fix}).`);
      },
    },
    execute: (args) =>
      safely(async () => {
        const id = given(args.id);
        const message = given(args.message);
        const fix = given(args.fix);
        const note = given(args.note);
        const category = given(args.category);
        const { status } = args;
        if ((id === undefined) === (message === undefined))
          return { error: "give exactly one of id and message" };
        if (args.fix !== undefined && fix === undefined)
          return { error: "fix is empty" };

        let target: { id: string; created: boolean };
        if (id !== undefined) {
          if (fix === undefined && status === undefined && note === undefined)
            return { error: "nothing to record: give fix, status or note" };
          const index = await readIndex();
          if (!Array.isArray(index)) return index;
          const found = byId(index, id);
          if (found === undefined)
            return { error: `no entry ${canonicalId(id)}` };
          target = { id: found.entry.id, created: false };
        } else {
          const outcome = await findOrAppend(
            message as string,
            category,
            fix,
            status,
            note,
          );
          if (isError(outcome)) return outcome;
          target = outcome;
        }

        // A new entry was written whole; an existing one takes the fix
        // through recordFix(), then status and note in one more write, so an
        // explicit status wins over the `fixed` a fix implies.
        let entry: Entry | undefined;
        if (!target.created) {
          if (fix !== undefined) {
            const fixed = await fixEntry(target.id, fix);
            if (isError(fixed)) return fixed;
            entry = fixed;
          }
          if (status !== undefined || note !== undefined) {
            const patched = await patch(target.id, status, note);
            if (isError(patched)) return patched;
            entry = patched;
          }
        }
        if (entry === undefined) {
          const index = await readIndex();
          if (!Array.isArray(index)) return index;
          const found = index.find((i) => i.entry.id === target.id);
          if (found === undefined) return { error: `no entry ${target.id}` };
          entry = found.entry;
        }
        return {
          id: target.id,
          created: target.created,
          status: entry.status,
          hasFix: entry.fix.trim() !== "",
        };
      }),
  });

  // -------------------------------------------------------------------------
  // err_list

  const list = defineTool({
    name: "err_list",
    description:
      "List entries of the errkb knowledge base: ID, title and hits only, no bodies.",
    parameters: {
      cat: {
        type: "string",
        description:
          "Only this category: tool, command-exit, llm, agent, or a display category such as tool / bash.",
      },
      status: { type: "string", enum: ENTRY_STATUSES },
      limit: {
        type: "integer",
        description: `At most this many entries (default ${DEFAULT_LIST_LIMIT}, at most ${MAX_LIST_LIMIT}).`,
      },
    },
    output: {
      schema: {
        oneOf: [
          {
            type: "object",
            properties: {
              entries: {
                type: "array",
                items: {
                  type: "object",
                  properties: {
                    id: { type: "string", required: true },
                    title: { type: "string", required: true },
                    hits: { type: "integer", required: true },
                  },
                  additionalProperties: false,
                },
                required: true,
              },
              total: { type: "integer", required: true },
            },
            additionalProperties: false,
          },
          ERROR,
        ],
      },
      render(_args, value) {
        if (isError(value)) return errorText("err_list", value);
        if (value.total === 0) return text("No entries.");
        return text(
          [
            ...value.entries.map((e) => `${e.id} (${e.hits}) ${e.title}`),
            `${value.entries.length} of ${value.total} shown.`,
          ].join("\n"),
        );
      },
    },
    isConcurrencySafe: () => true,
    execute: (args) =>
      safely(async () => {
        const limit = args.limit ?? DEFAULT_LIST_LIMIT;
        if (!Number.isInteger(limit) || limit < 1)
          return { error: "limit must be a positive integer" };
        const index = await readIndex();
        if (!Array.isArray(index)) return index;
        const cat = given(args.cat)?.toLowerCase();
        const display = (i: IndexedEntry) => i.entry.category.toLowerCase();
        const matches = index.filter(
          (i) =>
            (cat === undefined ||
              i.category.toLowerCase() === cat ||
              display(i) === cat) &&
            (args.status === undefined || i.entry.status === args.status),
        );
        return {
          entries: matches
            .slice(0, Math.min(limit, MAX_LIST_LIMIT))
            .map(({ entry }) => ({
              id: entry.id,
              title: short(entry.title, TITLE_MAX_CHARS),
              hits: entry.hits,
            })),
          total: matches.length,
        };
      }),
  });

  // -------------------------------------------------------------------------
  // err_forget

  const forget = defineTool({
    name: "err_forget",
    description:
      "Remove a misjudged entry from the errkb knowledge base. It moves to ERRORS.archive.md with the reason noted; nothing is deleted.",
    parameters: {
      id: { type: "string", required: true },
      reason: { type: "string", description: "Why, kept in the archive." },
    },
    output: {
      schema: {
        oneOf: [
          {
            type: "object",
            properties: {
              id: { type: "string", required: true },
              archived: { type: "boolean", required: true },
            },
            additionalProperties: false,
          },
          ERROR,
        ],
      },
      render(_args, value) {
        if (isError(value)) return errorText("err_forget", value);
        return text(`Archived ${value.id} to ERRORS.archive.md.`);
      },
    },
    execute: (args) =>
      safely(async () => {
        const id = given(args.id);
        if (id === undefined) return { error: "id is empty" };
        const index = await readIndex();
        if (!Array.isArray(index)) return index;
        const found = byId(index, id);
        if (found === undefined)
          return { error: `no entry ${canonicalId(id)}` };
        const target = found.entry.id;
        const reason = given(args.reason);
        const outcome = written(
          await recorder.write((store) => store.archive(target, reason)),
        );
        if (outcome !== undefined && isError(outcome)) return outcome;
        if (outcome === undefined) return { error: `no entry ${target}` };
        return { id: target, archived: true };
      }),
  });

  // -------------------------------------------------------------------------
  // err_stats

  const stats = defineTool({
    name: "err_stats",
    description:
      "Show the errkb ledger: entries, hits, notices injected, an estimate of tokens saved, open entries without a fix, distrusted fixes and the knowledge base path. Costs no model call.",
    parameters: {
      scope: {
        type: "string",
        enum: ["session", "all"],
        description:
          "Notices of this session or of every session since the plugin started (default all). Entry figures are always the whole knowledge base.",
      },
    },
    output: {
      schema: {
        oneOf: [
          {
            type: "object",
            properties: {
              scope: {
                type: "string",
                enum: ["session", "all"],
                required: true,
              },
              kbDir: { type: "string", required: true },
              entries: { type: "integer", required: true },
              hits: { type: "integer", required: true },
              openWithoutFix: { type: "integer", required: true },
              notices: { type: "integer", required: true },
              fixNotices: { type: "integer", required: true },
              noticeTokens: { type: "integer", required: true },
              estimatedTokensSaved: { type: "integer", required: true },
              suppressed: {
                type: "array",
                items: { type: "string" },
                required: true,
              },
            },
            additionalProperties: false,
          },
          ERROR,
        ],
      },
      render(_args, value) {
        if (isError(value)) return errorText("err_stats", value);
        const where =
          value.scope === "session" ? "this session" : "all sessions";
        return text(
          [
            `Knowledge base: ${value.kbDir}`,
            `Entries: ${value.entries} · hits: ${value.hits} · open without a fix: ${value.openWithoutFix}`,
            `Notices (${where}): ${value.notices}, ${value.fixNotices} with a fix, ${value.noticeTokens} tokens`,
            `Estimated tokens saved: ${value.estimatedTokensSaved} (estimate: ${value.fixNotices} fix notices × ${ASSUMED_DIAGNOSIS_TOKENS} − ${value.noticeTokens} notice tokens)`,
            `Distrusted fixes, not injected: ${value.suppressed.length === 0 ? "none" : value.suppressed.join(", ")}`,
          ].join("\n"),
        );
      },
    },
    isConcurrencySafe: () => true,
    execute: (args, exec: ToolRunContext) =>
      safely(async () => {
        const scope = args.scope ?? "all";
        const session = exec.agent?.id;
        if (scope === "session" && session === undefined)
          return { error: "scope session needs a call from an agent session" };
        const index = await readIndex();
        if (!Array.isArray(index)) return index;
        const counts = deps.injection.counts(
          scope === "session" ? session : undefined,
        );
        const { trust } = deps.injection;
        return {
          scope,
          kbDir: deps.kbDir,
          entries: index.length,
          hits: index.reduce((sum, { entry }) => sum + entry.hits, 0),
          openWithoutFix: index.filter(
            ({ entry }) => entry.status === "open" && entry.fix.trim() === "",
          ).length,
          ...counts,
          estimatedTokensSaved: estimateSaved(
            counts.fixNotices,
            counts.noticeTokens,
          ),
          suppressed: index
            .filter(
              ({ entry }) =>
                entry.fix.trim() !== "" &&
                trust.level(entry.id, entry.fix) === "suppressed",
            )
            .map(({ entry }) => entry.id),
        };
      }),
  });

  return [lookup, record, list, forget, stats];
}

/**
 * err_stats' estimate of the tokens the knowledge base saved: every notice
 * that carried a fix is assumed to replace one diagnosis of
 * {@link ASSUMED_DIAGNOSIS_TOKENS} tokens, and every notice delivered costs
 * its own tokens. It leaves out the standing costs (the system-prompt section
 * on every request, the session digest), so it is an estimate, and labelled
 * one wherever it is shown.
 *
 * @param fixNotices - notices that carried a fix.
 * @param noticeTokens - the tokens of every notice delivered.
 */
export function estimateSaved(
  fixNotices: number,
  noticeTokens: number,
): number {
  return fixNotices * ASSUMED_DIAGNOSIS_TOKENS - noticeTokens;
}

/**
 * Register the five tools (T15). A tool the registry refuses (a duplicate
 * name) is counted and logged; the others are still registered.
 *
 * @param ctx - the plugin's context, with the `tools` service.
 * @param deps - see {@link createTools}.
 */
export function registerTools(ctx: ToolHost, deps: ToolsDeps): void {
  for (const tool of createTools(deps)) {
    try {
      ctx.tools.register(tool);
    } catch (error) {
      deps.recorder.fail(error);
    }
  }
}
