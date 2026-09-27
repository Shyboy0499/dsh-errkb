// Where the knowledge base lives, and the file names inside it.
//
// The decision logic here is pure: `pluginRoot`, `dshHome`, `isWritable` and
// `isInsideNodeModules` are all injected, because two of the four tiers depend
// on conditions a test cannot construct on a real filesystem (an unwritable
// package root, a package root inside node_modules). `defaultProbe()` is the
// only place that touches the machine.
//
// Nothing in this module writes: no mkdir, no writeFile, no lock file. Tier
// resolution only probes whether a directory could be written to; locking and
// atomic writes belong to the store task.
import { accessSync, constants, statSync } from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

/** The six file names the knowledge base directory holds. */
export const KB_FILE = {
  errors: "ERRORS.md",
  archive: "ERRORS.archive.md",
  index: "errors.index.json",
  state: "state.json",
  machine: ".machine.json",
  lock: ".lock",
} as const;

/** Environment variable that overrides the harness home. */
export const DSH_HOME_ENV = "DSH_HOME";

/** Directory under the OS home that holds harness data when `$DSH_HOME` is unset. */
export const DSH_HOME_DIR_NAME = ".dsh";

/** Directory under the plugin root used by tiers 1a and 2. */
export const KB_DIR_NAME = "errors";

/** Directory under the harness home used by tier 3. */
export const KB_FALLBACK_DIR_NAME = "errkb";

/** Which rule produced the directory, for the startup log and for support. */
export type KbTier =
  "config-relative" | "config-absolute" | "plugin-root" | "dsh-home";

/** Absolute paths of the six files inside one knowledge base directory. */
export interface KbFiles {
  errors: string;
  archive: string;
  index: string;
  state: string;
  machine: string;
  lock: string;
}

/** A resolved knowledge base directory. */
export interface ResolvedKb {
  dir: string;
  tier: KbTier;
  files: KbFiles;
}

/** Everything tier resolution needs from the machine, injected for testability. */
export interface KbProbe {
  pluginRoot: string;
  dshHome: string;
  isWritable(dir: string): boolean;
  isInsideNodeModules(dir: string): boolean;
}

/**
 * Expand a leading `~`, `~/` or `~\` against the OS home.
 *
 * This mirrors `@deepseek-ai/dsh-home-paths`, so a `$DSH_HOME` written as
 * `~/dsh` lands where the harness would put it. It is deliberately *not* applied
 * to the `kbDir` setting: that setting resolves against the plugin package root,
 * and a `~` there is a configuration mistake we would rather surface as a
 * literal directory name than silently reinterpret. No environment variables are
 * interpolated anywhere.
 *
 * @param input - the raw path.
 * @returns the expanded path, or the input unchanged when no supported prefix is present.
 */
export function expandHomePath(input: string): string {
  if (input === "~") return homedir();
  if (input.startsWith("~/") || input.startsWith("~\\"))
    return join(homedir(), input.slice(2));
  return input;
}

/**
 * Resolve the harness home: `$DSH_HOME` when it is set and not blank, else
 * `<os home>/.dsh`. The precedence and the blank-is-unset rule are copied from
 * `@deepseek-ai/dsh-home-paths`, which also treats a whitespace-only value as
 * unset so a blank override never resolves to the current directory.
 *
 * @param env - environment mapping to read; injected for tests.
 * @returns an absolute harness home path.
 */
export function resolveDshHome(env: NodeJS.ProcessEnv = process.env): string {
  const fromEnv = env[DSH_HOME_ENV];
  const configured =
    fromEnv !== undefined && fromEnv.trim().length > 0
      ? fromEnv
      : join(homedir(), DSH_HOME_DIR_NAME);
  return resolve(expandHomePath(configured));
}

/**
 * Derive the plugin package root from a module URL.
 *
 * The same file is loaded from `lib/index.js` once bundled and from
 * `src/paths.ts` under the test runner, so both build directories are walked
 * out of. A `link:` install needs no special case: Node resolves the symlink to
 * the real repository path before this code runs.
 *
 * @param moduleUrl - `import.meta.url` of a module inside the package.
 * @returns the package root.
 */
export function pluginRootFrom(moduleUrl: string): string {
  let current = dirname(fileURLToPath(moduleUrl));
  while (basename(current) === "lib" || basename(current) === "src") {
    const parent = dirname(current);
    /* v8 ignore next -- a filesystem root exists, so the walk stops before this guard */
    if (parent === current) break;
    current = parent;
  }
  return current;
}

/**
 * Whether a path has a `node_modules` segment.
 *
 * Matching is by whole path segment, not by substring: `node_modules_backup`
 * and `my_node_modules` must not match. Both separators are accepted, and the
 * comparison is case-insensitive on Windows.
 *
 * @param dir - the path to test.
 * @param platform - platform to apply; injected for tests.
 * @returns true when a segment equals `node_modules`.
 */
export function isInsideNodeModulesPath(
  dir: string,
  platform: NodeJS.Platform = process.platform,
): boolean {
  const candidate = platform === "win32" ? dir.toLowerCase() : dir;
  return candidate.split(/[\\/]+/).includes("node_modules");
}

/**
 * Whether a directory can be written to, climbing to the nearest existing
 * ancestor when the directory itself does not exist yet.
 *
 * The climb stops at a regular file, because a path running through one can
 * never become a directory. That case is not theoretical on Windows: `stat` on
 * `<file>/child` reports ENOENT rather than ENOTDIR (measured on this machine),
 * so a climb that only watched for ENOENT would land on the file and call it
 * writable. Any other inspection error - including a permission denial - is a
 * refusal rather than a climb.
 *
 * Platform note, measured on Windows: `fs.access(W_OK)` reflects the read-only
 * attribute and the effective access token rather than the directory's ACL. A
 * directory carrying the read-only attribute still reports writable and still
 * accepts files, so this probe is evidence, not a promise; the store task keeps
 * its write path recoverable for that reason.
 *
 * @param dir - directory that may or may not exist.
 * @returns true when the directory, or its nearest existing ancestor, is a writable directory.
 */
export function isWritableDir(dir: string): boolean {
  let current = resolve(dir);
  for (;;) {
    try {
      if (!statSync(current).isDirectory()) return false;
      accessSync(current, constants.W_OK);
      return true;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") return false;
      const parent = dirname(current);
      /* v8 ignore next -- a filesystem root exists, so traversal resolves before this guard */
      if (parent === current) return false;
      current = parent;
    }
  }
}

/**
 * Build the six absolute file paths inside a knowledge base directory.
 *
 * @param dir - the resolved knowledge base directory.
 * @returns the file paths.
 */
export function filesIn(dir: string): KbFiles {
  return {
    errors: join(dir, KB_FILE.errors),
    archive: join(dir, KB_FILE.archive),
    index: join(dir, KB_FILE.index),
    state: join(dir, KB_FILE.state),
    machine: join(dir, KB_FILE.machine),
    lock: join(dir, KB_FILE.lock),
  };
}

/**
 * Name for the copy of a document that could not be parsed.
 *
 * Colons and dots are replaced so the name is legal on Windows, and the whole
 * stamp is derived from the argument so the caller can pin it in a test.
 *
 * @param now - timestamp to embed; defaults to the current time.
 * @returns a file name, never a path.
 */
export function corruptFileName(now: Date = new Date()): string {
  const stamp = now.toISOString().replace(/[:.]/g, "-");
  return `ERRORS.corrupt-${stamp}.md`;
}

/**
 * Resolve the knowledge base directory.
 *
 * Tier 1a: `kbDir` is a relative path -> `<pluginRoot>/<kbDir>`.
 * Tier 1b: `kbDir` is an absolute path -> used as written.
 * Tier 2:  `kbDir` is empty and the plugin root is writable and not inside
 *          `node_modules` -> `<pluginRoot>/errors`.
 * Tier 3:  otherwise (global install, read-only package directory, unwritable
 *          disk) -> `<dshHome>/errkb`.
 *
 * A blank or whitespace-only `kbDir` counts as empty, matching the harness's
 * treatment of a blank `$DSH_HOME`.
 *
 * @param kbDir - the configured directory, relative or absolute, or empty.
 * @param probe - machine facts, injected.
 * @returns the directory, the tier that produced it and the six file paths.
 */
export function resolveKbDir(kbDir: string, probe: KbProbe): ResolvedKb {
  const configured = kbDir.trim();
  if (configured.length > 0) {
    if (isAbsolute(configured)) {
      return {
        dir: configured,
        tier: "config-absolute",
        files: filesIn(configured),
      };
    }
    const dir = join(probe.pluginRoot, configured);
    return { dir, tier: "config-relative", files: filesIn(dir) };
  }
  if (
    probe.isWritable(probe.pluginRoot) &&
    !probe.isInsideNodeModules(probe.pluginRoot)
  ) {
    const dir = join(probe.pluginRoot, KB_DIR_NAME);
    return { dir, tier: "plugin-root", files: filesIn(dir) };
  }
  const dir = join(probe.dshHome, KB_FALLBACK_DIR_NAME);
  return { dir, tier: "dsh-home", files: filesIn(dir) };
}

/**
 * The one startup line: directory plus tier, so "which tier did the global
 * install take" is never a guess.
 *
 * @param resolved - the resolved knowledge base.
 * @returns a single line, without a trailing newline.
 */
export function formatKbLog(resolved: ResolvedKb): string {
  return `[errkb] kb dir: ${resolved.dir} (tier: ${resolved.tier})`;
}

/**
 * The only function here that reads the machine: package root, harness home,
 * and the two probes.
 *
 * @returns a probe bound to this process.
 */
export function defaultProbe(): KbProbe {
  return {
    pluginRoot: pluginRootFrom(import.meta.url),
    dshHome: resolveDshHome(),
    isWritable: isWritableDir,
    isInsideNodeModules: (dir: string) => isInsideNodeModulesPath(dir),
  };
}
