import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { describe, expect, it } from "vitest";
import {
  DSH_HOME_DIR_NAME,
  DSH_HOME_ENV,
  KB_DIR_NAME,
  KB_FALLBACK_DIR_NAME,
  KB_FILE,
  corruptFileName,
  defaultProbe,
  expandHomePath,
  filesIn,
  formatKbLog,
  isInsideNodeModulesPath,
  isWritableDir,
  pluginRootFrom,
  resolveDshHome,
  resolveKbDir,
} from "../src/paths";
import type { KbProbe } from "../src/paths";

const pluginRoot = resolve("/plugin-root");
const dshHome = resolve("/dsh-home");

/** A probe whose machine facts are fixed, so tier logic is testable anywhere. */
function probe(overrides: Partial<KbProbe> = {}): KbProbe {
  return {
    pluginRoot,
    dshHome,
    isWritable: () => true,
    isInsideNodeModules: () => false,
    ...overrides,
  };
}

describe("resolveKbDir", () => {
  it("tier 1a: a relative kbDir lands under the plugin root, never under the cwd", () => {
    const resolved = resolveKbDir("errors", probe());
    expect(resolved.tier).toBe("config-relative");
    expect(resolved.dir).toBe(join(pluginRoot, "errors"));
    expect(resolved.dir.startsWith(pluginRoot)).toBe(true);
    expect(resolved.dir).not.toContain(process.cwd());
  });

  it("tier 1a: a nested relative kbDir is joined, not resolved against the cwd", () => {
    expect(resolveKbDir("nested/dir", probe()).dir).toBe(
      join(pluginRoot, "nested", "dir"),
    );
  });

  it("tier 1b: an absolute kbDir is used as written", () => {
    const absolute = resolve("/absolute/kb");
    const resolved = resolveKbDir(absolute, probe());
    expect(resolved.tier).toBe("config-absolute");
    expect(resolved.dir).toBe(absolute);
  });

  it.runIf(process.platform === "win32")(
    "tier 1b: a drive-letter path is absolute on Windows",
    () => {
      const resolved = resolveKbDir("D:/kb", probe());
      expect(resolved.tier).toBe("config-absolute");
      expect(resolved.dir).toBe("D:/kb");
    },
  );

  it("tier 2: an empty kbDir with a writable plugin root outside node_modules", () => {
    const resolved = resolveKbDir("", probe());
    expect(resolved.tier).toBe("plugin-root");
    expect(resolved.dir).toBe(join(pluginRoot, KB_DIR_NAME));
  });

  it("tier 2: a whitespace-only kbDir counts as empty", () => {
    expect(resolveKbDir("   ", probe()).tier).toBe("plugin-root");
  });

  it("tier 3: an unwritable plugin root falls back to the harness home", () => {
    const resolved = resolveKbDir("", probe({ isWritable: () => false }));
    expect(resolved.tier).toBe("dsh-home");
    expect(resolved.dir).toBe(join(dshHome, KB_FALLBACK_DIR_NAME));
  });

  it("tier 3: a plugin root inside node_modules falls back to the harness home", () => {
    const resolved = resolveKbDir(
      "",
      probe({ isInsideNodeModules: () => true }),
    );
    expect(resolved.tier).toBe("dsh-home");
    expect(resolved.dir).toBe(join(dshHome, KB_FALLBACK_DIR_NAME));
  });

  it("tier 3 wins over an unwritable root inside node_modules", () => {
    const resolved = resolveKbDir(
      "",
      probe({ isWritable: () => false, isInsideNodeModules: () => true }),
    );
    expect(resolved.dir).toBe(join(dshHome, KB_FALLBACK_DIR_NAME));
  });

  it("returns the six file paths derived from the resolved directory", () => {
    const resolved = resolveKbDir("", probe());
    expect(resolved.files).toEqual(filesIn(resolved.dir));
  });
});

describe("file names", () => {
  it("matches the README table word for word", () => {
    expect(KB_FILE).toEqual({
      errors: "ERRORS.md",
      archive: "ERRORS.archive.md",
      index: "errors.index.json",
      state: "state.json",
      machine: ".machine.json",
      lock: ".lock",
    });
  });

  it("derives every path from the directory", () => {
    const dir = join(pluginRoot, "errors");
    expect(filesIn(dir)).toEqual({
      errors: join(dir, "ERRORS.md"),
      archive: join(dir, "ERRORS.archive.md"),
      index: join(dir, "errors.index.json"),
      state: join(dir, "state.json"),
      machine: join(dir, ".machine.json"),
      lock: join(dir, ".lock"),
    });
  });

  it("keeps every name relative: no drive letter and no separator", () => {
    for (const name of Object.values(KB_FILE)) {
      expect(name).not.toMatch(/^[A-Za-z]:/);
      expect(name).not.toContain("/");
      expect(name).not.toContain("\\");
    }
  });

  it("names a corrupt document copy with a Windows-legal stamp", () => {
    const name = corruptFileName(new Date("2026-09-27T01:02:03.004Z"));
    expect(name).toBe("ERRORS.corrupt-2026-09-27T01-02-03-004Z.md");
    expect(name).not.toContain(":");
    expect(dirname(name)).toBe(".");
  });
});

describe("pluginRootFrom", () => {
  const root = resolve("/tmp/plugin");

  it("walks out of a bundled lib directory", () => {
    expect(
      pluginRootFrom(pathToFileURL(join(root, "lib", "paths.js")).href),
    ).toBe(root);
  });

  it("walks out of a source src directory", () => {
    expect(
      pluginRootFrom(pathToFileURL(join(root, "src", "paths.ts")).href),
    ).toBe(root);
  });

  it("keeps a module that already sits in the root", () => {
    expect(pluginRootFrom(pathToFileURL(join(root, "index.js")).href)).toBe(
      root,
    );
  });

  it("stops at a filesystem root instead of walking past it", () => {
    const fsRoot = dirname(resolve("/"));
    expect(
      pluginRootFrom(pathToFileURL(join(fsRoot, "lib", "x.js")).href),
    ).toBe(fsRoot);
  });
});

describe("isInsideNodeModulesPath", () => {
  it("matches a node_modules segment", () => {
    expect(isInsideNodeModulesPath(join("/repo", "node_modules", "pkg"))).toBe(
      true,
    );
    expect(isInsideNodeModulesPath("/node_modules/pkg")).toBe(true);
  });

  it("does not match a lookalike segment", () => {
    expect(
      isInsideNodeModulesPath(join("/repo", "node_modules_backup", "pkg")),
    ).toBe(false);
    expect(
      isInsideNodeModulesPath(join("/repo", "my_node_modules", "pkg")),
    ).toBe(false);
    expect(isInsideNodeModulesPath(join("/repo", "sources"))).toBe(false);
  });

  it("accepts backslash separators", () => {
    expect(
      isInsideNodeModulesPath("C:\\repo\\node_modules\\pkg", "win32"),
    ).toBe(true);
  });

  it("is case-insensitive on Windows only", () => {
    expect(
      isInsideNodeModulesPath("C:\\repo\\NODE_MODULES\\pkg", "win32"),
    ).toBe(true);
    expect(isInsideNodeModulesPath("/repo/NODE_MODULES/pkg", "linux")).toBe(
      false,
    );
  });
});

describe("harness home", () => {
  it("expands a leading tilde the way the harness does", () => {
    expect(expandHomePath("~")).toBe(homedir());
    expect(expandHomePath("~/custom")).toBe(join(homedir(), "custom"));
    expect(expandHomePath("~\\custom")).toBe(join(homedir(), "custom"));
    expect(expandHomePath("/custom")).toBe("/custom");
  });

  it("prefers $DSH_HOME when it is set", () => {
    expect(resolveDshHome({ [DSH_HOME_ENV]: resolve("/custom/home") })).toBe(
      resolve("/custom/home"),
    );
  });

  it("expands a tilde in $DSH_HOME", () => {
    expect(resolveDshHome({ [DSH_HOME_ENV]: "~/custom" })).toBe(
      resolve(join(homedir(), "custom")),
    );
  });

  it("falls back to <home>/.dsh when $DSH_HOME is unset", () => {
    expect(resolveDshHome({})).toBe(
      resolve(join(homedir(), DSH_HOME_DIR_NAME)),
    );
  });

  it("treats a blank $DSH_HOME as unset rather than as the cwd", () => {
    expect(resolveDshHome({ [DSH_HOME_ENV]: "   " })).toBe(
      resolve(join(homedir(), DSH_HOME_DIR_NAME)),
    );
  });

  it("lets the environment override the fallback in the running process", () => {
    expect(resolveDshHome()).toBe(resolveDshHome(process.env));
  });
});

describe("startup log", () => {
  it("is exactly one line naming the directory and the tier", () => {
    const line = formatKbLog(resolveKbDir("", probe()));
    expect(line).toBe(
      `[errkb] kb dir: ${join(pluginRoot, KB_DIR_NAME)} (tier: plugin-root)`,
    );
    expect(line.split("\n")).toHaveLength(1);
  });

  it("makes the tier visible for every rule", () => {
    expect(formatKbLog(resolveKbDir("kb", probe()))).toContain(
      "(tier: config-relative)",
    );
    expect(formatKbLog(resolveKbDir(resolve("/abs"), probe()))).toContain(
      "(tier: config-absolute)",
    );
    expect(
      formatKbLog(resolveKbDir("", probe({ isWritable: () => false }))),
    ).toContain("(tier: dsh-home)");
  });
});

describe("defaultProbe", () => {
  const repoRoot = resolve(fileURLToPath(new URL("..", import.meta.url)));

  it("reports this package's root, not the lib or src directory", () => {
    expect(defaultProbe().pluginRoot).toBe(repoRoot);
  });

  it("resolves the harness home from the real environment", () => {
    expect(defaultProbe().dshHome).toBe(resolveDshHome());
  });

  it("sees this repository as writable", () => {
    expect(defaultProbe().isWritable(repoRoot)).toBe(true);
  });

  it("sees a package directory inside node_modules", () => {
    expect(
      defaultProbe().isInsideNodeModules(
        join(repoRoot, "node_modules", "tsdown"),
      ),
    ).toBe(true);
  });

  it("does not see the repository itself as inside node_modules", () => {
    expect(defaultProbe().isInsideNodeModules(repoRoot)).toBe(false);
  });

  it("climbs to an existing ancestor for a directory that does not exist yet", () => {
    expect(isWritableDir(join(repoRoot, "not-created-yet", "deeper"))).toBe(
      true,
    );
  });

  it("refuses a path that runs through a regular file", () => {
    // Windows reports ENOENT rather than ENOTDIR for `<file>/child` (measured on
    // this machine), which is the case a naive ancestor walk gets wrong.
    expect(isWritableDir(join(repoRoot, "package.json", "child"))).toBe(false);
  });

  it("treats a path it cannot inspect as not writable", () => {
    // Any inspection error other than ENOENT is a refusal rather than a climb. A
    // NUL byte can never occur in a real path, which makes it a portable way to
    // reach that branch on every platform.
    expect(isWritableDir("bad\0path")).toBe(false);
  });

  it("leaves no trace on disk", () => {
    const kbDir = join(repoRoot, KB_DIR_NAME);
    const existedBefore = existsSync(kbDir);
    resolveKbDir("", defaultProbe());
    formatKbLog(resolveKbDir("", defaultProbe()));
    expect(existsSync(kbDir)).toBe(existedBefore);
  });
});
