import { defineConfig } from "tsdown";

// The four harness packages are peer dependencies: the plugin runs inside the
// host application, so they must stay run-time imports and never be bundled
// into lib/. Their names are the scoped ones the host actually ships, not the
// bare "dsh" / "cordis" / "schemastery" labels used in prose.
//
// They go into `deps.neverBundle` rather than the top-level `external` option,
// which tsdown 0.22.2 reports as deprecated and refuses to combine with this
// one; the design document names `neverBundle` too (§11).
export default defineConfig({
  entry: ["src/index.ts"],
  outDir: "lib",
  format: ["esm"],
  platform: "node",
  // platform "node" makes fixedExtension default to true, which forces .mjs and
  // .d.mts regardless of the package type. `exports` in package.json points at
  // lib/index.js and lib/index.d.ts, so the build has to name its files the way
  // the package type does.
  fixedExtension: false,
  target: "node22",
  dts: true,
  clean: true,
  sourcemap: true,
  deps: {
    neverBundle: [
      "@deepseek-ai/cordis",
      "@deepseek-ai/dsh-tools",
      "@deepseek-ai/dsh-llm",
      "@deepseek-ai/schemastery",
    ],
  },
});
