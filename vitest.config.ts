import { defineConfig } from "vitest/config";

// P1 scaffold. The 99% gate from the design document (§14) is declared here but
// does not bite yet: `pnpm test` runs `vitest run` without `--coverage`, and
// vitest only evaluates coverage thresholds when coverage is collected. That is
// what keeps the scaffolding phase green while src/ is still empty - it is not
// an accident to rely on, so P2 has to turn the gate on for real by adding
// `--coverage` to the test script or to the CI step. No task owns that edit yet.
//
// testTimeout and hookTimeout are raised well above the 5s default for the
// concurrency case in §14 group 4 (50 simultaneous records must yield 50 unique
// IDs), which exercises the file lock and the atomic rename.
//
// passWithNoTests keeps `pnpm test` green while tests/ holds only a .gitkeep.
// Drop it once the first real suite lands, or a discovery mistake would pass
// silently instead of failing.
export default defineConfig({
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
    testTimeout: 15_000,
    hookTimeout: 15_000,
    passWithNoTests: true,
    coverage: {
      provider: "v8",
      reporter: ["text", "lcov"],
      include: ["src/**"],
      thresholds: {
        statements: 99,
        lines: 99,
      },
    },
  },
});
