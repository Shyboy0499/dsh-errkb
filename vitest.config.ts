import { defineConfig } from "vitest/config";

// The 99% statements/lines gate from the design document (§14) bites: `pnpm
// test` runs `vitest run --coverage`, and vitest evaluates the thresholds below
// whenever coverage is collected, so CI's Test step fails when coverage drops.
// T11 turned it on, once src/index.ts was covered too. Before that, P1 ran
// `vitest run` without `--coverage` so the scaffold stayed green while src/
// was still empty.
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
