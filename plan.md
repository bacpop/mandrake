Read AGENTS.md before starting work.

This file is the working plan. Keep it up to date.

At the beginning of each session:
- Read this file completely.
- Verify the Current Task still matches the repository.
- If not, update it before writing code.

During work:
- Mark completed checklist items.
- Record any important design decisions.
- Record blockers immediately.

Before finishing:
- Update Current Status.
- Update Next Task so another session can continue without context.
- Record any deviations from the original plan.

## Definition of Done

- [x] Builds with `cargo build`
- [x] Tests pass
- [x] Public API documented
- [x] plan.md updated

---

# Project

## Goal

Port the C++ package `mandrake` to Rust.

Source:
- `src/`
- primarily `src/wtsne_cpu.cpp`

Target:
- `rust/`
- standalone Cargo crate

Out of scope:
- CUDA (`*.cu`, `*.cuh`)
- Python bindings unless explicitly requested

---

# Engineering Principles

Priority:

1. Computational efficiency
2. Memory efficiency
3. Readability
4. Error reporting

Mirror the style of [sketchlib.rust]((https://github.com/bacpop/sketchlib.rust):

- clap
- anyhow
- log/env_logger
- rayon where appropriate
- wasm32 compatibility
- cargo test

---

# Public Interface

Required API:

```rust
pub fn wtsne(...)
```

Returns an `SceResults` object whose final state is an `ndarray` embedding.

Expose:

- Rust library
- CLI (deferred until an input/output file contract is defined)

---

# Current Task

## Objective

Implement the Rust `SceResults` result module and migrate `wtsne()` to return
the result object. Store the final embedding plus optional initial/intermediate
frames with linear or true exponential iteration spacing.

### Checklist

- [x] Define result/frame interface
- [x] Add frame schedule configuration
- [x] Capture initial/intermediate/final frames
- [x] Migrate `wtsne()` return type and callers
- [x] Add result/frame accessors
- [x] Tests
- [x] Build, lint, and documentation checks

### Current Status

Working on:

Completed `SceResults` and the `wtsne()` result-returning interface.

Last verified:

2026-08-07: `cargo doc --no-deps`, `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test` (11 tests), `cargo build`, and `git diff --check` pass.

---

# Design Notes

## Parallelism

- Prefer Rayon.
- Atomic floating-point updates are required for embedding writes; use a CAS
  loop over `AtomicU64` bit patterns and retain the C++ optimistic clash retry.
- Benchmark against the C++ implementation after correctness is established.

## RNG

- Thread-safe RNG compatible with Rayon.
- Use independent xoshiro32 streams derived from the configured seed for each
  logical worker and each initial embedding point; fixed-seed single-worker
  runs are reproducible.

## Progress

Use indicatif.
- Progress is optional through `WtsneOptions::progress` and is updated once per
  completed optimisation iteration.

## Conditional probabilities

Port implementation from `wtsne.hpp`.
- Preserve a valid uniform row when exponentials underflow for extreme finite
  distances, then normalize the complete edge-probability vector.

## v1 API and scope

- Public input is zero-based COO `I`/`J` endpoint vectors, distance values, and
  one non-negative weight per node.
- `wtsne(...)` returns `Result<SceResults>`; the final frame is an ndarray with
  shape `(n_nodes, 2)`.
- `WtsneOptions` carries perplexity, iteration count, repulsion samples,
  learning rate, initial exaggeration, worker count, progress setting, seed,
  and frame schedule.
- Tests target public behavior: validation, normalization, finite output,
  reproducibility for one worker, and successful parallel execution.
- No Python bindings, CUDA, or file-format CLI are part of this milestone.

## SceResults design

- `wtsne()` returns `Result<SceResults>`; `SceResults` owns a non-empty,
  chronological frame sequence and exposes borrowed or consuming final-
  embedding accessors.
- `FrameSchedule::FinalOnly` stores only the final state. `Linear` and
  `Exponential` counts include the initial and final states and require at
  least two distinct iteration positions.
- Exponential positions use geometric spacing
  `round((max_iterations + 1)^u - 1)` with monotonic clamping so the requested
  count remains exact.
- Each frame records outer iteration, worker-update count, `Eq`, and a 2D
  ndarray snapshot. Captures occur only after an optimisation iteration has
  completed, so snapshots are race-free.
- Frame storage is intentionally in-memory and scales with frame count and
  node count, matching the C++ `sce_results` role.

---

# Open Questions

- Benchmark the atomic update implementation against the C++ CPU version and
  investigate alternatives if it misses the 5% target.
- SIMD opportunities? Ideally compiler optimised rather than 'hand-optimised'.

---

# Next Task

After `SceResults` is complete:

Benchmark the atomic Rust optimiser against the C++ CPU implementation and
investigate alternatives if it misses the target.

Requirements:

- fixed-seed numerical/invariant comparison
- peak runtime and memory measurements

---

# Later tasks

- Add the pairsnp code to allow input from a fasta alignment file,
  starting from src/gene.rs. Add
  sketchlib.rust as a dependency to call sparse distances as input from
  a list of fasta files. Wire these up to both the API and the CLI.
- Code style changes following sketchlib.rust. Move to rust RNG library
  xoshiro rather than manual implementation

---

# Session Log

## 2026-08-06

Completed:
- ...

Discovered:
- ...

Next session:
- ...

Blockers:
- None for the v1 implementation.

## 2026-08-07

Completed the Rust `wtsne` v1 library: conditional probabilities, alias
sampling, reproducible xoshiro streams, Rayon workers, atomic embedding
updates, optional indicatif progress, validation, and public API documentation.
Input-index conversion is checked before `usize` conversion for wasm32 safety,
and underflowed probability rows use a uniform fallback. All six integration
tests, build, formatting, clippy, and diff checks pass.

Next session:
- Implement `SceResults`, migrate `wtsne()` to return it, and add schedule and
  accessor tests.

## 2026-08-07 (SceResults phase)

Completed the `SceResults` phase. Chosen interface: `wtsne()` returns a result
object; `FinalOnly` is the default; configured frame counts include initial and
final states; exponential spacing is geometric with exact monotonic clamping.
Added 11 public integration tests; documentation, formatting, Clippy, build,
test, and diff checks pass.

Next session:
- Benchmark the Rust optimiser against the C++ CPU implementation.

Blockers:
- None for the `SceResults` phase; the C++ performance comparison remains
  pending.
