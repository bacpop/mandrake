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
- End-to-end CLI for one distance source and embedding output

---

# Current Task

## Objective

Refactor the completed Rust implementation into deep optimiser, distance, API,
and CLI modules, and replace the hand-written xoshiro128+ implementation with
`rand_xoshiro::Xoshiro256PlusPlus` using a deliberate reseeding migration.

Before code changes, this task updates this plan. Before finishing, it records
completed milestones, design decisions, verification, blockers, and the next
task here.

Before code changes, this task must update this plan. Before finishing, it must
record completed milestones, decisions, verification, blockers, and the next
task here.

### Checklist

- [x] Update the plan before implementation with this phase’s seams and RNG contract
- [x] Move public SCE configuration/result types into `api.rs` and preserve root re-exports
- [x] Move optimiser implementation and `wtsne()` into a dedicated SCE module
- [x] Refactor distance code into a module directory with private gene parsing
- [x] Move CLI parsing/execution behind a dedicated binary-only CLI module
- [x] Replace manual RNG state/transition code with `rand_xoshiro`
- [x] Add reproducibility and changed-seed regression coverage
- [x] Run focused tests, formatting, lint, build, documentation, and diff checks
- [x] Update this plan at completion with status, decisions, blockers, log, and next task

### Current Status

Working on:

Module refactoring and library-backed RNG migration are complete.

Last verified:

2026-08-07: `cargo test --all-targets --offline` (28 tests),
`cargo build --all-targets --offline`, `cargo doc --no-deps --offline`,
`cargo fmt --all`, `cargo clippy --all-targets --offline -- -D warnings`, and
`git diff --check` pass. The online Cargo attempt was blocked by unavailable
crates.io DNS; all required dependencies were available in the local cache.

---

# Design Notes

## Parallelism

- Prefer Rayon.
- Atomic floating-point updates are required for embedding writes; use a CAS
  loop over `AtomicU64` bit patterns and retain the C++ optimistic clash retry.
- Benchmark against the C++ implementation after correctness is established.

## RNG

- Use `rand_xoshiro::Xoshiro256PlusPlus` through its `RngCore` and
  `SeedableRng` interfaces.
- Derive independent domain-separated `seed_from_u64` streams from the
  configured seed for each logical worker and each initial embedding point.
- Preserve same-seed reproducibility, but intentionally do not preserve the
  previous manual-generator embedding values; this is a clean seeded-output
  compatibility break chosen for the phase.

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
- No Python bindings or CUDA are part of this milestone.
- Distance constructors return `SparseDistances` and accept
  `Sparsification::Knn` or `Sparsification::Threshold` where the source
  supports that mode.

## Distance-input phase decisions

- All three sources return a public `SparseDistances` value containing sample
  names and zero-based COO row, column, and normalized-distance vectors.
- Use released crates.io `sketchlib = "0.4.1"` for file-compatible `.skm`/`.skd`
  loading and sparse kNN distances.
- Pair-SNP alignment input retains legacy pairsnp self edges and tie behavior.
  Accessory input retains legacy sklearn behavior (no self edges, exact kNN,
  strict threshold filtering).
- Sketch inputs support kNN only in this phase. Threshold mode is rejected for
  sketches because the released API does not provide a streaming threshold
  operation and dense all-pairs materialization would defeat sparse input.
- The CLI accepts one source at a time, runs `wtsne`, and writes
  `<output>.embedding.txt` plus `<output>.names.txt`; plotting and clustering
  remain outside this phase.
- Use the existing `rust/src/gene.rs` parsing semantics (IUPAC overlap,
  unknown-base matching, and its tested gap handling) rather than duplicating
  the parser.
- For multi-kmer sketch accessory distances, consume sketchlib's streamed
  all-pairs output into bounded per-row top-k candidates, avoiding a dense
  distance matrix while retaining accessory-specific neighbour selection.
- Single-kmer sketch databases use sketchlib's Jaccard sparse path; accessory
  selection is rejected unless at least two k-mer lengths are available.

## Module seams for the refactor

- `api.rs` owns public SCE configuration/result types; `lib.rs` remains a thin
  facade that re-exports existing root paths and the new `api` module.
- `sce.rs` owns `wtsne()` and optimiser implementation details, including
  validation, probabilities, sampling, atomic embedding updates, and frames.
- `distances/mod.rs` owns the current distance interface and implementations;
  `distances/gene.rs` remains private to that module.
- `main.rs` becomes a minimal entry point; binary-only argument parsing,
  source dispatch, embedding invocation, and output writing move to `cli.rs`.
- The public interface is tested through root re-exports and CLI behaviour;
  internal seams remain private implementation details.
- The clean RNG migration derives worker seeds from
  `seed + WORKER_STREAM_DOMAIN + index * STREAM_MULTIPLIER` and initial-point
  seeds from the analogous `INITIAL_STREAM_DOMAIN` stream. Unit sampling uses
  the full-width `next_u64` output over `[0, 1)`. This retains deterministic
  same-seed runs while intentionally changing fixed-seed embedding values from
  the former hand-written xoshiro128+ implementation.

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

- SIMD opportunities? Ideally compiler optimised rather than 'hand-optimised'.

---

# Next Task

After this refactor and RNG phase is complete:

- Benchmark the Rust optimiser against the C++ CPU implementation and measure
  peak memory for the distance builders on representative fixtures.

---

# Session Log

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
- Distance-input phase (completed in the following session-log entry)

Blockers:
- None for the `SceResults` phase; the C++ performance comparison remains
  pending.

## 2026-08-07 (distance-input phase)

Completed:
- Started the phase by updating `plan.md` before code changes with the shared
  sparse interface, source compatibility rules, sketch threshold decision,
  CLI outputs, fixture organization, and verification checklist.
- Added `SparseDistances`, alignment and accessory constructors, sketchlib
  database/FASTA-list support, the embedding CLI, organized fixtures, and
  focused source/CLI integration tests.
- Preserved alignment self-edge/tie behavior and accessory no-self/exact-kNN/
  strict-threshold behavior; sketch threshold mode reports a clear error.
- Finished the phase by updating this plan with implementation decisions,
  verification evidence, blockers, and the next task.

Discovered:
- The existing `rust/src/gene.rs` contains the alignment bitmap/parser logic to
  reuse. The placeholder executable was replaced, and the supplied fixtures are
  organized under `rust/tests/fixtures`.

Next session:
- Refactor the crate into separate optimiser, distance, API, and CLI modules;
  then evaluate replacing the manual RNG with the planned xoshiro library.

Blockers:
- None for this phase. Full-scale C++ performance comparison and real-dataset
  peak-memory benchmarking remain follow-up work.

## 2026-08-07 (module/RNG refactor phase)

Completed:
- Started the phase by updating `plan.md` before code changes with the module
  seams, public re-export contract, clean reseeding decision, checklist, and
  verification requirements.
- Moved the public SCE types into `src/api.rs`, optimiser internals into
  `src/sce.rs`, distance code into `src/distances/`, and the executable logic
  into binary-only `src/cli.rs` with a minimal `src/main.rs`.
- Replaced the hand-written xoshiro state machine with
  `rand_xoshiro::Xoshiro256PlusPlus`, preserving deterministic same-seed
  streams and adding changed-seed divergence plus public `api` module tests.
- Ran formatting, 28 tests, Clippy, build, documentation, and diff checks
  successfully using Cargo's offline cache.

Discovered:
- The current optimiser, public result/configuration types, and manual RNG are
  concentrated in `rust/src/lib.rs`; distance construction is already isolated
  enough to become a directory module, while the CLI remains in `main.rs`.

Next session:
- Benchmark the Rust optimiser against the C++ CPU implementation and measure
  peak memory for the distance builders on representative fixtures.

Blockers:
- Online crates.io resolution was unavailable in the sandbox, so verification
  used `--offline`; no dependency was missing from the local cache.
