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

Add a separate PyO3-backed visualization workflow for the completed Rust
embedding: static output, interactive Plotly HTML, density output, HDBSCAN
colouring, and optional animation. Preserve the existing embedding CLI and
keep all Python integration behind a default-enabled `PyO3` feature that can be
disabled for non-Python and future wasm builds.

Before code changes, this task updates this plan. Before finishing, it records
completed milestones, design decisions, verification, blockers, and the next
task here.

Before code changes, this task must update this plan. Before finishing, it must
record completed milestones, decisions, verification, blockers, and the next
task here.

### Checklist

- [x] Update the plan before implementation with the visualization seam, feature contract, and archive workflow
- [x] Add default-enabled `PyO3` feature and compile-out path for non-Python builds
- [x] Add `--save-animation` frame capture and compressed NPZ archive output
- [x] Add separate `plot` CLI command with prefix-based input discovery
- [x] Add embedded Python plotting adapter for static, HTML, density, HDBSCAN, and MP4 outputs
- [x] Add labels, clustering, hover-label, and deterministic-colour behavior matching the Python reference
- [x] Add actionable dependency, input, archive, and FFmpeg errors
- [x] Add CLI, archive, PyO3 seam, and visualization artifact coverage
- [x] Run default-feature and `--no-default-features` formatting, tests, lint, build, docs, and diff checks
- [x] Update this plan at completion with status, decisions, blockers, log, and next task

### Current Status

Working on:

The PyO3 visualization workflow is complete.

Last verified:

2026-08-07: default-feature `cargo test --all-targets --offline` (30 tests),
Clippy with warnings denied, build, docs, formatting, and diff checks pass.
`--no-default-features` tests (28 tests), Clippy, formatting, and diff checks
also pass. Direct `intel_pp` Python smoke tests produced HTML, static PNG,
density PDF, NPZ frames, HDBSCAN CSV, and MP4 outputs.

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
- No Python extension bindings or CUDA are part of this milestone; PyO3 is an
  executable-only embedding adapter.
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

## Visualization phase design

- Keep the existing root embedding invocation and add a separate
  `mandrake plot --input-prefix PREFIX` command. The command discovers
  `<prefix>.embedding.txt`, `<prefix>.names.txt`, and an optional
  `<prefix>.embedding_frames.npz`; `--output` defaults to the input prefix.
- Add a default-enabled Cargo feature named exactly `PyO3`, backed by
  `pyo3 = "0.21.2"` with `auto-initialize`. Plotting code, its CLI arguments,
  and the Python dependency are compiled out with `--no-default-features`.
- `--save-animation` on the embedding command retains up to 400 exponentially
  scheduled frames, capped at `max_iterations + 1`, and writes a compressed
  NPZ archive with version, embeddings, iteration, worker-update, and Eq arrays.
  `plot --animate` requires this archive and reports how to create it when it
  is absent.
- A binary-only Rust visualization module embeds a self-contained Python asset
  based on the original `mandrake/plot.py` and `clustering.py`. A small PyO3
  adapter converts validated Rust arrays and invokes the Python HTML, static,
  density, HDBSCAN, and animation functions without exposing Python types in
  the Rust library interface.
- Visualization writes `.embedding_static.png`, `.embedding_density.pdf`,
  `.embedding.html`, an HDBSCAN CSV when automatic clustering runs, and an
  `.embedding_animation.mp4` when animation is requested. Animation sound is
  out of scope for this phase.
- Labels remain the original headerless name/tab/label format and must cover
  the embedding names exactly without duplicate keys. Supplied labels override
  HDBSCAN; otherwise the original HDBSCAN parameters run unless
  `--no-clustering` is set. Colours remain deterministic from `--seed`, sample
  names remain optional HTML hover labels, and HDBSCAN noise remains black.
- Python runtime requirements are Python 3.8+, NumPy, Pandas, Plotly,
  Matplotlib, HDBSCAN, tqdm, and FFmpeg, matching `environment.yml`. Missing
  imports or FFmpeg are surfaced as contextual Rust errors.
- PyO3 uses `abi3-py38` in addition to `auto-initialize`, allowing the cached
  PyO3 0.21.2 build to configure against newer host Python versions. The Rust
  arm64 build is restored with the arm64 system Python; the requested
  `intel_pp` environment is x86_64 and is therefore used for direct Python
  artifact smoke tests rather than linked into the arm64 binary.
- `intel_pp` already contained the scientific Python packages. Its legacy
  FFmpeg 2.8/x264 mismatch was repaired with `ffmpeg>=4` and `x264>=1!164`,
  yielding FFmpeg 5.1.2 and x264 164.3095 for MP4 verification.

---

# Open Questions

- SIMD opportunities? Ideally compiler optimised rather than 'hand-optimised'.

---

# Next Task

After this visualization phase is complete:

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

## 2026-08-07 (PyO3 visualization phase)

Started:
- Started the phase by rereading `AGENTS.md` and `plan.md`, then updating this
  plan before implementation with the CLI workflow, PyO3 feature contract,
  frame archive schema, plotting outputs, and verification requirements.

Discovered:
- The original plotting implementation is in `mandrake/plot.py`, with HDBSCAN
  behavior in `mandrake/clustering.py`; it imports scientific Python packages
  that are absent from the system interpreter in this checkout.
- The existing Rust CLI writes only final embedding and names files, so a
  separate animation command requires an explicit sampled-frame archive.

During implementation:
- Implement the feature-gated PyO3 adapter, NPZ frame archive, plot command,
  embedded Python plotting asset, and focused tests; then complete this entry
  with verification evidence and any dependency blockers.

Initial blocker:
- The configured system Python currently lacks NumPy, Pandas, Plotly,
  Matplotlib, HDBSCAN, and FFmpeg. Artifact smoke tests require the project
  Python environment described by `environment.yml`.

Completed:
- Added the default `PyO3` feature, preserving a clean `--no-default-features`
  build for future wasm/non-Python targets.
- Added `--save-animation` sampled-frame capture and NPZ persistence, plus the
  prefix-based `plot` subcommand with strict embedding/name/label validation.
- Added the embedded plotting adapter and Python reference implementation for
  deterministic labels, HDBSCAN, Plotly HTML, static/density plots, and MP4
  animations.
- Verified the Python asset in the repaired `intel_pp` environment, including
  all five expected artifact types and automatic HDBSCAN output.
- Ran default and feature-off Rust tests, Clippy, build, docs, formatting, and
  diff checks successfully.

Discovered:
- The requested `intel_pp` environment is x86_64 while the Rust toolchain and
  host are arm64. Linking PyO3 to that environment fails with architecture
  mismatch, so the Rust binary's in-process Python artifact smoke test remains
  blocked even though direct Python rendering succeeds.

Next session:
- Benchmark the Rust optimiser against the C++ CPU implementation and measure
  peak memory for the distance builders on representative fixtures.

Blockers:
- No source or test blocker remains. A native arm64 Python environment with the
  plotting packages is required to run the full Rust-to-Python artifact smoke
  test; the direct `intel_pp` smoke test is the available evidence for Python
  behavior.
