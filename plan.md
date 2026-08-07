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

- [] Builds with `cargo build`
- [] Tests pass
- [] Public API documented
- [] plan.md updated

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

Returns an `ndarray` embedding.

Expose:

- Rust library
- CLI (deferred until an input/output file contract is defined)

---

# Current Task

## Objective

Implement the Rust v1 `wtsne()` library function: accept C++-compatible COO
vectors and return a final 2D `ndarray::Array2<f64>` embedding. Animation
(`sce_results`) and the end-user CLI remain deferred to later tasks.

### Checklist

- [x] Sparse graph/vector input
- [x] Embedding allocation
- [x] Optimisation loop
- [x] Parallel updates
- [x] Progress reporting
- [x] Parallel RNG
- [x] Conditional probabilities
- [x] Tests

### Current Status

Working on:

Completed v1 Rust `wtsne` library implementation and invariant tests.

Last verified:

2026-08-07: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
`cargo test` (6 tests), `cargo build`, and `git diff --check` pass.

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
- `wtsne(...)` returns `Result<ndarray::Array2<f64>>` with shape `(n_nodes, 2)`.
- `WtsneOptions` carries perplexity, iteration count, repulsion samples,
  learning rate, initial exaggeration, worker count, progress setting, and seed.
- Tests target public behavior: validation, normalization, finite output,
  reproducibility for one worker, and successful parallel execution.
- No Python bindings, CUDA, animation frames, or file-format CLI are part of
  this milestone.

---

# Open Questions

- Benchmark the atomic update implementation against the C++ CPU version and
  investigate alternatives if it misses the 5% target.
- SIMD opportunities? Ideally compiler optimised rather than 'hand-optimised'.

---

# Next Task

After `wtsne` is complete:

Implement `sce_results`.

Requirements:

- intermediate embeddings
- configurable frame count
- exponential or linear frame spacing

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
- Implement `sce_results` with configurable linear or exponentially spaced
  intermediate embedding frames, then add its public API and tests.
