Read agents.md. Write and update this plan as you progress through each
step.

## Definition of Done

- [ ] Builds with `cargo build`
- [ ] Tests pass
- [ ] Public API documented
- [ ] Benchmark within 5% of C++ implementation
- [ ] plan.md updated


# Current aim

1. Create a function `wtsne` which takes the sparse distance vectors,
   and returns an embedding.

## wtsne notes (1)

- Parallelise the function, ideally using rayon. Take care for the race
condition when updating the embedding (C++ uses omp atomic update to
guard these lines), so a suitable atomic is likely needed here.
- Use a suitable progress bar.
- Use an RNG which is compatible with parallel threads
- Conditional probabilities as in `wtsne.hpp` will also need to be
  implemented, and parallelised.

# Next aim

2. Create a version of the `sce_results.hpp` object `sce_results` which
   outputs intermediate embedding states at regular 'frames' which are
   either exponentially distributed in time (more frames early on), or
   evenly distributed through time.


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
- CLI

---

# Current Task

## Objective

Implement the Rust `wtsne()` function.

### Checklist

- [ ] Sparse graph/vector input
- [ ] Embedding allocation
- [ ] Optimisation loop
- [ ] Parallel updates
- [ ] Progress reporting
- [ ] Parallel RNG
- [ ] Conditional probabilities
- [ ] Tests

### Current Status

Working on:

...

Last verified:

...

---

# Design Notes

## Parallelism

- Prefer Rayon.
- Atomic updates required for embedding writes.
- Investigate alternatives to atomics if performance suffers.

## RNG

- Thread-safe RNG compatible with Rayon.

## Progress

Use indicatif.

## Conditional probabilities

Port implementation from `wtsne.hpp`.

---

# Open Questions

- Best representation for sparse graph?
- Is atomic update the fastest approach?
- SIMD opportunities? Ideally compiler optimised rather than
  'hand-optimised'

---

# Next Task

After `wtsne` is complete:

Implement `sce_results`.

Requirements:

- intermediate embeddings
- configurable frame count
- exponential or linear frame spacing

---

# Session Log

## 2026-08-06

Completed:
- ...

Discovered:
- ...

Next session:
- ...
