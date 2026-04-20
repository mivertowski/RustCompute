# ringkernel-montecarlo

Monte Carlo simulation primitives for the RingKernel workspace.

## Overview

`ringkernel-montecarlo` provides building blocks for Monte Carlo workloads that
target GPU execution through `ringkernel-core`. The focus is on
counter-based PRNGs (suitable for stateless parallel generation) and classical
variance-reduction techniques.

Included:

- **Philox RNG** (`PhiloxRng`, `PhiloxState`, `GpuRng`) — counter-based
  generator that is stateless per call and reproducible across CPU and GPU.
- **Antithetic variates** (`AntitheticVariates`, `antithetic_pair`) —
  negatively correlated sample pairs for variance reduction.
- **Control variates** (`ControlVariates`, `control_variate_estimate`) —
  bias-free estimator using a correlated auxiliary statistic.
- **Importance sampling** (`ImportanceSampling`, `importance_sample`) —
  weighted sampling from a proposal distribution.

A `prelude` module re-exports the common types for convenient imports.

## Relationship to `ringkernel-core`

This crate depends on `ringkernel-core` for shared error types and runtime
integration. CPU paths are always available; GPU kernels live behind the
`cuda` feature and reuse the `ringkernel-cuda` backend when enabled upstream.

## Feature flags

| Flag | Default | Description                                               |
|------|---------|-----------------------------------------------------------|
| `cuda` | off   | Enable CUDA kernels via `cudarc` (requires a CUDA install) |

Without `cuda`, only the CPU reference implementations are compiled.

## Example

Estimate pi by sampling points in the unit square with a Philox RNG:

```rust
use ringkernel_montecarlo::prelude::*;

fn main() {
    // Deterministic seed (key, counter_base).
    let mut rng = PhiloxRng::new(0, 42);

    let n = 1_000_000;
    let mut inside = 0u64;

    for _ in 0..n {
        let x: f32 = rng.next_uniform();
        let y: f32 = rng.next_uniform();
        if x * x + y * y <= 1.0 {
            inside += 1;
        }
    }

    let pi_estimate = 4.0 * (inside as f64) / (n as f64);
    println!("pi ≈ {pi_estimate}");
}
```

For variance reduction, pair each sample with its antithetic counterpart via
`antithetic_pair(&mut rng)`.

## Workspace context

For an overview of how this crate fits into the wider RingKernel workspace, see
[`docs/02-crate-structure.md`](../../docs/02-crate-structure.md).

## Status

v1.1.0 — part of the RingKernel 1.1 release. API is considered stable within
the 1.x series; breaking changes will follow semver.

## License

Apache-2.0. See the repository-level `LICENSE` file.
