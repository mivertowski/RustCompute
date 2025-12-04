# RingKernel WaveSim

Interactive 2D acoustic wave propagation simulator showcasing RingKernel's GPU compute and actor model capabilities.

![WaveSim Screenshot](../../docs/screenshots/wavesim.png)

## Overview

WaveSim implements a Finite-Difference Time-Domain (FDTD) solver for the 2D wave equation, demonstrating several RingKernel features:

- **Tile-based Actor Model**: 16×16 cell tiles as actors with K2K messaging for halo exchange
- **Multiple Backends**: CPU (SoA + SIMD + Rayon), CUDA, and WGPU
- **GPU-Only Halo Exchange**: Zero host transfers during simulation (CUDA Packed backend)

## Performance

| Backend | 256×256 | 512×512 | Notes |
|---------|---------|---------|-------|
| CPU SimulationGrid | 35,418 steps/s | 7,229 steps/s | SoA + SIMD + Rayon |
| CPU TileKernelGrid | 1,357 steps/s | — | Actor model with K2K |
| **CUDA Packed** | **112,837 steps/s** | **71,324 steps/s** | GPU-only halo exchange |

**GPU vs CPU speedup: 3.1x at 256×256, 9.9x at 512×512**

See [PERFORMANCE.md](PERFORMANCE.md) for detailed analysis.

## Quick Start

### Interactive GUI

```bash
# CPU backend (default)
cargo run -p ringkernel-wavesim --release

# With GPU compute (requires wgpu feature)
cargo run -p ringkernel-wavesim --release --features wgpu
```

Click anywhere on the canvas to inject wave impulses.

### Benchmarks

```bash
# Full benchmark suite (CPU, CUDA, WGPU)
cargo run -p ringkernel-wavesim --bin full_benchmark --release --features "cuda,wgpu"

# CUDA Packed backend benchmark (GPU-only halo exchange)
cargo run -p ringkernel-wavesim --bin bench_packed --release --features cuda

# Verify CUDA correctness against CPU
cargo run -p ringkernel-wavesim --bin verify_packed --release --features cuda
```

## Architecture

### Backends

1. **CPU SimulationGrid**: Optimized SoA layout with SIMD (f32x8) and Rayon parallelization. Best for small-to-medium grids.

2. **CPU TileKernelGrid**: 16×16 tile actors demonstrating RingKernel's K2K messaging. Each tile exchanges halo data with neighbors.

3. **CUDA Packed** (recommended for large grids): All tiles packed in a single GPU buffer. Halo exchange happens entirely on GPU via memory copies—zero host transfers during simulation.

### Memory Layout (CUDA Packed)

```
GPU Buffer: [Tile(0,0)][Tile(1,0)]...[Tile(n,m)]

Each tile: 18×18 floats
  ┌───┬────────────────┬───┐
  │ NW│  North Halo    │NE │  ← Row 0 (from neighbor)
  ├───┼────────────────┼───┤
  │ W │  16×16 Interior│ E │  ← Owned cells
  ├───┼────────────────┼───┤
  │ SW│  South Halo    │SE │  ← Row 17 (from neighbor)
  └───┴────────────────┴───┘
```

### Simulation Step (CUDA Packed)

```
1. exchange_all_halos kernel  ─┐
   (GPU-to-GPU memory copies)  │  Zero host transfers
2. apply_boundary_conditions   │  Only 2-3 kernel launches
3. fdtd_all_tiles kernel      ─┘
4. Swap buffer pointers (host-side, trivial)
```

## Features

| Feature | Description |
|---------|-------------|
| `cpu` | CPU backend (default) |
| `cuda` | NVIDIA CUDA backend |
| `wgpu` | WebGPU cross-platform backend |
| `simd` | SIMD optimizations (requires nightly) |
| `all-backends` | Enable all GPU backends |

## Files

```
src/
├── simulation/
│   ├── grid.rs           # CPU SimulationGrid (SoA + SIMD)
│   ├── tile_grid.rs      # TileKernelGrid (actor model)
│   ├── cuda_packed.rs    # CUDA Packed backend
│   ├── cuda_compute.rs   # CUDA per-tile backend
│   └── wgpu_compute.rs   # WGPU backend
├── shaders/
│   ├── fdtd_packed.cu    # CUDA kernels (packed layout)
│   └── fdtd_tile.cu      # CUDA kernels (per-tile)
└── bin/
    ├── wavesim.rs        # Interactive GUI
    ├── benchmark.rs      # Quick benchmark
    ├── full_benchmark.rs # Comprehensive benchmark
    ├── bench_packed.rs   # CUDA Packed benchmark
    └── verify_packed.rs  # Correctness verification
```

## Physics

The simulation solves the 2D acoustic wave equation:

```
∂²p/∂t² = c² (∂²p/∂x² + ∂²p/∂y²) - γ·∂p/∂t
```

Where:
- `p` = pressure field
- `c` = speed of sound (343 m/s default)
- `γ` = damping coefficient

Discretized using central differences (FDTD):

```
p_new = 2p - p_prev + c²Δt²/Δx² · (p_N + p_S + p_E + p_W - 4p) - γ·(p - p_prev)
```

## License

Apache-2.0
