# WaveSim3D - 3D Acoustic Wave Simulation

A comprehensive 3D acoustic wave simulation with realistic physics, binaural audio, and GPU acceleration.

## Features

### Realistic 3D Physics
- **Temperature-dependent speed of sound**: Uses accurate formula `c = 331.3 * sqrt(1 + T/273.15)` with humidity correction
- **Frequency-dependent absorption**: Implements ISO 9613-1 atmospheric absorption model
- **Multiple propagation media**:
  - **Air**: Temperature and humidity-dependent properties
  - **Water**: Bilaniuk-Wong speed of sound formula
  - **Metal**: Steel and aluminum with proper material properties

### 3D Wave Simulation
- **FDTD (Finite-Difference Time-Domain)** wave equation solver
- **7-point stencil Laplacian** for accurate 3D propagation
- **CFL stability** automatically maintained: `dt ≤ dx / (c * sqrt(3))`
- **Absorbing boundary conditions** to minimize reflections
- Support for obstacles, reflectors, and custom geometries

### Audio System
- **Multiple source types**:
  - Impulse (click/clap)
  - Continuous tone (sine wave)
  - Chirp (frequency sweep)
  - White/pink noise
  - Gaussian pulse
  - WAV file playback
- **3D positioned sources** with real-time movement
- **Binaural microphone** with virtual head:
  - Realistic ear spacing (~17cm)
  - ITD (Interaural Time Difference) calculation
  - ILD (Interaural Level Difference) modeling
- **WAV file I/O** for recording and playback

### GPU Acceleration
- **CUDA backend** for NVIDIA GPUs
- **CPU fallback** with Rayon parallelization
- Efficient 3D packed stencil kernel

### 3D Visualization
- **wgpu-based rendering** (Vulkan, Metal, DX12)
- **Interactive camera** with orbit, pan, and zoom controls
- **Slice visualization**: XY, XZ, and YZ plane cuts
- **Color mapping**: Blue-White-Red pressure visualization
- **Source and listener markers**

## Quick Start

```bash
# Run with CPU backend
cargo run -p ringkernel-wavesim3d --bin wavesim3d --features cpu

# Run with CUDA acceleration (requires NVIDIA GPU)
cargo run -p ringkernel-wavesim3d --bin wavesim3d --features cuda
```

## Controls

| Key | Action |
|-----|--------|
| Space | Play/Pause simulation |
| R | Reset simulation |
| I | Inject impulse at source position |
| 1 | Toggle XY slice visibility |
| 2 | Toggle XZ slice visibility |
| 3 | Toggle YZ slice visibility |
| Left Mouse | Rotate camera |
| Right Mouse | Pan camera |
| Scroll | Zoom camera |
| Escape | Quit |

## Usage Example

```rust
use ringkernel_wavesim3d::simulation::{
    SimulationConfig, SimulationEngine, Environment, Position3D
};
use ringkernel_wavesim3d::audio::{AudioSystem, AudioSource, VirtualHead};

// Create simulation with custom environment
let config = SimulationConfig {
    width: 64,
    height: 32,
    depth: 64,
    cell_size: 0.1,  // 10cm cells
    environment: Environment {
        temperature_c: 20.0,
        humidity_percent: 50.0,
        medium: Medium::Air,
        ..Default::default()
    },
    prefer_gpu: true,
};

let mut engine = config.build();

// Add audio source
let mut audio = AudioSystem::new(Default::default());
let source = AudioSource::tone(
    0,
    Position3D::new(3.2, 1.6, 3.2),  // Source position
    440.0,  // Frequency (Hz)
    1.0,    // Amplitude
);
audio.add_source(source);

// Set up binaural microphone
let head = VirtualHead::new(Position3D::new(3.2, 1.6, 5.0));
audio.init_microphone(head, engine.grid.params.time_step);

// Run simulation steps
for _ in 0..1000 {
    engine.step();
    if let Some(mic) = &mut audio.microphone {
        mic.capture(&engine.grid);
    }
}

// Get binaural audio
if let Some(mic) = &audio.microphone {
    let (left, right) = mic.get_samples(1024);
    // Process or save stereo audio...
}
```

## Physics Model

### Wave Equation
The simulation solves the 3D acoustic wave equation:

```
∂²p/∂t² = c² ∇²p - α ∂p/∂t
```

Where:
- `p` is pressure
- `c` is speed of sound
- `∇²` is the 3D Laplacian
- `α` is damping coefficient

### FDTD Discretization
Using a 7-point stencil for the 3D Laplacian:

```
∇²p ≈ (p_W + p_E + p_S + p_N + p_D + p_U - 6p) / Δx²
```

Time stepping:
```
p_new = 2p - p_prev + c²Δt²∇²p
```

### Atmospheric Absorption (ISO 9613-1)
Frequency-dependent absorption in air:
- Oxygen relaxation (dominant ~10 kHz)
- Nitrogen relaxation (dominant ~100 Hz)
- Classical absorption (molecular viscosity)

## Configuration Options

### Environment
```rust
Environment {
    temperature_c: 20.0,      // Temperature in Celsius
    humidity_percent: 50.0,   // Relative humidity
    pressure_pa: 101325.0,    // Atmospheric pressure (Pa)
    medium: Medium::Air,      // Air, Water, or Metal
}
```

### Grid Size
- Recommended: 64x32x64 for real-time visualization
- Maximum frequency: `f_max ≈ c / (10 * cell_size)`
- For 10cm cells in air: ~343 Hz accurate simulation

### Binaural Audio
- Default ear spacing: 17cm (average human)
- Sample rate: 44.1 kHz
- ITD range: ±0.6ms (full lateralization)

## Benchmarks

| Configuration | Backend | Performance |
|---------------|---------|-------------|
| 64³ cells | CPU (Rayon) | ~120 steps/sec |
| 64³ cells | CUDA (RTX 3090) | ~2000 steps/sec |
| 128³ cells | CPU (Rayon) | ~15 steps/sec |
| 128³ cells | CUDA (RTX 3090) | ~400 steps/sec |

## Feature Flags

| Feature | Description |
|---------|-------------|
| `cpu` (default) | CPU backend with Rayon parallelization |
| `cuda` | NVIDIA CUDA acceleration |
| `audio-output` | Real-time audio output via cpal |

## Dependencies

- **wgpu**: Cross-platform graphics
- **winit**: Window management
- **egui**: GUI controls
- **glam**: 3D math
- **hound**: WAV file I/O
- **rayon**: CPU parallelization
- **cudarc**: CUDA bindings (optional)

## Related Crates

- `ringkernel-wavesim`: 2D wave simulation
- `ringkernel-cuda-codegen`: CUDA kernel generation

## License

Same license as the parent RingKernel project.
