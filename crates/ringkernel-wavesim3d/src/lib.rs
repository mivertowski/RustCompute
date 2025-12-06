//! RingKernel WaveSim3D - 3D Acoustic Wave Simulation
//!
//! A comprehensive 3D acoustic wave simulation with realistic physics,
//! binaural audio, and GPU acceleration.
//!
//! # Features
//!
//! - **3D FDTD Simulation**: Full 3D wave propagation using finite-difference
//!   time-domain methods with a 7-point stencil
//!
//! - **Realistic Physics**:
//!   - Temperature-dependent speed of sound
//!   - Humidity-dependent atmospheric absorption
//!   - Frequency-dependent damping (ISO 9613-1)
//!   - Multiple media: air, water, steel, aluminum
//!
//! - **Binaural Audio**:
//!   - Virtual head with configurable ear spacing
//!   - Interaural Time Difference (ITD) simulation
//!   - Interaural Level Difference (ILD) / head shadow
//!   - Real-time stereo output
//!
//! - **Audio Sources**:
//!   - Point impulses and spherical impulses
//!   - Continuous tones (sinusoidal)
//!   - Noise (white/pink)
//!   - Chirp (frequency sweep)
//!   - Gaussian pulses
//!   - Audio file playback
//!
//! - **GPU Acceleration**:
//!   - CUDA backend for NVIDIA GPUs
//!   - 3D stencil operations
//!   - Efficient packed buffer layout
//!
//! - **3D Visualization**:
//!   - Slice rendering (XY, XZ, YZ planes)
//!   - Interactive camera controls
//!   - Source and listener markers
//!   - Real-time pressure field display
//!
//! # Example
//!
//! ```rust,no_run
//! use ringkernel_wavesim3d::prelude::*;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create simulation with default parameters
//! let config = SimulationConfig::small_room();
//! let mut engine = config.build();
//!
//! // Inject an impulse at the center
//! let (w, h, d) = engine.dimensions();
//! engine.inject_impulse(w/2, h/2, d/2, 1.0);
//!
//! // Run simulation steps
//! for _ in 0..100 {
//!     engine.step();
//! }
//!
//! println!("Simulated {} seconds", engine.time());
//! # Ok(())
//! # }
//! ```
//!
//! # Modules
//!
//! - [`simulation`]: Core simulation engine, physics, and grid
//! - [`audio`]: Audio sources and binaural microphone
//! - [`visualization`]: 3D rendering and camera controls

pub mod audio;
pub mod gui;
pub mod simulation;
pub mod visualization;

/// Re-exports for convenient access.
pub mod prelude {
    pub use crate::audio::{
        AudioConfig, AudioSource, AudioSystem, BinauralMicrophone, BinauralProcessor,
        SourceManager, SourceType, VirtualHead,
    };
    pub use crate::simulation::{
        AcousticParams3D, CellType, Environment, Medium, MediumProperties, Orientation3D,
        Position3D, SimulationConfig, SimulationEngine, SimulationGrid3D,
    };
    pub use crate::visualization::{
        Camera3D, CameraController, ColorMap, RenderConfig, SliceRenderer, VisualizationMode,
    };
}

pub use prelude::*;
