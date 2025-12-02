// Enable portable_simd feature for SIMD optimizations
#![cfg_attr(feature = "simd", feature(portable_simd))]

//! # RingKernel WaveSim
//!
//! Interactive 2D acoustic wave propagation showcase demonstrating RingKernel's
//! GPU-native actor model.
//!
//! Each cell in a 2D grid is a persistent kernel actor that communicates with
//! neighbors via K2K messaging to propagate pressure waves using the FDTD
//! (Finite-Difference Time-Domain) method.
//!
//! ## Features
//!
//! - Real-time acoustic wave simulation
//! - Click-to-inject impulse interaction
//! - Adjustable speed of sound
//! - Configurable grid size
//! - CPU/GPU backend toggle
//! - Boundary reflection and absorption
//!
//! ## Run
//!
//! ```bash
//! cargo run -p ringkernel-wavesim --bin wavesim
//! ```

pub mod gui;
pub mod simulation;

pub use gui::WaveSimApp;
pub use simulation::{AcousticParams, CellState, Direction, SimulationGrid};
