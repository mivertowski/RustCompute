//! GUI module for 3D wave simulation.
//!
//! Provides egui-based controls for:
//! - Simulation parameters
//! - Source placement
//! - Listener positioning
//! - Visualization settings

pub mod controls;

pub use controls::{GuiState, SimulationControls};
