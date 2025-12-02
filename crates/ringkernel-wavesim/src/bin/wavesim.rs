//! WaveSim - Interactive 2D Acoustic Wave Propagation
//!
//! A showcase application demonstrating RingKernel's GPU-native actor model
//! through an interactive wave simulation.
//!
//! ## Usage
//!
//! ```bash
//! cargo run -p ringkernel-wavesim --bin wavesim
//! ```

use ringkernel_wavesim::gui;

fn main() -> iced::Result {
    // Initialize tracing for debug output
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("ringkernel_wavesim=info".parse().unwrap()),
        )
        .init();

    tracing::info!("Starting RingKernel WaveSim...");

    gui::run()
}
