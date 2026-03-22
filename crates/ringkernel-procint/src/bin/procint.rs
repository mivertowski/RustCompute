//! Process Intelligence GUI Application
//!
//! Real-time process mining visualization with GPU-accelerated analysis.

use ringkernel_procint::gui::ProcessIntelligenceApp;

fn main() -> eframe::Result<()> {
    // Initialize logging
    env_logger::init();

    tracing::info!("╔══════════════════════════════════════════════════════════════╗");
    tracing::info!("║     RingKernel Process Intelligence - GPU-Accelerated PM     ║");
    tracing::info!("╚══════════════════════════════════════════════════════════════╝");
    tracing::info!("Starting GUI...");

    // Run the application
    ProcessIntelligenceApp::new().run()
}
