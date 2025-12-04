//! Process Intelligence GUI Application
//!
//! Real-time process mining visualization with GPU-accelerated analysis.

use ringkernel_procint::gui::ProcessIntelligenceApp;

fn main() -> eframe::Result<()> {
    // Initialize logging
    env_logger::init();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     RingKernel Process Intelligence - GPU-Accelerated PM     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Starting GUI...");

    // Run the application
    ProcessIntelligenceApp::new().run()
}
