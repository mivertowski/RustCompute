//! Accounting Network Analytics GUI application.
//!
//! GPU-accelerated visualization of double-entry bookkeeping as a network graph
//! with real-time fraud detection and GAAP compliance monitoring.

fn main() -> eframe::Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    log::info!("Starting Accounting Network Analytics");
    log::info!("Version: {}", ringkernel_accnet::VERSION);

    // Run the application
    ringkernel_accnet::gui::app::run()
}
