//! Test persistent backend creation and execution

#[cfg(all(feature = "cuda", feature = "cuda-codegen"))]
use ringkernel_wavesim3d::simulation::{
    grid3d::SimulationGrid3D,
    persistent_backend::{PersistentBackend, PersistentBackendConfig},
    AcousticParams3D, Environment,
};

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    tracing::info!("testing persistent backend creation and execution");

    #[cfg(all(feature = "cuda", feature = "cuda-codegen"))]
    {
        // Use small grid that fits cooperative limits (48 blocks max)
        // 24x24x24 with 8x8x8 tiles = 3x3x3 = 27 blocks
        let width = 24;
        let height = 24;
        let depth = 24;

        tracing::info!(width, height, depth, "creating grid");
        let params = AcousticParams3D::new(Environment::default(), 0.1);
        let mut grid = SimulationGrid3D::new(width, height, depth, params);

        tracing::info!("creating PersistentBackendConfig");
        let config = PersistentBackendConfig::default();
        tracing::debug!(tile_size = ?config.tile_size, use_cooperative = config.use_cooperative, "config details");

        tracing::info!("creating PersistentBackend");
        match PersistentBackend::new(&grid, config) {
            Ok(mut backend) => {
                tracing::info!("backend created successfully");
                let stats = backend.stats();
                tracing::debug!(is_running = stats.is_running, "initial state");

                // Test starting the kernel
                tracing::info!("starting kernel");
                if let Err(e) = backend.start() {
                    tracing::error!(error = %e, "failed to start kernel");
                    return;
                }
                tracing::info!("kernel started");
                let stats = backend.stats();
                tracing::debug!(is_running = stats.is_running, "post-start state");

                // Small delay for kernel to initialize
                std::thread::sleep(std::time::Duration::from_millis(100));
                let stats = backend.stats();
                tracing::debug!(
                    current_step = stats.current_step,
                    steps_remaining = stats.steps_remaining,
                    "state after 100ms initialization delay"
                );

                // Test running steps - first batch
                tracing::info!(num_steps = 10, batch = 1, "running steps");
                match backend.step(&mut grid, 10) {
                    Ok(()) => {
                        let stats = backend.stats();
                        tracing::info!(
                            current_step = stats.current_step,
                            steps_remaining = stats.steps_remaining,
                            messages_processed = stats.messages_processed,
                            "batch 1 completed"
                        );
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "batch 1 failed");
                        let stats = backend.stats();
                        tracing::debug!(
                            current_step = stats.current_step,
                            steps_remaining = stats.steps_remaining,
                            has_terminated = stats.has_terminated,
                            "state after failure"
                        );
                    }
                }

                // Test running more steps - second batch
                tracing::info!(num_steps = 50, batch = 2, "running steps");
                match backend.step(&mut grid, 50) {
                    Ok(()) => {
                        let stats = backend.stats();
                        tracing::info!(
                            current_step = stats.current_step,
                            messages_processed = stats.messages_processed,
                            "batch 2 completed"
                        );
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "batch 2 failed");
                    }
                }

                // Test running yet more steps - third batch
                tracing::info!(num_steps = 100, batch = 3, "running steps");
                match backend.step(&mut grid, 100) {
                    Ok(()) => {
                        let stats = backend.stats();
                        tracing::info!(
                            current_step = stats.current_step,
                            messages_processed = stats.messages_processed,
                            "batch 3 completed"
                        );
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "batch 3 failed");
                    }
                }

                // Shutdown
                tracing::info!("shutting down");
                if let Err(e) = backend.shutdown() {
                    tracing::error!(error = %e, "shutdown failed");
                }
                tracing::info!("shutdown complete");
            }
            Err(e) => {
                tracing::error!(error = %e, "failed to create persistent backend");
            }
        }
    }

    #[cfg(not(all(feature = "cuda", feature = "cuda-codegen")))]
    {
        tracing::warn!("persistent backend requires both 'cuda' and 'cuda-codegen' features");
    }
}
