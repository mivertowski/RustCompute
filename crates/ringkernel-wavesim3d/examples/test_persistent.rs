//! Test persistent backend creation and execution
use ringkernel_wavesim3d::simulation::{grid3d::SimulationGrid3D, AcousticParams3D, Environment};

#[cfg(all(feature = "cuda", feature = "cuda-codegen"))]
use ringkernel_wavesim3d::simulation::persistent_backend::{
    PersistentBackend, PersistentBackendConfig,
};

fn main() {
    println!("Testing Persistent Backend Creation and Execution...\n");

    #[cfg(all(feature = "cuda", feature = "cuda-codegen"))]
    {
        // Use small grid that fits cooperative limits (48 blocks max)
        // 24×24×24 with 8×8×8 tiles = 3×3×3 = 27 blocks
        let width = 24;
        let height = 24;
        let depth = 24;

        println!("Creating grid {}x{}x{}", width, height, depth);
        let params = AcousticParams3D::new(Environment::default(), 0.1);
        let mut grid = SimulationGrid3D::new(width, height, depth, params);

        println!("Creating PersistentBackendConfig...");
        let config = PersistentBackendConfig::default();
        println!("  tile_size: {:?}", config.tile_size);
        println!("  use_cooperative: {}", config.use_cooperative);

        println!("\nCreating PersistentBackend...");
        match PersistentBackend::new(&grid, config) {
            Ok(mut backend) => {
                println!("SUCCESS! Backend created!");
                let stats = backend.stats();
                println!("  is_running: {}", stats.is_running);

                // Test starting the kernel
                println!("\nStarting kernel...");
                if let Err(e) = backend.start() {
                    println!("  Failed to start: {}", e);
                    return;
                }
                println!("  Kernel started!");
                let stats = backend.stats();
                println!("  is_running: {}", stats.is_running);

                // Small delay for kernel to initialize
                std::thread::sleep(std::time::Duration::from_millis(100));
                let stats = backend.stats();
                println!(
                    "  After 100ms - current_step: {}, steps_remaining: {}",
                    stats.current_step, stats.steps_remaining
                );

                // Test running steps - first batch
                println!("\nRunning 10 steps (batch 1)...");
                match backend.step(&mut grid, 10) {
                    Ok(()) => {
                        println!("  SUCCESS! Steps completed.");
                        let stats = backend.stats();
                        println!("  current_step: {}", stats.current_step);
                        println!("  steps_remaining: {}", stats.steps_remaining);
                        println!("  messages_processed: {}", stats.messages_processed);
                    }
                    Err(e) => {
                        println!("  FAILED: {}", e);
                        let stats = backend.stats();
                        println!("  current_step: {}", stats.current_step);
                        println!("  steps_remaining: {}", stats.steps_remaining);
                        println!("  has_terminated: {}", stats.has_terminated);
                    }
                }

                // Test running more steps - second batch
                println!("\nRunning 50 steps (batch 2)...");
                match backend.step(&mut grid, 50) {
                    Ok(()) => {
                        println!("  SUCCESS! Steps completed.");
                        let stats = backend.stats();
                        println!("  current_step: {}", stats.current_step);
                        println!("  messages_processed: {}", stats.messages_processed);
                    }
                    Err(e) => {
                        println!("  FAILED: {}", e);
                    }
                }

                // Test running yet more steps - third batch
                println!("\nRunning 100 steps (batch 3)...");
                match backend.step(&mut grid, 100) {
                    Ok(()) => {
                        println!("  SUCCESS! Steps completed.");
                        let stats = backend.stats();
                        println!("  current_step: {}", stats.current_step);
                        println!("  messages_processed: {}", stats.messages_processed);
                    }
                    Err(e) => {
                        println!("  FAILED: {}", e);
                    }
                }

                // Shutdown
                println!("\nShutting down...");
                if let Err(e) = backend.shutdown() {
                    println!("  Shutdown error: {}", e);
                }
                println!("  Done.");
            }
            Err(e) => {
                println!("FAILED: {}", e);
            }
        }
    }

    #[cfg(not(all(feature = "cuda", feature = "cuda-codegen")))]
    {
        println!("Persistent backend requires both 'cuda' and 'cuda-codegen' features");
    }
}
