//! Real-time Audio Separator Example
//!
//! This example demonstrates real-time streaming audio separation
//! using the GPU bin actor network with K2K messaging.
//!
//! Usage:
//!   cargo run --example realtime_separator --features device-input
//!
//! Note: Requires a working audio input device.

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ringkernel_audio_fft::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("ringkernel_audio_fft=debug".parse().unwrap()),
        )
        .init();

    println!("=== RingKernel Real-time Audio Separator ===\n");

    // Configuration
    let fft_size = 1024;
    let hop_size = 256;
    let sample_rate = 44100;

    // Create bin network
    println!("Creating bin actor network...");
    let config = SeparationConfig::speech_preset();
    let mut bin_network = BinNetwork::new(fft_size / 2 + 1, config).await?;
    println!("  {} bin actors created", bin_network.num_bins());
    println!("  K2K messaging enabled between neighbors");
    println!();

    // Create synthetic input for demonstration (since we might not have device input)
    println!("Generating test signal for real-time demonstration...");

    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    // Handle Ctrl+C
    ctrlc_handler(running_clone);

    // Simulate real-time processing
    let mut fft = FftProcessor::new(fft_size, hop_size, sample_rate)?;
    let mut ifft_direct = IfftProcessor::new(fft_size, hop_size)?;
    let mut ifft_ambience = IfftProcessor::new(fft_size, hop_size)?;
    let mut mixer = DryWetMixer::new();
    mixer.set_dry_wet(0.3); // Mostly direct

    let mut frame_counter = 0u64;
    let mut total_latency = Duration::ZERO;
    let mut frame_count = 0u64;

    println!("\nProcessing (press Ctrl+C to stop)...\n");
    println!("Frame | Latency | Direct dB | Ambient dB | K2K msgs");
    println!("------|---------|-----------|------------|--------");

    let start = Instant::now();
    let block_duration = Duration::from_secs_f64(hop_size as f64 * 4.0 / sample_rate as f64);

    while running.load(Ordering::Relaxed) && start.elapsed() < Duration::from_secs(10) {
        let frame_start = Instant::now();

        // Generate synthetic audio block
        let samples = generate_audio_block(sample_rate, hop_size * 4, frame_counter);

        // Process through FFT
        for fft_frame in fft.process_all(&samples) {
            // Process through bin network
            let separated = bin_network
                .process_frame(frame_counter, &fft_frame, sample_rate, fft_size)
                .await?;

            // Extract direct and ambience
            let direct_bins: Vec<Complex> = separated.iter().map(|b| b.direct).collect();
            let ambience_bins: Vec<Complex> = separated.iter().map(|b| b.ambience).collect();

            // IFFT back to time domain
            let _direct_samples = ifft_direct.process_frame(&direct_bins);
            let _ambience_samples = ifft_ambience.process_frame(&ambience_bins);

            // Track peaks
            for bin in &separated {
                mixer.mix_bin(bin);
            }

            frame_counter += 1;
        }

        let latency = frame_start.elapsed();
        total_latency += latency;
        frame_count += 1;

        // Print status every 10 frames
        if frame_count.is_multiple_of(10) {
            let (direct_peak, amb_peak, _) = mixer.peak_levels_db();
            let k2k_stats = bin_network.k2k_stats();

            print!(
                "\r{:5} | {:>5.1}ms | {:>+7.1} dB | {:>+8.1} dB | {:>8}",
                frame_counter,
                latency.as_secs_f64() * 1000.0,
                direct_peak,
                amb_peak,
                k2k_stats.messages_delivered
            );
            io::stdout().flush().ok();

            mixer.reset_peaks();
        }

        // Simulate real-time timing
        let elapsed = frame_start.elapsed();
        if elapsed < block_duration {
            tokio::time::sleep(block_duration - elapsed).await;
        }
    }

    println!("\n\n=== Processing Summary ===");
    println!("  Total frames: {}", frame_counter);
    println!("  Total time: {:?}", start.elapsed());
    if frame_count > 0 {
        println!(
            "  Average latency: {:.2} ms",
            total_latency.as_secs_f64() * 1000.0 / frame_count as f64
        );
    }

    // Cleanup
    bin_network.stop().await?;
    println!("\nDone!");

    Ok(())
}

fn generate_audio_block(sample_rate: u32, block_size: usize, frame: u64) -> Vec<f32> {
    let t_offset = frame as f32 * block_size as f32 / sample_rate as f32;

    (0..block_size)
        .map(|i| {
            let t = t_offset + i as f32 / sample_rate as f32;

            // Frequency modulation for interesting signal
            let freq = 440.0 + 100.0 * (0.5 * t).sin();

            // Direct signal (coherent)
            let direct = 0.4 * (2.0 * std::f32::consts::PI * freq * t).sin();

            // Ambient signal (diffuse)
            let ambient = 0.1
                * ((2.0 * std::f32::consts::PI * (freq * 1.003) * t).sin()
                    + (2.0 * std::f32::consts::PI * (freq * 0.997) * t + 1.0).sin()
                    + (2.0 * std::f32::consts::PI * (freq * 2.01) * t + 2.0).sin() * 0.5);

            direct + ambient
        })
        .collect()
}

fn ctrlc_handler(running: Arc<AtomicBool>) {
    #[cfg(unix)]
    {
        tokio::spawn(async move {
            tokio::signal::ctrl_c().await.ok();
            running.store(false, Ordering::Relaxed);
            println!("\nShutting down...");
        });
    }

    #[cfg(not(unix))]
    {
        // On non-Unix systems, just let it run for the duration
        let _ = running;
    }
}
