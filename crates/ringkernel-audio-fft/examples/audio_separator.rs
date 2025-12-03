//! Audio Separator Example
//!
//! This example demonstrates how to use the RingKernel Audio FFT processor
//! to separate direct sound from room ambience in an audio file.
//!
//! Usage:
//!   cargo run --example audio_separator -- [input.wav] [--dry-wet 0.5] [--gain 0]
//!
//! If no input file is provided, a synthetic test signal is used.

use std::env;
use std::time::Instant;

use ringkernel_audio_fft::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("ringkernel_audio_fft=info".parse().unwrap()),
        )
        .init();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let (input_path, dry_wet, gain_db) = parse_args(&args);

    println!("=== RingKernel Audio FFT Separator ===\n");

    // Create or load input
    let (input, sample_rate, duration) = if let Some(path) = &input_path {
        println!("Loading audio file: {}", path);
        let input = AudioInput::from_file(path)?;
        let sr = input.sample_rate();
        let dur = input.total_samples().unwrap_or(0) as f64 / sr as f64;
        (input, sr, dur)
    } else {
        println!("No input file provided, using synthetic test signal");
        let (input, sr, dur) = create_test_signal();
        (input, sr, dur)
    };

    println!("  Sample rate: {} Hz", sample_rate);
    println!("  Duration: {:.2} seconds", duration);
    println!();

    // Create the processor
    println!("Initializing processor...");
    let start = Instant::now();

    let mut processor = AudioFftProcessor::builder()
        .fft_size(2048)
        .hop_size(512)
        .sample_rate(sample_rate)
        .music_mode() // Use music preset for separation
        .mixer_config(
            MixerConfig::new()
                .with_dry_wet(dry_wet)
                .with_output_gain(10.0_f32.powf(gain_db / 20.0)),
        )
        .build()
        .await?;

    println!("  Processor initialized in {:?}", start.elapsed());
    println!("  FFT size: {}", processor.fft_size());
    println!("  Frequency bins: {}", processor.num_bins());
    println!();

    // Process the audio
    println!("Processing audio...");
    let start = Instant::now();

    let output = processor.process(input).await?;

    let elapsed = start.elapsed();
    println!("  Processing completed in {:?}", elapsed);
    println!();

    // Print statistics
    println!("=== Processing Statistics ===");
    println!("  Frames processed: {}", output.stats.frames_processed);
    println!("  Samples processed: {}", output.stats.samples_processed);
    println!("  K2K messages exchanged: {}", output.stats.k2k_messages);
    println!(
        "  Average frame time: {:.1} us",
        output.stats.avg_frame_time_us
    );
    println!(
        "  Peak direct level: {:.2} ({:.1} dB)",
        output.stats.peak_direct,
        if output.stats.peak_direct > 1e-10 {
            20.0 * output.stats.peak_direct.log10()
        } else {
            -200.0
        }
    );
    println!(
        "  Peak ambience level: {:.2} ({:.1} dB)",
        output.stats.peak_ambience,
        if output.stats.peak_ambience > 1e-10 {
            20.0 * output.stats.peak_ambience.log10()
        } else {
            -200.0
        }
    );
    println!();

    // Print output info
    println!("=== Output ===");
    println!(
        "  Direct signal: {:.2}s ({} samples)",
        output.direct.duration_secs(),
        output.direct.samples.len()
    );
    println!(
        "  Ambience signal: {:.2}s ({} samples)",
        output.ambience.duration_secs(),
        output.ambience.samples.len()
    );
    println!(
        "  Mixed signal: {:.2}s ({} samples)",
        output.mixed.duration_secs(),
        output.mixed.samples.len()
    );
    println!();

    // Write output files
    if input_path.is_some() {
        let base_name = input_path
            .as_ref()
            .and_then(|p| std::path::Path::new(p).file_stem())
            .and_then(|s| s.to_str())
            .unwrap_or("output");

        println!("Writing output files...");
        output
            .direct
            .write_to_file(format!("{}_direct.wav", base_name))?;
        output
            .ambience
            .write_to_file(format!("{}_ambience.wav", base_name))?;
        output
            .mixed
            .write_to_file(format!("{}_mixed.wav", base_name))?;
        println!("  Written: {}_direct.wav", base_name);
        println!("  Written: {}_ambience.wav", base_name);
        println!("  Written: {}_mixed.wav", base_name);
    } else {
        println!("Writing test output files...");
        output.direct.write_to_file("test_direct.wav")?;
        output.ambience.write_to_file("test_ambience.wav")?;
        output.mixed.write_to_file("test_mixed.wav")?;
        println!("  Written: test_direct.wav");
        println!("  Written: test_ambience.wav");
        println!("  Written: test_mixed.wav");
    }

    // Cleanup
    processor.shutdown().await?;

    println!("\nDone!");
    Ok(())
}

fn parse_args(args: &[String]) -> (Option<String>, f32, f32) {
    let mut input_path = None;
    let mut dry_wet = 0.5;
    let mut gain_db = 0.0;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dry-wet" if i + 1 < args.len() => {
                dry_wet = args[i + 1].parse().unwrap_or(0.5);
                i += 2;
            }
            "--gain" if i + 1 < args.len() => {
                gain_db = args[i + 1].parse().unwrap_or(0.0);
                i += 2;
            }
            "--help" | "-h" => {
                println!("Usage: audio_separator [input.wav] [options]");
                println!();
                println!("Options:");
                println!("  --dry-wet <0.0-1.0>  Dry/wet mix (0=direct, 1=ambience)");
                println!("  --gain <dB>          Output gain in dB");
                println!("  --help               Show this help");
                std::process::exit(0);
            }
            path if !path.starts_with('-') => {
                input_path = Some(path.to_string());
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    println!("Settings:");
    println!("  Dry/wet mix: {:.2}", dry_wet);
    println!("  Output gain: {:.1} dB", gain_db);

    (input_path, dry_wet, gain_db)
}

/// Create a synthetic test signal with both "direct" and "ambient" characteristics.
fn create_test_signal() -> (AudioInput, u32, f64) {
    let sample_rate = 44100;
    let duration = 2.0; // 2 seconds
    let num_samples = (sample_rate as f64 * duration) as usize;

    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;

        // "Direct" component: clean sine waves with harmonics
        let direct = 0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            + 0.15 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
            + 0.1 * (2.0 * std::f32::consts::PI * 1320.0 * t).sin();

        // "Ambient" component: noise-like with varying frequency content
        // Simulated reverb tail using multiple delayed/detuned components
        let ambient = 0.05
            * ((2.0 * std::f32::consts::PI * 441.3 * t + 0.1).sin()
                + (2.0 * std::f32::consts::PI * 439.7 * t + 0.3).sin()
                + (2.0 * std::f32::consts::PI * 442.8 * t + 0.5).sin()
                + (2.0 * std::f32::consts::PI * 878.5 * t + 0.7).sin()
                + (2.0 * std::f32::consts::PI * 881.2 * t + 0.9).sin());

        // Add some envelope to simulate attack/decay
        let envelope = if t < 0.1 {
            t / 0.1
        } else if t > duration as f32 - 0.2 {
            (duration as f32 - t) / 0.2
        } else {
            1.0
        };

        samples.push((direct + ambient) * envelope);
    }

    let input = AudioInput::from_samples(samples, sample_rate, 1);
    (input, sample_rate, duration)
}
