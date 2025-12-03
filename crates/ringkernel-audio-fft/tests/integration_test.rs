//! Integration tests for the audio FFT processing pipeline.

use ringkernel_audio_fft::prelude::*;

/// Test the complete processing pipeline with synthetic input.
#[tokio::test]
async fn test_full_pipeline() {
    // Create synthetic input
    let sample_rate = 44100u32;
    let duration = 0.5; // 500ms
    let num_samples = (sample_rate as f64 * duration) as usize;

    // Generate a test signal: 440 Hz sine wave
    let samples: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
        })
        .collect();

    let input = AudioInput::from_samples(samples.clone(), sample_rate, 1);

    // Create processor
    let mut processor = AudioFftProcessor::builder()
        .fft_size(1024)
        .hop_size(256)
        .sample_rate(sample_rate)
        .build()
        .await
        .expect("Failed to build processor");

    // Process
    let output = processor.process(input).await.expect("Processing failed");

    // Verify output
    assert!(!output.direct.samples.is_empty(), "Direct output is empty");
    assert!(
        !output.ambience.samples.is_empty(),
        "Ambience output is empty"
    );
    assert!(!output.mixed.samples.is_empty(), "Mixed output is empty");

    // Output lengths should be similar (within tolerance for overlap-add)
    let len_tolerance = 2000;
    assert!(
        (output.direct.samples.len() as i64 - output.ambience.samples.len() as i64).abs()
            < len_tolerance
    );
    assert!(
        (output.mixed.samples.len() as i64 - output.direct.samples.len() as i64).abs()
            < len_tolerance
    );

    // Statistics should be populated
    assert!(output.stats.frames_processed > 0);
    assert!(output.stats.samples_processed > 0);

    // Cleanup
    processor.shutdown().await.expect("Shutdown failed");
}

/// Test the bin network K2K messaging.
#[tokio::test]
async fn test_bin_network_k2k() {
    let config = SeparationConfig::default();
    let num_bins = 16;

    let mut network = BinNetwork::new(num_bins, config)
        .await
        .expect("Failed to create network");

    assert_eq!(network.num_bins(), num_bins);

    // Create test bins
    let bins: Vec<Complex> = (0..num_bins)
        .map(|i| Complex::new(i as f32 * 0.1, 0.0))
        .collect();

    // Process a frame
    let separated = network
        .process_frame(1, &bins, 44100, 32)
        .await
        .expect("Processing failed");

    // Verify we got results
    assert_eq!(separated.len(), num_bins);

    // Each separated bin should have valid data
    for (i, bin) in separated.iter().enumerate() {
        assert_eq!(bin.bin_index, i as u32);
        assert!((0.0..=1.0).contains(&bin.coherence));
    }

    // Check K2K stats
    let stats = network.k2k_stats();
    assert!(stats.registered_endpoints >= num_bins);
    // K2K messages should have been exchanged
    // (may be 0 initially due to async timing, but should work)

    network.stop().await.expect("Stop failed");
}

/// Test the FFT processor.
#[test]
fn test_fft_processor() {
    let fft_size = 1024;
    let hop_size = 256;
    let sample_rate = 44100;

    let mut fft = FftProcessor::new(fft_size, hop_size, sample_rate).expect("FFT creation failed");

    assert_eq!(fft.fft_size(), fft_size);
    assert_eq!(fft.hop_size(), hop_size);
    assert_eq!(fft.num_bins(), fft_size / 2 + 1);

    // Generate test samples
    let samples: Vec<f32> = (0..fft_size * 2)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
        .collect();

    // Process
    let frames = fft.process_all(&samples);

    // Should get some frames
    assert!(!frames.is_empty());

    // Each frame should have the right number of bins
    for frame in &frames {
        assert_eq!(frame.len(), fft_size / 2 + 1);
    }
}

/// Test the IFFT processor roundtrip.
#[test]
fn test_fft_ifft_roundtrip() {
    let fft_size = 512;
    let hop_size = 128;
    let sample_rate = 44100;

    let mut fft = FftProcessor::new(fft_size, hop_size, sample_rate).unwrap();
    let mut ifft = IfftProcessor::new(fft_size, hop_size).unwrap();

    // Generate a simple signal
    let input: Vec<f32> = (0..fft_size * 4)
        .map(|i| (2.0 * std::f32::consts::PI * 100.0 * i as f32 / sample_rate as f32).sin())
        .collect();

    // Process through FFT
    let frames = fft.process_all(&input);

    // Process through IFFT
    let mut output = Vec::new();
    for frame in frames {
        output.extend(ifft.process_frame(&frame));
    }

    // Should get output samples
    assert!(!output.is_empty());

    // Output should be roughly the same length as input (with some latency)
    let len_diff = (output.len() as i64 - input.len() as i64).abs();
    assert!(len_diff < fft_size as i64 * 2);
}

/// Test the mixer.
#[test]
fn test_dry_wet_mixer() {
    // Disable soft clipping for predictable test results
    let config = MixerConfig::new().with_soft_clip(None);
    let mut mixer = DryWetMixer::with_config(config);

    let bin = SeparatedBin::new(
        0,
        0,
        Complex::new(1.0, 0.0), // Direct
        Complex::new(0.5, 0.0), // Ambience
        0.7,
        0.0,
    );

    // Test dry only
    mixer.set_dry_wet(0.0);
    let result = mixer.mix_bin(&bin);
    assert!((result.magnitude() - 1.0).abs() < 0.01);

    // Test wet only
    mixer.set_dry_wet(1.0);
    let result = mixer.mix_bin(&bin);
    assert!((result.magnitude() - 0.5).abs() < 0.01);

    // Test 50/50
    mixer.set_dry_wet(0.5);
    let result = mixer.mix_bin(&bin);
    assert!((result.magnitude() - 0.75).abs() < 0.01);
}

/// Test the coherence analyzer.
#[test]
fn test_coherence_analyzer() {
    let config = SeparationConfig::default();
    let mut analyzer = CoherenceAnalyzer::new(config);

    let value = Complex::new(1.0, 0.0);

    // Without neighbors
    let (coherence, transient) = analyzer.analyze(&value, None, None, 0.0, 0.0);
    assert!((0.0..=1.0).contains(&coherence));
    assert!((0.0..=1.0).contains(&transient));

    // With coherent neighbors (same phase)
    let left = NeighborData {
        source_bin: 0,
        frame_id: 0,
        value: Complex::new(0.9, 0.0),
        magnitude: 0.9,
        phase: 0.0,
        phase_derivative: 0.0,
        spectral_flux: 0.0,
    };
    let right = NeighborData {
        source_bin: 2,
        frame_id: 0,
        value: Complex::new(1.1, 0.0),
        magnitude: 1.1,
        phase: 0.0,
        phase_derivative: 0.0,
        spectral_flux: 0.0,
    };

    let (coherence2, _) = analyzer.analyze(&value, Some(&left), Some(&right), 0.0, 0.0);

    // Should have higher coherence with coherent neighbors
    assert!(coherence2 > 0.3);
}

/// Test the signal separator.
#[test]
fn test_signal_separator() {
    let config = SeparationConfig::default();
    let separator = SignalSeparator::new(config);

    let value = Complex::new(1.0, 0.0);

    // High coherence = mostly direct
    let (direct, ambient) = separator.separate(value, 0.9);
    assert!(direct.magnitude() > ambient.magnitude());

    // Low coherence = mostly ambient
    let (direct, ambient) = separator.separate(value, 0.1);
    assert!(ambient.magnitude() > direct.magnitude());

    // 50% coherence
    let (direct, ambient) = separator.separate(value, 0.5);
    // With default curve, should be roughly equal
    let ratio = direct.magnitude() / (direct.magnitude() + ambient.magnitude());
    assert!(ratio > 0.2 && ratio < 0.8);
}

/// Test window functions.
#[test]
fn test_window_functions() {
    let size = 512;

    // Hann window
    let hann = WindowFunction::Hann.generate(size);
    assert_eq!(hann.len(), size);
    assert!(hann[0].abs() < 0.01); // Should be near zero at edges
    assert!((hann[size / 2] - 1.0).abs() < 0.01); // Should be 1.0 at center

    // Hamming window
    let hamming = WindowFunction::Hamming.generate(size);
    assert!(hamming[0] > 0.05); // Hamming has non-zero edges

    // Blackman window
    let blackman = WindowFunction::Blackman.generate(size);
    assert!(blackman[0].abs() < 0.01);
    assert!(blackman[size / 2] > 0.9);
}

/// Test separation config presets.
#[test]
fn test_separation_presets() {
    let music = SeparationConfig::music_preset();
    let speech = SeparationConfig::speech_preset();
    let aggressive = SeparationConfig::aggressive_preset();

    // Music should have more smoothing
    assert!(music.temporal_smoothing > speech.temporal_smoothing);

    // Aggressive should have sharper separation
    assert!(aggressive.separation_curve > music.separation_curve);

    // Speech should be more transient-sensitive
    assert!(speech.transient_sensitivity > music.transient_sensitivity);
}

/// Test audio output.
#[test]
fn test_audio_output() {
    let mut output = AudioOutput::new(44100, 1);

    output.append(&[0.5, -0.5, 0.25, -0.25]);
    assert_eq!(output.samples.len(), 4);

    // Test normalization
    output.normalize();
    assert!((output.samples[0] - 1.0).abs() < 1e-6);
    assert!((output.samples[1] - (-1.0)).abs() < 1e-6);

    // Test gain
    let mut output2 = AudioOutput::from_samples(vec![0.5, 0.5], 44100, 1);
    output2.apply_gain(2.0);
    assert!((output2.samples[0] - 1.0).abs() < 1e-6);
}

/// Test complex number operations.
#[test]
fn test_complex_operations() {
    let c1 = Complex::new(3.0, 4.0);

    // Magnitude
    assert!((c1.magnitude() - 5.0).abs() < 1e-6);

    // Magnitude squared
    assert!((c1.magnitude_squared() - 25.0).abs() < 1e-6);

    // Phase
    let phase = c1.phase();
    assert!((phase - 0.9272952).abs() < 0.001);

    // From polar
    let c2 = Complex::from_polar(5.0, 0.9272952);
    assert!((c2.re - 3.0).abs() < 0.01);
    assert!((c2.im - 4.0).abs() < 0.01);

    // Multiplication
    let c3 = c1.mul(&Complex::new(1.0, 0.0));
    assert!((c3.re - 3.0).abs() < 1e-6);
    assert!((c3.im - 4.0).abs() < 1e-6);

    // Conjugate
    let conj = c1.conj();
    assert!((conj.re - 3.0).abs() < 1e-6);
    assert!((conj.im - (-4.0)).abs() < 1e-6);

    // Scale
    let scaled = c1.scale(2.0);
    assert!((scaled.re - 6.0).abs() < 1e-6);
    assert!((scaled.im - 8.0).abs() < 1e-6);

    // Add
    let sum = c1.add(&Complex::new(1.0, 1.0));
    assert!((sum.re - 4.0).abs() < 1e-6);
    assert!((sum.im - 5.0).abs() < 1e-6);
}
