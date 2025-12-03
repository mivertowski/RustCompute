//! Main audio FFT processor orchestrating the full pipeline.
//!
//! This module provides the high-level API for processing audio through
//! the GPU-accelerated FFT bin actor network with direct/ambience separation.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use tracing::info;

use crate::audio_input::{AudioInput, AudioOutput, AudioSource};
use crate::bin_actor::BinNetwork;
use crate::error::{AudioFftError, Result};
use crate::fft::{FftProcessor, IfftProcessor, WindowFunction};
use crate::mixer::{FrameMixer, MixerConfig};
use crate::separation::SeparationConfig;

/// Builder for AudioFftProcessor.
#[derive(Debug, Clone)]
pub struct AudioFftProcessorBuilder {
    fft_size: usize,
    hop_size: usize,
    sample_rate: Option<u32>,
    window: WindowFunction,
    separation_config: SeparationConfig,
    mixer_config: MixerConfig,
}

impl Default for AudioFftProcessorBuilder {
    fn default() -> Self {
        Self {
            fft_size: 2048,
            hop_size: 512,
            sample_rate: None,
            window: WindowFunction::Hann,
            separation_config: SeparationConfig::default(),
            mixer_config: MixerConfig::default(),
        }
    }
}

impl AudioFftProcessorBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the FFT size (must be power of 2).
    pub fn fft_size(mut self, size: usize) -> Self {
        self.fft_size = size;
        self
    }

    /// Set the hop size (for overlap-add).
    pub fn hop_size(mut self, size: usize) -> Self {
        self.hop_size = size;
        self
    }

    /// Set the sample rate (optional, will use input's rate if not set).
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = Some(rate);
        self
    }

    /// Set the window function.
    pub fn window(mut self, window: WindowFunction) -> Self {
        self.window = window;
        self
    }

    /// Set the separation configuration.
    pub fn separation_config(mut self, config: SeparationConfig) -> Self {
        self.separation_config = config;
        self
    }

    /// Set the mixer configuration.
    pub fn mixer_config(mut self, config: MixerConfig) -> Self {
        self.mixer_config = config;
        self
    }

    /// Use music preset for separation.
    pub fn music_mode(mut self) -> Self {
        self.separation_config = SeparationConfig::music_preset();
        self
    }

    /// Use speech preset for separation.
    pub fn speech_mode(mut self) -> Self {
        self.separation_config = SeparationConfig::speech_preset();
        self
    }

    /// Build the processor.
    pub async fn build(self) -> Result<AudioFftProcessor> {
        let sample_rate = self.sample_rate.unwrap_or(44100);

        info!(
            "Building AudioFftProcessor: FFT size={}, hop={}, sample_rate={}",
            self.fft_size, self.hop_size, sample_rate
        );

        let num_bins = self.fft_size / 2 + 1;
        let bin_network = BinNetwork::new(num_bins, self.separation_config.clone()).await?;

        Ok(AudioFftProcessor {
            fft_size: self.fft_size,
            hop_size: self.hop_size,
            sample_rate,
            window: self.window,
            separation_config: self.separation_config,
            mixer_config: self.mixer_config,
            bin_network: Some(bin_network),
            frame_counter: AtomicU64::new(0),
            stats: Arc::new(RwLock::new(ProcessingStats::default())),
        })
    }
}

/// Processing statistics.
#[derive(Debug, Clone, Default)]
pub struct ProcessingStats {
    /// Total frames processed.
    pub frames_processed: u64,
    /// Total samples processed.
    pub samples_processed: u64,
    /// Total K2K messages exchanged.
    pub k2k_messages: u64,
    /// Average processing time per frame (microseconds).
    pub avg_frame_time_us: f64,
    /// Peak direct signal level.
    pub peak_direct: f32,
    /// Peak ambience signal level.
    pub peak_ambience: f32,
}

/// Output from processing.
#[derive(Debug)]
pub struct ProcessingOutput {
    /// Direct signal output.
    pub direct: AudioOutput,
    /// Ambience signal output.
    pub ambience: AudioOutput,
    /// Mixed output (based on dry/wet settings).
    pub mixed: AudioOutput,
    /// Processing statistics.
    pub stats: ProcessingStats,
}

impl ProcessingOutput {
    /// Create a new processing output.
    pub fn new(sample_rate: u32, channels: u8) -> Self {
        Self {
            direct: AudioOutput::new(sample_rate, channels),
            ambience: AudioOutput::new(sample_rate, channels),
            mixed: AudioOutput::new(sample_rate, channels),
            stats: ProcessingStats::default(),
        }
    }
}

/// Main audio FFT processor with GPU bin actors.
pub struct AudioFftProcessor {
    /// FFT size.
    fft_size: usize,
    /// Hop size.
    hop_size: usize,
    /// Sample rate (reserved for resampling support).
    #[allow(dead_code)]
    sample_rate: u32,
    /// Window function.
    window: WindowFunction,
    /// Separation configuration.
    separation_config: SeparationConfig,
    /// Mixer configuration.
    mixer_config: MixerConfig,
    /// Bin actor network.
    bin_network: Option<BinNetwork>,
    /// Frame counter.
    frame_counter: AtomicU64,
    /// Processing statistics.
    stats: Arc<RwLock<ProcessingStats>>,
}

impl AudioFftProcessor {
    /// Create a new builder.
    pub fn builder() -> AudioFftProcessorBuilder {
        AudioFftProcessorBuilder::new()
    }

    /// Get the FFT size.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Get the hop size.
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Get the number of frequency bins.
    pub fn num_bins(&self) -> usize {
        self.fft_size / 2 + 1
    }

    /// Get processing statistics.
    pub fn stats(&self) -> ProcessingStats {
        self.stats.read().clone()
    }

    /// Process an audio input and return separated outputs.
    pub async fn process(&mut self, mut input: AudioInput) -> Result<ProcessingOutput> {
        // Use input's sample rate if we don't have one
        let sample_rate = input.sample_rate();
        let channels = input.channels();

        info!(
            "Processing audio: {} Hz, {} channels",
            sample_rate, channels
        );

        let mut output = ProcessingOutput::new(sample_rate, channels);
        let mut fft_processor =
            FftProcessor::with_window(self.fft_size, self.hop_size, sample_rate, self.window)?;
        let mut ifft_processor =
            IfftProcessor::with_window(self.fft_size, self.hop_size, self.window)?;

        let mut frame_mixer = FrameMixer::new(self.mixer_config.clone());

        let bin_network = self
            .bin_network
            .as_mut()
            .ok_or_else(|| AudioFftError::kernel("Bin network not initialized"))?;

        // Process mono for now (extract first channel if stereo)
        let mut total_frames = 0u64;
        let start_time = std::time::Instant::now();

        while let Some(audio_frame) = input.read_frame(self.hop_size * 4)? {
            // Get mono samples
            let samples = if channels > 1 {
                audio_frame.channel_samples(0)
            } else {
                audio_frame.samples.clone()
            };

            // Process through FFT
            for fft_frame in fft_processor.process_all(&samples) {
                let frame_id = self.frame_counter.fetch_add(1, Ordering::Relaxed);

                // Send to bin network and get separated results
                let separated = bin_network
                    .process_frame(frame_id, &fft_frame, sample_rate, self.fft_size)
                    .await?;

                // Mix the separated bins
                let mixed = frame_mixer.process(&separated);

                // IFFT back to time domain
                let direct_samples = ifft_processor.process_frame(&mixed.direct_bins);
                let ambience_samples = ifft_processor.process_frame(&mixed.ambience_bins);
                let mixed_samples = ifft_processor.process_frame(&mixed.bins);

                // Append to outputs
                output.direct.append(&direct_samples);
                output.ambience.append(&ambience_samples);
                output.mixed.append(&mixed_samples);

                total_frames += 1;
            }
        }

        // Flush remaining samples
        if let Some(last_frame) = fft_processor.flush() {
            let frame_id = self.frame_counter.fetch_add(1, Ordering::Relaxed);
            let separated = bin_network
                .process_frame(frame_id, &last_frame, sample_rate, self.fft_size)
                .await?;
            let mixed = frame_mixer.process(&separated);

            output
                .direct
                .append(&ifft_processor.process_frame(&mixed.direct_bins));
            output
                .ambience
                .append(&ifft_processor.process_frame(&mixed.ambience_bins));
            output
                .mixed
                .append(&ifft_processor.process_frame(&mixed.bins));
        }

        // Flush IFFT
        output.direct.append(&ifft_processor.flush());
        output.ambience.append(&ifft_processor.flush());
        output.mixed.append(&ifft_processor.flush());

        let elapsed = start_time.elapsed();
        let avg_time = if total_frames > 0 {
            elapsed.as_micros() as f64 / total_frames as f64
        } else {
            0.0
        };

        // Update stats
        let k2k_stats = bin_network.k2k_stats();
        {
            let mut stats = self.stats.write();
            stats.frames_processed = total_frames;
            stats.samples_processed = output.mixed.samples.len() as u64;
            stats.k2k_messages = k2k_stats.messages_delivered;
            stats.avg_frame_time_us = avg_time;

            let (direct_peak, amb_peak, _) = frame_mixer.mixer().peak_levels();
            stats.peak_direct = direct_peak;
            stats.peak_ambience = amb_peak;
        }

        output.stats = self.stats();

        info!(
            "Processed {} frames in {:?} ({:.1} us/frame)",
            total_frames, elapsed, avg_time
        );

        Ok(output)
    }

    /// Process with streaming output (for real-time use).
    pub fn process_streaming(&mut self, input: AudioInput) -> Result<StreamingProcessor> {
        let sample_rate = input.sample_rate();

        Ok(StreamingProcessor {
            input: Some(input),
            fft: FftProcessor::with_window(self.fft_size, self.hop_size, sample_rate, self.window)?,
            ifft_direct: IfftProcessor::with_window(self.fft_size, self.hop_size, self.window)?,
            ifft_ambience: IfftProcessor::with_window(self.fft_size, self.hop_size, self.window)?,
            ifft_mixed: IfftProcessor::with_window(self.fft_size, self.hop_size, self.window)?,
            sample_rate,
            fft_size: self.fft_size,
            hop_size: self.hop_size,
            mixer: FrameMixer::new(self.mixer_config.clone()),
            frame_counter: 0,
        })
    }

    /// Update the dry/wet mix.
    pub fn set_dry_wet(&mut self, dry_wet: f32) {
        self.mixer_config.dry_wet = dry_wet.clamp(0.0, 1.0);
    }

    /// Update the output gain in dB.
    pub fn set_gain_db(&mut self, gain_db: f32) {
        self.mixer_config.output_gain = 10.0_f32.powf(gain_db / 20.0);
    }

    /// Update separation configuration.
    pub fn set_separation_config(&mut self, config: SeparationConfig) {
        self.separation_config = config;
    }

    /// Shutdown the processor and release resources.
    pub async fn shutdown(&mut self) -> Result<()> {
        if let Some(mut network) = self.bin_network.take() {
            network.stop().await?;
        }
        Ok(())
    }
}

/// Streaming processor for frame-by-frame processing.
pub struct StreamingProcessor {
    input: Option<AudioInput>,
    fft: FftProcessor,
    ifft_direct: IfftProcessor,
    ifft_ambience: IfftProcessor,
    ifft_mixed: IfftProcessor,
    sample_rate: u32,
    fft_size: usize,
    hop_size: usize,
    mixer: FrameMixer,
    frame_counter: u64,
}

impl StreamingProcessor {
    /// Set the dry/wet mix.
    pub fn set_dry_wet(&mut self, dry_wet: f32) {
        self.mixer.set_dry_wet(dry_wet);
    }

    /// Set the output gain in dB.
    pub fn set_gain_db(&mut self, gain_db: f32) {
        self.mixer.set_gain_db(gain_db);
    }

    /// Process the next chunk of audio.
    /// Returns (direct_samples, ambience_samples, mixed_samples) or None if input exhausted.
    pub async fn next(
        &mut self,
        bin_network: &mut BinNetwork,
    ) -> Result<Option<(Vec<f32>, Vec<f32>, Vec<f32>)>> {
        let input = match &mut self.input {
            Some(input) => input,
            None => return Ok(None),
        };

        if input.is_exhausted() {
            return Ok(None);
        }

        // Read audio frame
        let audio_frame = match input.read_frame(self.hop_size * 2)? {
            Some(frame) => frame,
            None => return Ok(None),
        };

        // Get mono samples
        let samples = if audio_frame.channels > 1 {
            audio_frame.channel_samples(0)
        } else {
            audio_frame.samples.clone()
        };

        let mut direct_out = Vec::new();
        let mut ambience_out = Vec::new();
        let mut mixed_out = Vec::new();

        // Process through FFT
        for fft_frame in self.fft.process_all(&samples) {
            let frame_id = self.frame_counter;
            self.frame_counter += 1;

            // Process through bin network
            let separated = bin_network
                .process_frame(frame_id, &fft_frame, self.sample_rate, self.fft_size)
                .await?;

            // Mix
            let mixed = self.mixer.process(&separated);

            // IFFT
            direct_out.extend(self.ifft_direct.process_frame(&mixed.direct_bins));
            ambience_out.extend(self.ifft_ambience.process_frame(&mixed.ambience_bins));
            mixed_out.extend(self.ifft_mixed.process_frame(&mixed.bins));
        }

        Ok(Some((direct_out, ambience_out, mixed_out)))
    }

    /// Flush remaining samples.
    pub async fn flush(
        &mut self,
        bin_network: &mut BinNetwork,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let mut direct_out = Vec::new();
        let mut ambience_out = Vec::new();
        let mut mixed_out = Vec::new();

        // Flush FFT
        if let Some(last_frame) = self.fft.flush() {
            let frame_id = self.frame_counter;
            self.frame_counter += 1;

            let separated = bin_network
                .process_frame(frame_id, &last_frame, self.sample_rate, self.fft_size)
                .await?;

            let mixed = self.mixer.process(&separated);

            direct_out.extend(self.ifft_direct.process_frame(&mixed.direct_bins));
            ambience_out.extend(self.ifft_ambience.process_frame(&mixed.ambience_bins));
            mixed_out.extend(self.ifft_mixed.process_frame(&mixed.bins));
        }

        // Flush IFFTs
        direct_out.extend(self.ifft_direct.flush());
        ambience_out.extend(self.ifft_ambience.flush());
        mixed_out.extend(self.ifft_mixed.flush());

        Ok((direct_out, ambience_out, mixed_out))
    }
}

/// Simplified API for quick processing.
pub async fn process_file(
    input_path: &str,
    output_dir: &str,
    dry_wet: f32,
    gain_db: f32,
) -> Result<ProcessingStats> {
    let input = AudioInput::from_file(input_path)?;

    let mut processor = AudioFftProcessor::builder()
        .fft_size(2048)
        .hop_size(512)
        .mixer_config(
            MixerConfig::new()
                .with_dry_wet(dry_wet)
                .with_output_gain(10.0_f32.powf(gain_db / 20.0)),
        )
        .build()
        .await?;

    let output = processor.process(input).await?;

    // Write outputs
    let base_name = std::path::Path::new(input_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    output
        .direct
        .write_to_file(format!("{}/{}_direct.wav", output_dir, base_name))?;
    output
        .ambience
        .write_to_file(format!("{}/{}_ambience.wav", output_dir, base_name))?;
    output
        .mixed
        .write_to_file(format!("{}/{}_mixed.wav", output_dir, base_name))?;

    processor.shutdown().await?;

    Ok(output.stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_processor_builder() {
        let processor = AudioFftProcessor::builder()
            .fft_size(1024)
            .hop_size(256)
            .sample_rate(44100)
            .music_mode()
            .build()
            .await
            .unwrap();

        assert_eq!(processor.fft_size(), 1024);
        assert_eq!(processor.hop_size(), 256);
        assert_eq!(processor.num_bins(), 513);
    }

    #[tokio::test]
    async fn test_processor_with_synthetic_input() {
        let mut processor = AudioFftProcessor::builder()
            .fft_size(512)
            .hop_size(128)
            .sample_rate(44100)
            .build()
            .await
            .unwrap();

        // Create synthetic input (sine wave)
        let duration = 0.5;
        let sample_rate = 44100;
        let samples: Vec<f32> = (0..(sample_rate as f32 * duration) as usize)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5
            })
            .collect();

        let input = AudioInput::from_samples(samples.clone(), sample_rate, 1);
        let output = processor.process(input).await.unwrap();

        // Verify output lengths are reasonable
        assert!(!output.direct.samples.is_empty());
        assert!(!output.ambience.samples.is_empty());
        assert!(!output.mixed.samples.is_empty());

        // All outputs should have similar length
        let len_diff =
            (output.direct.samples.len() as i64 - output.ambience.samples.len() as i64).abs();
        assert!(len_diff < 1000);

        processor.shutdown().await.unwrap();
    }
}
