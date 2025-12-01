//! Audio input handling for files and device streams.
//!
//! This module provides a unified interface for reading audio from:
//! - WAV files (via hound)
//! - Real-time audio devices (via cpal)

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crossbeam::channel::{Receiver, Sender};
#[cfg(feature = "device-input")]
use crossbeam::channel::bounded;
#[cfg(feature = "device-input")]
use tracing::{debug, error, info, warn};
#[cfg(not(feature = "device-input"))]
use tracing::{debug, info};

use crate::error::{AudioFftError, Result};
use crate::messages::AudioFrame;

/// Audio source trait for unified input handling.
pub trait AudioSource: Send + Sync {
    /// Get the sample rate in Hz.
    fn sample_rate(&self) -> u32;

    /// Get the number of channels.
    fn channels(&self) -> u8;

    /// Get the total number of samples (None for streams).
    fn total_samples(&self) -> Option<u64>;

    /// Read the next frame of audio.
    fn read_frame(&mut self, frame_size: usize) -> Result<Option<AudioFrame>>;

    /// Check if the source is exhausted.
    fn is_exhausted(&self) -> bool;

    /// Reset to the beginning (if supported).
    fn reset(&mut self) -> Result<()>;
}

/// Audio input from a WAV file.
pub struct FileSource {
    path: String,
    sample_rate: u32,
    channels: u8,
    samples: Vec<f32>,
    position: usize,
    frame_counter: u64,
}

impl FileSource {
    /// Open a WAV file for reading.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        info!("Opening audio file: {}", path_str);

        let reader = hound::WavReader::open(path.as_ref())
            .map_err(|e| AudioFftError::file_read(format!("{}: {}", path_str, e)))?;

        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let channels = spec.channels as u8;

        debug!(
            "File spec: {} Hz, {} channels, {} bits, {:?}",
            sample_rate, channels, spec.bits_per_sample, spec.sample_format
        );

        // Read all samples and convert to f32
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .into_samples::<f32>()
                .filter_map(|s| s.ok())
                .collect(),
            hound::SampleFormat::Int => {
                let scale = 1.0 / (1 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .into_samples::<i32>()
                    .filter_map(|s| s.ok())
                    .map(|s| s as f32 * scale)
                    .collect()
            }
        };

        info!("Loaded {} samples ({:.2} seconds)",
            samples.len(),
            samples.len() as f64 / channels as f64 / sample_rate as f64
        );

        Ok(Self {
            path: path_str,
            sample_rate,
            channels,
            samples,
            position: 0,
            frame_counter: 0,
        })
    }

    /// Get the file path.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Get the duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / self.channels as f64 / self.sample_rate as f64
    }
}

impl AudioSource for FileSource {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u8 {
        self.channels
    }

    fn total_samples(&self) -> Option<u64> {
        Some(self.samples.len() as u64 / self.channels as u64)
    }

    fn read_frame(&mut self, frame_size: usize) -> Result<Option<AudioFrame>> {
        if self.position >= self.samples.len() {
            return Ok(None);
        }

        let samples_to_read = (frame_size * self.channels as usize)
            .min(self.samples.len() - self.position);

        let frame_samples = self.samples[self.position..self.position + samples_to_read].to_vec();
        let timestamp = self.position as u64 / self.channels as u64;

        self.position += samples_to_read;
        self.frame_counter += 1;

        Ok(Some(AudioFrame::new(
            self.frame_counter,
            self.sample_rate,
            self.channels,
            frame_samples,
            timestamp,
        )))
    }

    fn is_exhausted(&self) -> bool {
        self.position >= self.samples.len()
    }

    fn reset(&mut self) -> Result<()> {
        self.position = 0;
        self.frame_counter = 0;
        Ok(())
    }
}

/// Real-time audio device stream.
pub struct DeviceStream {
    sample_rate: u32,
    channels: u8,
    receiver: Receiver<Vec<f32>>,
    buffer: Vec<f32>,
    frame_counter: Arc<AtomicU64>,
    running: Arc<AtomicBool>,
    // Keep the stream alive (only with device-input feature)
    #[cfg(feature = "device-input")]
    _stream: Option<cpal::Stream>,
}

/// Device stream configuration.
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    /// Preferred sample rate (None = use device default).
    pub sample_rate: Option<u32>,
    /// Preferred channels (None = use device default).
    pub channels: Option<u8>,
    /// Buffer size in samples.
    pub buffer_size: usize,
    /// Device name (None = default device).
    pub device_name: Option<String>,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            sample_rate: None,
            channels: None,
            buffer_size: 4096,
            device_name: None,
        }
    }
}

impl DeviceStream {
    /// Create a new device stream with the default input device.
    #[cfg(feature = "device-input")]
    pub fn new(config: DeviceConfig) -> Result<Self> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let host = cpal::default_host();

        let device = if let Some(name) = &config.device_name {
            host.input_devices()
                .map_err(|e| AudioFftError::device(format!("Failed to enumerate devices: {}", e)))?
                .find(|d| d.name().map(|n| n.contains(name)).unwrap_or(false))
                .ok_or_else(|| AudioFftError::device(format!("Device '{}' not found", name)))?
        } else {
            host.default_input_device()
                .ok_or_else(|| AudioFftError::device("No default input device"))?
        };

        let device_name = device.name().unwrap_or_else(|_| "unknown".to_string());
        info!("Using input device: {}", device_name);

        let supported_config = device
            .default_input_config()
            .map_err(|e| AudioFftError::device(format!("Failed to get device config: {}", e)))?;

        let sample_rate = config.sample_rate.unwrap_or(supported_config.sample_rate().0);
        let channels = config.channels.unwrap_or(supported_config.channels() as u8);

        debug!("Stream config: {} Hz, {} channels", sample_rate, channels);

        let (sender, receiver) = bounded(64);
        let running = Arc::new(AtomicBool::new(true));
        let frame_counter = Arc::new(AtomicU64::new(0));

        let running_clone = running.clone();
        let sender_clone = sender.clone();

        let stream_config = cpal::StreamConfig {
            channels: channels as u16,
            sample_rate: cpal::SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Fixed(config.buffer_size as u32),
        };

        let stream = device
            .build_input_stream(
                &stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if running_clone.load(Ordering::Relaxed) {
                        if sender_clone.try_send(data.to_vec()).is_err() {
                            warn!("Audio buffer overflow - dropping samples");
                        }
                    }
                },
                move |err| {
                    error!("Audio stream error: {}", err);
                },
                None,
            )
            .map_err(|e| AudioFftError::device(format!("Failed to build stream: {}", e)))?;

        stream
            .play()
            .map_err(|e| AudioFftError::device(format!("Failed to start stream: {}", e)))?;

        info!("Audio device stream started");

        Ok(Self {
            sample_rate,
            channels,
            receiver,
            buffer: Vec::with_capacity(config.buffer_size * 2),
            frame_counter,
            running,
            _stream: Some(stream),
        })
    }

    /// Create a dummy device stream (for testing without audio device).
    #[cfg(not(feature = "device-input"))]
    pub fn new(_config: DeviceConfig) -> Result<Self> {
        Err(AudioFftError::device(
            "Device input not enabled. Compile with --features device-input",
        ))
    }

    /// Create a mock stream for testing.
    #[cfg(feature = "device-input")]
    pub fn mock(sample_rate: u32, channels: u8, _sender: Sender<Vec<f32>>, receiver: Receiver<Vec<f32>>) -> Self {
        Self {
            sample_rate,
            channels,
            receiver,
            buffer: Vec::new(),
            frame_counter: Arc::new(AtomicU64::new(0)),
            running: Arc::new(AtomicBool::new(true)),
            _stream: None,
        }
    }

    /// Create a mock stream for testing.
    #[cfg(not(feature = "device-input"))]
    pub fn mock(sample_rate: u32, channels: u8, _sender: Sender<Vec<f32>>, receiver: Receiver<Vec<f32>>) -> Self {
        Self {
            sample_rate,
            channels,
            receiver,
            buffer: Vec::new(),
            frame_counter: Arc::new(AtomicU64::new(0)),
            running: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Stop the stream.
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }

    /// Check if the stream is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

impl AudioSource for DeviceStream {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn channels(&self) -> u8 {
        self.channels
    }

    fn total_samples(&self) -> Option<u64> {
        None // Stream has no defined length
    }

    fn read_frame(&mut self, frame_size: usize) -> Result<Option<AudioFrame>> {
        let required_samples = frame_size * self.channels as usize;

        // Fill buffer from receiver
        while self.buffer.len() < required_samples {
            match self.receiver.try_recv() {
                Ok(samples) => self.buffer.extend(samples),
                Err(crossbeam::channel::TryRecvError::Empty) => {
                    // Not enough data yet
                    if !self.is_running() && self.buffer.is_empty() {
                        return Ok(None);
                    }
                    // Return what we have (might be less than frame_size)
                    if !self.buffer.is_empty() {
                        break;
                    }
                    // Wait for more data
                    match self.receiver.recv_timeout(std::time::Duration::from_millis(100)) {
                        Ok(samples) => self.buffer.extend(samples),
                        Err(_) => {
                            if !self.is_running() {
                                return Ok(None);
                            }
                            continue;
                        }
                    }
                }
                Err(crossbeam::channel::TryRecvError::Disconnected) => {
                    if self.buffer.is_empty() {
                        return Ok(None);
                    }
                    break;
                }
            }
        }

        if self.buffer.is_empty() {
            return Ok(None);
        }

        let samples_to_take = required_samples.min(self.buffer.len());
        let frame_samples: Vec<f32> = self.buffer.drain(..samples_to_take).collect();

        let frame_id = self.frame_counter.fetch_add(1, Ordering::Relaxed);
        let timestamp = frame_id * frame_size as u64;

        Ok(Some(AudioFrame::new(
            frame_id,
            self.sample_rate,
            self.channels,
            frame_samples,
            timestamp,
        )))
    }

    fn is_exhausted(&self) -> bool {
        !self.is_running() && self.buffer.is_empty()
    }

    fn reset(&mut self) -> Result<()> {
        self.buffer.clear();
        self.frame_counter.store(0, Ordering::Relaxed);
        Ok(())
    }
}

impl Drop for DeviceStream {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Unified audio input that can be either file or device.
pub enum AudioInput {
    /// File-based input.
    File(FileSource),
    /// Device stream input.
    Device(DeviceStream),
}

impl AudioInput {
    /// Create input from a file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(Self::File(FileSource::open(path)?))
    }

    /// Create input from the default audio device.
    pub fn from_device(config: DeviceConfig) -> Result<Self> {
        Ok(Self::Device(DeviceStream::new(config)?))
    }

    /// Create input from raw samples (for testing).
    pub fn from_samples(samples: Vec<f32>, sample_rate: u32, channels: u8) -> Self {
        Self::File(FileSource {
            path: "<memory>".to_string(),
            sample_rate,
            channels,
            samples,
            position: 0,
            frame_counter: 0,
        })
    }
}

impl AudioSource for AudioInput {
    fn sample_rate(&self) -> u32 {
        match self {
            Self::File(f) => f.sample_rate(),
            Self::Device(d) => d.sample_rate(),
        }
    }

    fn channels(&self) -> u8 {
        match self {
            Self::File(f) => f.channels(),
            Self::Device(d) => d.channels(),
        }
    }

    fn total_samples(&self) -> Option<u64> {
        match self {
            Self::File(f) => f.total_samples(),
            Self::Device(d) => d.total_samples(),
        }
    }

    fn read_frame(&mut self, frame_size: usize) -> Result<Option<AudioFrame>> {
        match self {
            Self::File(f) => f.read_frame(frame_size),
            Self::Device(d) => d.read_frame(frame_size),
        }
    }

    fn is_exhausted(&self) -> bool {
        match self {
            Self::File(f) => f.is_exhausted(),
            Self::Device(d) => d.is_exhausted(),
        }
    }

    fn reset(&mut self) -> Result<()> {
        match self {
            Self::File(f) => f.reset(),
            Self::Device(d) => d.reset(),
        }
    }
}

/// Audio output writer for WAV files.
#[derive(Debug, Clone)]
pub struct AudioOutput {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u8,
    /// Audio samples.
    pub samples: Vec<f32>,
}

impl AudioOutput {
    /// Create a new empty audio output.
    pub fn new(sample_rate: u32, channels: u8) -> Self {
        Self {
            sample_rate,
            channels,
            samples: Vec::new(),
        }
    }

    /// Create from existing samples.
    pub fn from_samples(samples: Vec<f32>, sample_rate: u32, channels: u8) -> Self {
        Self {
            sample_rate,
            channels,
            samples,
        }
    }

    /// Append samples.
    pub fn append(&mut self, samples: &[f32]) {
        self.samples.extend_from_slice(samples);
    }

    /// Get duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / self.channels as f64 / self.sample_rate as f64
    }

    /// Write to a WAV file.
    pub fn write_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let spec = hound::WavSpec {
            channels: self.channels as u16,
            sample_rate: self.sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut writer = hound::WavWriter::create(path.as_ref(), spec)
            .map_err(|e| AudioFftError::file_write(e.to_string()))?;

        for sample in &self.samples {
            writer
                .write_sample(*sample)
                .map_err(|e| AudioFftError::file_write(e.to_string()))?;
        }

        writer
            .finalize()
            .map_err(|e| AudioFftError::file_write(e.to_string()))?;

        info!(
            "Wrote {} samples to {}",
            self.samples.len(),
            path.as_ref().display()
        );

        Ok(())
    }

    /// Normalize the audio to peak at 1.0.
    pub fn normalize(&mut self) {
        let max = self
            .samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);

        if max > 1e-6 {
            let scale = 1.0 / max;
            for sample in &mut self.samples {
                *sample *= scale;
            }
        }
    }

    /// Apply gain.
    pub fn apply_gain(&mut self, gain: f32) {
        for sample in &mut self.samples {
            *sample *= gain;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_input_from_samples() {
        let samples = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let mut input = AudioInput::from_samples(samples, 44100, 2);

        assert_eq!(input.sample_rate(), 44100);
        assert_eq!(input.channels(), 2);
        assert_eq!(input.total_samples(), Some(4)); // 8 samples / 2 channels

        let frame = input.read_frame(2).unwrap().unwrap();
        assert_eq!(frame.samples.len(), 4); // 2 samples * 2 channels
        assert!(!input.is_exhausted());

        let frame2 = input.read_frame(2).unwrap().unwrap();
        assert_eq!(frame2.samples.len(), 4);
        assert!(input.is_exhausted());
    }

    #[test]
    fn test_audio_output() {
        let mut output = AudioOutput::new(44100, 1);
        output.append(&[0.5, -0.5, 0.25, -0.25]);

        assert_eq!(output.samples.len(), 4);
        assert!((output.duration_secs() - 4.0 / 44100.0).abs() < 1e-6);

        output.normalize();
        assert!((output.samples[0] - 1.0).abs() < 1e-6);
        assert!((output.samples[1] - (-1.0)).abs() < 1e-6);
    }
}
