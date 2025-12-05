//! Audio system for 3D acoustic simulation.
//!
//! Provides:
//! - Audio sources with 3D positioning
//! - Binaural microphone capture
//! - Audio file I/O
//! - Real-time audio streaming

pub mod binaural;
pub mod source;

pub use binaural::{BinauralMicrophone, BinauralProcessor, DelayLine, VirtualHead};
pub use source::{AudioSource, SourceManager, SourceType};

use crate::simulation::physics::Position3D;
use std::path::Path;

/// Load audio samples from a WAV file.
pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, u32), WavError> {
    let reader = hound::WavReader::open(path).map_err(|e| WavError::ReadError(e.to_string()))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    // Convert samples to f32 normalized to -1.0 to 1.0
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(Result::ok)
            .collect(),
        hound::SampleFormat::Int => {
            let max_val = (1i32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    // Convert stereo to mono if needed
    let mono_samples = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|chunk| (chunk[0] + chunk.get(1).copied().unwrap_or(0.0)) / 2.0)
            .collect()
    } else {
        samples
    };

    Ok((mono_samples, sample_rate))
}

/// Save audio samples to a WAV file.
pub fn save_wav<P: AsRef<Path>>(
    path: P,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<(), WavError> {
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer =
        hound::WavWriter::create(path, spec).map_err(|e| WavError::WriteError(e.to_string()))?;

    for &sample in samples {
        writer
            .write_sample(sample)
            .map_err(|e| WavError::WriteError(e.to_string()))?;
    }

    writer
        .finalize()
        .map_err(|e| WavError::WriteError(e.to_string()))?;

    Ok(())
}

/// Save stereo audio to a WAV file.
pub fn save_stereo_wav<P: AsRef<Path>>(
    path: P,
    left: &[f32],
    right: &[f32],
    sample_rate: u32,
) -> Result<(), WavError> {
    let mut interleaved = Vec::with_capacity(left.len() * 2);
    for (l, r) in left.iter().zip(right.iter()) {
        interleaved.push(*l);
        interleaved.push(*r);
    }
    save_wav(path, &interleaved, sample_rate, 2)
}

/// Error type for WAV operations.
#[derive(Debug, Clone)]
pub enum WavError {
    ReadError(String),
    WriteError(String),
}

impl std::fmt::Display for WavError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WavError::ReadError(msg) => write!(f, "WAV read error: {}", msg),
            WavError::WriteError(msg) => write!(f, "WAV write error: {}", msg),
        }
    }
}

impl std::error::Error for WavError {}

/// Audio configuration for the simulation.
#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// Output sample rate
    pub sample_rate: u32,
    /// Buffer size in samples
    pub buffer_size: usize,
    /// Enable real-time audio output
    pub enable_output: bool,
    /// Enable recording
    pub enable_recording: bool,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            buffer_size: 1024,
            enable_output: true,
            enable_recording: false,
        }
    }
}

/// Combined audio system manager.
pub struct AudioSystem {
    /// Audio configuration
    pub config: AudioConfig,
    /// Source manager
    pub sources: SourceManager,
    /// Binaural microphone
    pub microphone: Option<BinauralMicrophone>,
    /// Recording buffer (left channel)
    recording_left: Vec<f32>,
    /// Recording buffer (right channel)
    recording_right: Vec<f32>,
    /// Is recording active
    is_recording: bool,
}

impl AudioSystem {
    /// Create a new audio system.
    pub fn new(config: AudioConfig) -> Self {
        Self {
            config,
            sources: SourceManager::new(),
            microphone: None,
            recording_left: Vec::new(),
            recording_right: Vec::new(),
            is_recording: false,
        }
    }

    /// Initialize the binaural microphone.
    pub fn init_microphone(&mut self, head: VirtualHead, simulation_dt: f32) {
        self.microphone = Some(BinauralMicrophone::new(
            head,
            self.config.sample_rate,
            simulation_dt,
        ));
    }

    /// Add an audio source.
    pub fn add_source(&mut self, source: AudioSource) -> u32 {
        self.sources.add(source)
    }

    /// Create and add an impulse source.
    pub fn add_impulse(&mut self, position: Position3D, amplitude: f32) -> u32 {
        self.sources
            .add(AudioSource::impulse(0, position, amplitude))
    }

    /// Create and add a tone source.
    pub fn add_tone(&mut self, position: Position3D, frequency: f32, amplitude: f32) -> u32 {
        self.sources
            .add(AudioSource::tone(0, position, frequency, amplitude))
    }

    /// Start recording.
    pub fn start_recording(&mut self) {
        self.recording_left.clear();
        self.recording_right.clear();
        self.is_recording = true;
    }

    /// Stop recording and return the recorded audio.
    pub fn stop_recording(&mut self) -> (Vec<f32>, Vec<f32>) {
        self.is_recording = false;
        (
            std::mem::take(&mut self.recording_left),
            std::mem::take(&mut self.recording_right),
        )
    }

    /// Process recording (call after microphone capture).
    pub fn process_recording(&mut self) {
        if self.is_recording {
            if let Some(mic) = &mut self.microphone {
                let (left, right) = mic.get_samples(self.config.buffer_size);
                self.recording_left.extend(left);
                self.recording_right.extend(right);
            }
        }
    }

    /// Get the recording sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Reset all audio state.
    pub fn reset(&mut self) {
        self.sources.reset_all();
        if let Some(mic) = &mut self.microphone {
            mic.clear();
        }
        self.recording_left.clear();
        self.recording_right.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_system_creation() {
        let config = AudioConfig::default();
        let system = AudioSystem::new(config);

        assert!(system.sources.is_empty());
        assert!(system.microphone.is_none());
    }

    #[test]
    fn test_add_sources() {
        let mut system = AudioSystem::new(AudioConfig::default());

        let id1 = system.add_impulse(Position3D::origin(), 1.0);
        let id2 = system.add_tone(Position3D::new(1.0, 0.0, 0.0), 440.0, 0.5);

        assert_ne!(id1, id2);
        assert_eq!(system.sources.len(), 2);
    }

    #[test]
    fn test_recording() {
        let mut system = AudioSystem::new(AudioConfig::default());

        system.start_recording();
        assert!(system.is_recording);

        let (left, right) = system.stop_recording();
        assert!(!system.is_recording);
        assert!(left.is_empty());
        assert!(right.is_empty());
    }
}
