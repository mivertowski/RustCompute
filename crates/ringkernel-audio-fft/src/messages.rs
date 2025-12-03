//! Message types for audio FFT processing and K2K communication.
//!
//! This module defines all message types used for communication between
//! the host and GPU bin actors, as well as between neighboring bin actors.

use bytemuck::{Pod, Zeroable};
use ringkernel_core::prelude::*;
use rkyv::{Archive, Deserialize, Serialize};

/// Message type IDs for audio FFT messages.
pub mod message_types {
    /// Audio frame input message.
    pub const AUDIO_FRAME: u64 = 0x4155_4449_4F00_0001; // "AUDIO" + 1
    /// Frequency bin data message.
    pub const FREQUENCY_BIN: u64 = 0x4155_4449_4F00_0002;
    /// Neighbor data exchange message.
    pub const NEIGHBOR_DATA: u64 = 0x4155_4449_4F00_0003;
    /// Separated bin output message.
    pub const SEPARATED_BIN: u64 = 0x4155_4449_4F00_0004;
    /// Processing complete signal.
    pub const FRAME_COMPLETE: u64 = 0x4155_4449_4F00_0005;
    /// Bin actor control message.
    pub const BIN_CONTROL: u64 = 0x4155_4449_4F00_0006;
}

/// A frame of audio samples for FFT processing.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct AudioFrame {
    /// Unique frame identifier.
    pub frame_id: u64,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo).
    pub channels: u8,
    /// Audio samples (interleaved if stereo).
    pub samples: Vec<f32>,
    /// Timestamp in samples from start.
    pub timestamp_samples: u64,
}

impl AudioFrame {
    /// Create a new audio frame.
    pub fn new(
        frame_id: u64,
        sample_rate: u32,
        channels: u8,
        samples: Vec<f32>,
        timestamp_samples: u64,
    ) -> Self {
        Self {
            frame_id,
            sample_rate,
            channels,
            samples,
            timestamp_samples,
        }
    }

    /// Get the duration of this frame in seconds.
    pub fn duration_secs(&self) -> f64 {
        let sample_count = self.samples.len() / self.channels as usize;
        sample_count as f64 / self.sample_rate as f64
    }

    /// Get samples for a specific channel.
    pub fn channel_samples(&self, channel: usize) -> Vec<f32> {
        if channel >= self.channels as usize {
            return Vec::new();
        }
        self.samples
            .iter()
            .skip(channel)
            .step_by(self.channels as usize)
            .copied()
            .collect()
    }
}

impl RingMessage for AudioFrame {
    fn message_type() -> u64 {
        message_types::AUDIO_FRAME
    }

    fn message_id(&self) -> MessageId {
        MessageId::new(self.frame_id)
    }

    fn serialize(&self) -> Vec<u8> {
        rkyv::to_bytes::<_, 256>(self)
            .map(|v| v.to_vec())
            .unwrap_or_default()
    }

    fn deserialize(bytes: &[u8]) -> ringkernel_core::error::Result<Self> {
        // SAFETY: We trust the serialized data from our own FFT processing
        let archived = unsafe { rkyv::archived_root::<Self>(bytes) };
        Ok(archived.deserialize(&mut rkyv::Infallible).unwrap())
    }
}

/// Complex number representation for FFT bins.
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable, Archive, Serialize, Deserialize)]
#[repr(C)]
pub struct Complex {
    /// Real component.
    pub re: f32,
    /// Imaginary component.
    pub im: f32,
}

impl Complex {
    /// Create a new complex number.
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// Create from polar coordinates.
    pub fn from_polar(magnitude: f32, phase: f32) -> Self {
        Self {
            re: magnitude * phase.cos(),
            im: magnitude * phase.sin(),
        }
    }

    /// Get magnitude.
    pub fn magnitude(&self) -> f32 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    /// Get phase in radians.
    pub fn phase(&self) -> f32 {
        self.im.atan2(self.re)
    }

    /// Get magnitude squared (more efficient than magnitude).
    pub fn magnitude_squared(&self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// Multiply by another complex number.
    pub fn mul(&self, other: &Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    /// Complex conjugate.
    pub fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Add another complex number.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    /// Scale by a real number.
    pub fn scale(&self, s: f32) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

/// Data for a single frequency bin.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct FrequencyBin {
    /// Frame this bin belongs to.
    pub frame_id: u64,
    /// Bin index (0 = DC, fft_size/2 = Nyquist).
    pub bin_index: u32,
    /// Total number of bins.
    pub total_bins: u32,
    /// Complex FFT value.
    pub value: Complex,
    /// Frequency in Hz for this bin.
    pub frequency_hz: f32,
    /// Previous frame's value (for temporal analysis).
    pub prev_value: Option<Complex>,
}

impl FrequencyBin {
    /// Create a new frequency bin.
    pub fn new(
        frame_id: u64,
        bin_index: u32,
        total_bins: u32,
        value: Complex,
        frequency_hz: f32,
    ) -> Self {
        Self {
            frame_id,
            bin_index,
            total_bins,
            value,
            frequency_hz,
            prev_value: None,
        }
    }

    /// Set the previous frame's value.
    pub fn with_prev_value(mut self, prev: Complex) -> Self {
        self.prev_value = Some(prev);
        self
    }

    /// Get magnitude in dB.
    pub fn magnitude_db(&self) -> f32 {
        let mag = self.value.magnitude();
        if mag > 1e-10 {
            20.0 * mag.log10()
        } else {
            -200.0 // Floor
        }
    }
}

impl RingMessage for FrequencyBin {
    fn message_type() -> u64 {
        message_types::FREQUENCY_BIN
    }

    fn message_id(&self) -> MessageId {
        MessageId::new(self.frame_id * 10000 + self.bin_index as u64)
    }

    fn serialize(&self) -> Vec<u8> {
        rkyv::to_bytes::<_, 128>(self)
            .map(|v| v.to_vec())
            .unwrap_or_default()
    }

    fn deserialize(bytes: &[u8]) -> ringkernel_core::error::Result<Self> {
        // SAFETY: We trust the serialized data from our own FFT processing
        let archived = unsafe { rkyv::archived_root::<Self>(bytes) };
        Ok(archived.deserialize(&mut rkyv::Infallible).unwrap())
    }
}

/// Neighbor data exchanged via K2K messaging.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct NeighborData {
    /// Source bin index.
    pub source_bin: u32,
    /// Frame ID.
    pub frame_id: u64,
    /// Complex value.
    pub value: Complex,
    /// Magnitude.
    pub magnitude: f32,
    /// Phase.
    pub phase: f32,
    /// Temporal derivative (phase change from previous frame).
    pub phase_derivative: f32,
    /// Spectral flux (magnitude change).
    pub spectral_flux: f32,
}

impl NeighborData {
    /// Create neighbor data from a frequency bin.
    pub fn from_bin(bin: &FrequencyBin, phase_derivative: f32, spectral_flux: f32) -> Self {
        Self {
            source_bin: bin.bin_index,
            frame_id: bin.frame_id,
            value: bin.value,
            magnitude: bin.value.magnitude(),
            phase: bin.value.phase(),
            phase_derivative,
            spectral_flux,
        }
    }
}

impl RingMessage for NeighborData {
    fn message_type() -> u64 {
        message_types::NEIGHBOR_DATA
    }

    fn message_id(&self) -> MessageId {
        MessageId::new(self.frame_id * 10000 + self.source_bin as u64)
    }

    fn serialize(&self) -> Vec<u8> {
        rkyv::to_bytes::<_, 64>(self)
            .map(|v| v.to_vec())
            .unwrap_or_default()
    }

    fn deserialize(bytes: &[u8]) -> ringkernel_core::error::Result<Self> {
        // SAFETY: We trust the serialized data from our own FFT processing
        let archived = unsafe { rkyv::archived_root::<Self>(bytes) };
        Ok(archived.deserialize(&mut rkyv::Infallible).unwrap())
    }
}

/// Separated frequency bin with direct and ambience components.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct SeparatedBin {
    /// Frame ID.
    pub frame_id: u64,
    /// Bin index.
    pub bin_index: u32,
    /// Direct signal component.
    pub direct: Complex,
    /// Ambience component.
    pub ambience: Complex,
    /// Coherence score (0.0 = pure ambience, 1.0 = pure direct).
    pub coherence: f32,
    /// Transient score (0.0 = steady, 1.0 = transient).
    pub transient: f32,
}

impl SeparatedBin {
    /// Create a new separated bin.
    pub fn new(
        frame_id: u64,
        bin_index: u32,
        direct: Complex,
        ambience: Complex,
        coherence: f32,
        transient: f32,
    ) -> Self {
        Self {
            frame_id,
            bin_index,
            direct,
            ambience,
            coherence,
            transient,
        }
    }

    /// Get the original combined value.
    pub fn combined(&self) -> Complex {
        self.direct.add(&self.ambience)
    }

    /// Get direct signal magnitude ratio.
    pub fn direct_ratio(&self) -> f32 {
        let total = self.direct.magnitude() + self.ambience.magnitude();
        if total > 1e-10 {
            self.direct.magnitude() / total
        } else {
            0.5
        }
    }
}

impl RingMessage for SeparatedBin {
    fn message_type() -> u64 {
        message_types::SEPARATED_BIN
    }

    fn message_id(&self) -> MessageId {
        MessageId::new(self.frame_id * 10000 + self.bin_index as u64)
    }

    fn serialize(&self) -> Vec<u8> {
        rkyv::to_bytes::<_, 128>(self)
            .map(|v| v.to_vec())
            .unwrap_or_default()
    }

    fn deserialize(bytes: &[u8]) -> ringkernel_core::error::Result<Self> {
        // SAFETY: We trust the serialized data from our own FFT processing
        let archived = unsafe { rkyv::archived_root::<Self>(bytes) };
        Ok(archived.deserialize(&mut rkyv::Infallible).unwrap())
    }
}

/// Control message for bin actors.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub enum BinControl {
    /// Reset state for new audio stream.
    Reset,
    /// Update separation parameters.
    UpdateParams {
        /// Coherence threshold.
        coherence_threshold: f32,
        /// Transient sensitivity.
        transient_sensitivity: f32,
        /// Temporal smoothing factor.
        temporal_smoothing: f32,
    },
    /// Request current state.
    GetState,
    /// Shutdown the actor.
    Shutdown,
}

impl RingMessage for BinControl {
    fn message_type() -> u64 {
        message_types::BIN_CONTROL
    }

    fn message_id(&self) -> MessageId {
        MessageId::generate()
    }

    fn serialize(&self) -> Vec<u8> {
        rkyv::to_bytes::<_, 64>(self)
            .map(|v| v.to_vec())
            .unwrap_or_default()
    }

    fn deserialize(bytes: &[u8]) -> ringkernel_core::error::Result<Self> {
        // SAFETY: We trust the serialized data from our own FFT processing
        let archived = unsafe { rkyv::archived_root::<Self>(bytes) };
        Ok(archived.deserialize(&mut rkyv::Infallible).unwrap())
    }
}

/// Frame processing complete notification.
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
pub struct FrameComplete {
    /// Frame ID.
    pub frame_id: u64,
    /// Processing time in microseconds.
    pub processing_time_us: u64,
    /// Number of bins processed.
    pub bins_processed: u32,
}

impl RingMessage for FrameComplete {
    fn message_type() -> u64 {
        message_types::FRAME_COMPLETE
    }

    fn message_id(&self) -> MessageId {
        MessageId::new(self.frame_id)
    }

    fn serialize(&self) -> Vec<u8> {
        rkyv::to_bytes::<_, 32>(self)
            .map(|v| v.to_vec())
            .unwrap_or_default()
    }

    fn deserialize(bytes: &[u8]) -> ringkernel_core::error::Result<Self> {
        // SAFETY: We trust the serialized data from our own FFT processing
        let archived = unsafe { rkyv::archived_root::<Self>(bytes) };
        Ok(archived.deserialize(&mut rkyv::Infallible).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_operations() {
        let c1 = Complex::new(3.0, 4.0);
        assert!((c1.magnitude() - 5.0).abs() < 1e-6);

        let c2 = Complex::from_polar(5.0, 0.927295); // approx atan2(4, 3)
        assert!((c2.re - 3.0).abs() < 0.01);
        assert!((c2.im - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_audio_frame_serialization() {
        use ringkernel_core::RingMessage;
        let frame = AudioFrame::new(1, 44100, 2, vec![0.0, 0.1, 0.2, 0.3], 0);
        let bytes = RingMessage::serialize(&frame);
        let restored = <AudioFrame as RingMessage>::deserialize(&bytes).unwrap();
        assert_eq!(restored.frame_id, 1);
        assert_eq!(restored.sample_rate, 44100);
        assert_eq!(restored.samples.len(), 4);
    }

    #[test]
    fn test_frequency_bin_db() {
        let bin = FrequencyBin::new(0, 10, 1024, Complex::new(1.0, 0.0), 440.0);
        assert!((bin.magnitude_db() - 0.0).abs() < 0.1); // 1.0 magnitude = 0 dB

        let quiet_bin = FrequencyBin::new(0, 10, 1024, Complex::new(0.1, 0.0), 440.0);
        assert!((quiet_bin.magnitude_db() - (-20.0)).abs() < 0.1); // 0.1 magnitude = -20 dB
    }
}
