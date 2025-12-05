//! Audio sources for 3D acoustic simulation.
//!
//! Supports various audio source types:
//! - Impulse (single spike)
//! - Continuous tone (sinusoidal)
//! - Audio file playback
//! - Real-time microphone input

use crate::simulation::physics::Position3D;
use std::sync::Arc;

/// Type of audio source signal.
#[derive(Debug, Clone)]
pub enum SourceType {
    /// Single impulse (delta function)
    Impulse {
        amplitude: f32,
        fired: bool,
    },
    /// Continuous sinusoidal tone
    Tone {
        frequency_hz: f32,
        amplitude: f32,
        phase: f32,
    },
    /// Noise (white or pink)
    Noise {
        amplitude: f32,
        pink: bool,
        state: [f32; 7], // Pink noise filter state
    },
    /// Audio file playback
    AudioFile {
        samples: Arc<Vec<f32>>,
        sample_rate: u32,
        position: usize,
        looping: bool,
    },
    /// Real-time input buffer
    LiveInput {
        buffer: Arc<std::sync::Mutex<Vec<f32>>>,
        read_pos: usize,
    },
    /// Chirp (frequency sweep)
    Chirp {
        start_freq: f32,
        end_freq: f32,
        duration_s: f32,
        amplitude: f32,
        elapsed: f32,
    },
    /// Gaussian pulse (smooth impulse)
    GaussianPulse {
        center_time: f32,
        sigma: f32,
        amplitude: f32,
        elapsed: f32,
    },
}

impl Default for SourceType {
    fn default() -> Self {
        SourceType::Impulse {
            amplitude: 1.0,
            fired: false,
        }
    }
}

/// A 3D audio source in the simulation.
#[derive(Debug, Clone)]
pub struct AudioSource {
    /// Unique identifier
    pub id: u32,
    /// Position in 3D space (meters)
    pub position: Position3D,
    /// Source signal type
    pub source_type: SourceType,
    /// Whether the source is active
    pub active: bool,
    /// Directivity pattern (1.0 = omnidirectional)
    pub directivity: f32,
    /// Source radius for injection (cells)
    pub injection_radius: f32,
}

impl AudioSource {
    /// Create a new impulse source.
    pub fn impulse(id: u32, position: Position3D, amplitude: f32) -> Self {
        Self {
            id,
            position,
            source_type: SourceType::Impulse {
                amplitude,
                fired: false,
            },
            active: true,
            directivity: 1.0,
            injection_radius: 1.0,
        }
    }

    /// Create a new tone source.
    pub fn tone(id: u32, position: Position3D, frequency_hz: f32, amplitude: f32) -> Self {
        Self {
            id,
            position,
            source_type: SourceType::Tone {
                frequency_hz,
                amplitude,
                phase: 0.0,
            },
            active: true,
            directivity: 1.0,
            injection_radius: 1.0,
        }
    }

    /// Create a noise source.
    pub fn noise(id: u32, position: Position3D, amplitude: f32, pink: bool) -> Self {
        Self {
            id,
            position,
            source_type: SourceType::Noise {
                amplitude,
                pink,
                state: [0.0; 7],
            },
            active: true,
            directivity: 1.0,
            injection_radius: 1.0,
        }
    }

    /// Create a chirp (frequency sweep) source.
    pub fn chirp(
        id: u32,
        position: Position3D,
        start_freq: f32,
        end_freq: f32,
        duration_s: f32,
        amplitude: f32,
    ) -> Self {
        Self {
            id,
            position,
            source_type: SourceType::Chirp {
                start_freq,
                end_freq,
                duration_s,
                amplitude,
                elapsed: 0.0,
            },
            active: true,
            directivity: 1.0,
            injection_radius: 1.0,
        }
    }

    /// Create a Gaussian pulse source.
    pub fn gaussian_pulse(
        id: u32,
        position: Position3D,
        center_time: f32,
        sigma: f32,
        amplitude: f32,
    ) -> Self {
        Self {
            id,
            position,
            source_type: SourceType::GaussianPulse {
                center_time,
                sigma,
                amplitude,
                elapsed: 0.0,
            },
            active: true,
            directivity: 1.0,
            injection_radius: 1.0,
        }
    }

    /// Create an audio file source.
    pub fn from_samples(
        id: u32,
        position: Position3D,
        samples: Vec<f32>,
        sample_rate: u32,
        looping: bool,
    ) -> Self {
        Self {
            id,
            position,
            source_type: SourceType::AudioFile {
                samples: Arc::new(samples),
                sample_rate,
                position: 0,
                looping,
            },
            active: true,
            directivity: 1.0,
            injection_radius: 1.0,
        }
    }

    /// Set the injection radius (in grid cells).
    pub fn with_radius(mut self, radius: f32) -> Self {
        self.injection_radius = radius.max(0.5);
        self
    }

    /// Set directivity (1.0 = omnidirectional).
    pub fn with_directivity(mut self, directivity: f32) -> Self {
        self.directivity = directivity.clamp(0.0, 1.0);
        self
    }

    /// Move the source to a new position.
    pub fn set_position(&mut self, pos: Position3D) {
        self.position = pos;
    }

    /// Get the next sample value and advance the source state.
    ///
    /// Returns the pressure value to inject at this time step.
    pub fn next_sample(&mut self, time_step: f32) -> f32 {
        if !self.active {
            return 0.0;
        }

        match &mut self.source_type {
            SourceType::Impulse { amplitude, fired } => {
                if !*fired {
                    *fired = true;
                    *amplitude
                } else {
                    0.0
                }
            }
            SourceType::Tone {
                frequency_hz,
                amplitude,
                phase,
            } => {
                let sample = *amplitude * (*phase * 2.0 * std::f32::consts::PI).sin();
                *phase += *frequency_hz * time_step;
                if *phase > 1.0 {
                    *phase -= 1.0;
                }
                sample
            }
            SourceType::Noise {
                amplitude,
                pink,
                state,
            } => {
                // Generate white noise
                let white = (rand::random::<f32>() * 2.0 - 1.0) * *amplitude;

                if *pink {
                    // Paul Kellet's pink noise filter
                    state[0] = 0.99886 * state[0] + white * 0.0555179;
                    state[1] = 0.99332 * state[1] + white * 0.0750759;
                    state[2] = 0.96900 * state[2] + white * 0.1538520;
                    state[3] = 0.86650 * state[3] + white * 0.3104856;
                    state[4] = 0.55000 * state[4] + white * 0.5329522;
                    state[5] = -0.7616 * state[5] - white * 0.0168980;
                    let pink = state[0] + state[1] + state[2] + state[3] + state[4] + state[5]
                        + state[6]
                        + white * 0.5362;
                    state[6] = white * 0.115926;
                    pink * 0.11
                } else {
                    white
                }
            }
            SourceType::AudioFile {
                samples,
                sample_rate,
                position,
                looping,
            } => {
                if *position >= samples.len() {
                    if *looping {
                        *position = 0;
                    } else {
                        return 0.0;
                    }
                }

                // Simple nearest-neighbor resampling
                // TODO: Implement proper resampling
                let sample = samples[*position];
                let samples_per_step = (*sample_rate as f32 * time_step) as usize;
                *position += samples_per_step.max(1);

                sample
            }
            SourceType::LiveInput { buffer, read_pos } => {
                if let Ok(buf) = buffer.lock() {
                    if *read_pos < buf.len() {
                        let sample = buf[*read_pos];
                        *read_pos += 1;
                        return sample;
                    }
                }
                0.0
            }
            SourceType::Chirp {
                start_freq,
                end_freq,
                duration_s,
                amplitude,
                elapsed,
            } => {
                if *elapsed >= *duration_s {
                    return 0.0;
                }

                let t = *elapsed / *duration_s;
                // Logarithmic frequency sweep
                let freq = *start_freq * (*end_freq / *start_freq).powf(t);
                let phase = 2.0 * std::f32::consts::PI * freq * *elapsed;
                let sample = *amplitude * phase.sin();

                *elapsed += time_step;
                sample
            }
            SourceType::GaussianPulse {
                center_time,
                sigma,
                amplitude,
                elapsed,
            } => {
                let t = *elapsed - *center_time;
                let gaussian = (-t * t / (2.0 * *sigma * *sigma)).exp();
                let sample = *amplitude * gaussian;

                *elapsed += time_step;
                sample
            }
        }
    }

    /// Reset the source to its initial state.
    pub fn reset(&mut self) {
        match &mut self.source_type {
            SourceType::Impulse { fired, .. } => *fired = false,
            SourceType::Tone { phase, .. } => *phase = 0.0,
            SourceType::Noise { state, .. } => *state = [0.0; 7],
            SourceType::AudioFile { position, .. } => *position = 0,
            SourceType::LiveInput { read_pos, .. } => *read_pos = 0,
            SourceType::Chirp { elapsed, .. } => *elapsed = 0.0,
            SourceType::GaussianPulse { elapsed, .. } => *elapsed = 0.0,
        }
    }

    /// Check if the source has finished (for one-shot sources).
    pub fn is_finished(&self) -> bool {
        match &self.source_type {
            SourceType::Impulse { fired, .. } => *fired,
            SourceType::AudioFile {
                samples,
                position,
                looping,
                ..
            } => !*looping && *position >= samples.len(),
            SourceType::Chirp {
                duration_s,
                elapsed,
                ..
            } => *elapsed >= *duration_s,
            SourceType::GaussianPulse { elapsed, sigma, .. } => *elapsed > sigma * 6.0,
            _ => false, // Continuous sources never finish
        }
    }
}

/// Manager for multiple audio sources.
#[derive(Default)]
pub struct SourceManager {
    sources: Vec<AudioSource>,
    next_id: u32,
}

impl SourceManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a source and return its ID.
    pub fn add(&mut self, mut source: AudioSource) -> u32 {
        let id = self.next_id;
        source.id = id;
        self.next_id += 1;
        self.sources.push(source);
        id
    }

    /// Remove a source by ID.
    pub fn remove(&mut self, id: u32) -> bool {
        if let Some(pos) = self.sources.iter().position(|s| s.id == id) {
            self.sources.remove(pos);
            true
        } else {
            false
        }
    }

    /// Get a reference to a source by ID.
    pub fn get(&self, id: u32) -> Option<&AudioSource> {
        self.sources.iter().find(|s| s.id == id)
    }

    /// Get a mutable reference to a source by ID.
    pub fn get_mut(&mut self, id: u32) -> Option<&mut AudioSource> {
        self.sources.iter_mut().find(|s| s.id == id)
    }

    /// Iterate over all sources.
    pub fn iter(&self) -> impl Iterator<Item = &AudioSource> {
        self.sources.iter()
    }

    /// Iterate mutably over all sources.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut AudioSource> {
        self.sources.iter_mut()
    }

    /// Get the number of sources.
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// Check if there are no sources.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// Reset all sources.
    pub fn reset_all(&mut self) {
        for source in &mut self.sources {
            source.reset();
        }
    }

    /// Remove finished one-shot sources.
    pub fn cleanup_finished(&mut self) {
        self.sources.retain(|s| !s.is_finished() || s.active);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_impulse_source() {
        let mut source = AudioSource::impulse(0, Position3D::origin(), 1.0);
        let dt = 0.001;

        // First sample should be the impulse
        assert_eq!(source.next_sample(dt), 1.0);

        // Subsequent samples should be zero
        assert_eq!(source.next_sample(dt), 0.0);
        assert_eq!(source.next_sample(dt), 0.0);

        // After reset, should fire again
        source.reset();
        assert_eq!(source.next_sample(dt), 1.0);
    }

    #[test]
    fn test_tone_source() {
        let mut source = AudioSource::tone(0, Position3D::origin(), 440.0, 1.0);
        let dt = 0.001;

        // Generate some samples
        let mut samples = Vec::new();
        for _ in 0..100 {
            samples.push(source.next_sample(dt));
        }

        // Should have oscillating values
        let max = samples.iter().fold(0.0_f32, |a, &b| a.max(b));
        let min = samples.iter().fold(0.0_f32, |a, &b| a.min(b));

        assert!(max > 0.5, "Max should be positive: {}", max);
        assert!(min < -0.5, "Min should be negative: {}", min);
    }

    #[test]
    fn test_chirp_source() {
        let mut source = AudioSource::chirp(0, Position3D::origin(), 100.0, 1000.0, 0.01, 1.0);
        let dt = 0.0001;

        let mut count = 0;
        while source.next_sample(dt) != 0.0 || count < 10 {
            count += 1;
            if count > 1000 {
                break;
            }
        }

        // Should finish after duration
        assert!(source.is_finished());
    }

    #[test]
    fn test_source_manager() {
        let mut manager = SourceManager::new();

        let id1 = manager.add(AudioSource::impulse(0, Position3D::origin(), 1.0));
        let id2 = manager.add(AudioSource::tone(0, Position3D::new(1.0, 0.0, 0.0), 440.0, 0.5));

        assert_eq!(manager.len(), 2);

        // IDs should be unique
        assert_ne!(id1, id2);

        // Should be able to get sources
        assert!(manager.get(id1).is_some());
        assert!(manager.get(id2).is_some());

        // Remove one
        assert!(manager.remove(id1));
        assert_eq!(manager.len(), 1);
        assert!(manager.get(id1).is_none());
    }

    #[test]
    fn test_gaussian_pulse() {
        let mut source =
            AudioSource::gaussian_pulse(0, Position3D::origin(), 0.005, 0.001, 1.0);
        let dt = 0.0001;

        let mut max_val = 0.0_f32;
        for _ in 0..200 {
            let sample = source.next_sample(dt);
            max_val = max_val.max(sample.abs());
        }

        // Should have a significant pulse
        assert!(max_val > 0.5, "Peak should be significant: {}", max_val);
    }
}
