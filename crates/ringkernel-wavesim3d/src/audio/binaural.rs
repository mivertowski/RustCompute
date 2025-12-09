//! Binaural audio system for 3D spatial hearing simulation.
//!
//! Implements a virtual head with two ears for stereo audio output.
//! Uses physical modeling based on:
//! - Interaural Time Difference (ITD)
//! - Interaural Level Difference (ILD)
//! - Head shadow effect

use crate::simulation::physics::{constants, Orientation3D, Position3D};
use crate::simulation::grid3d::SimulationGrid3D;
use std::collections::VecDeque;

/// Virtual head for binaural audio capture.
#[derive(Debug, Clone)]
pub struct VirtualHead {
    /// Center position of the head (between ears)
    pub position: Position3D,
    /// Head orientation (yaw, pitch, roll)
    pub orientation: Orientation3D,
    /// Distance between ears (default: 0.17m)
    pub ear_spacing: f32,
    /// Head radius for shadowing calculations (default: 0.0875m)
    pub head_radius: f32,
}

impl Default for VirtualHead {
    fn default() -> Self {
        Self {
            position: Position3D::origin(),
            orientation: Orientation3D::default(),
            ear_spacing: constants::EAR_SPACING,
            head_radius: 0.0875, // Average adult head radius
        }
    }
}

impl VirtualHead {
    /// Create a new virtual head at the given position.
    pub fn new(position: Position3D) -> Self {
        Self {
            position,
            ..Default::default()
        }
    }

    /// Create a virtual head with custom ear spacing.
    pub fn with_ear_spacing(mut self, spacing: f32) -> Self {
        self.ear_spacing = spacing.max(0.05);
        self
    }

    /// Set head orientation.
    pub fn with_orientation(mut self, orientation: Orientation3D) -> Self {
        self.orientation = orientation;
        self
    }

    /// Move the head to a new position.
    pub fn set_position(&mut self, pos: Position3D) {
        self.position = pos;
    }

    /// Rotate the head.
    pub fn set_orientation(&mut self, orientation: Orientation3D) {
        self.orientation = orientation;
    }

    /// Get the position of the left ear.
    pub fn left_ear_position(&self) -> Position3D {
        let (rx, ry, rz) = self.orientation.right();
        let half_spacing = self.ear_spacing / 2.0;
        Position3D::new(
            self.position.x - rx * half_spacing,
            self.position.y - ry * half_spacing,
            self.position.z - rz * half_spacing,
        )
    }

    /// Get the position of the right ear.
    pub fn right_ear_position(&self) -> Position3D {
        let (rx, ry, rz) = self.orientation.right();
        let half_spacing = self.ear_spacing / 2.0;
        Position3D::new(
            self.position.x + rx * half_spacing,
            self.position.y + ry * half_spacing,
            self.position.z + rz * half_spacing,
        )
    }

    /// Calculate the angle to a sound source relative to the head's forward direction.
    pub fn angle_to_source(&self, source: Position3D) -> (f32, f32) {
        // Vector from head to source
        let dx = source.x - self.position.x;
        let dy = source.y - self.position.y;
        let dz = source.z - self.position.z;

        // Forward and right vectors
        let (fx, fy, fz) = self.orientation.forward();
        let (rx, ry, rz) = self.orientation.right();

        // Project onto forward-right plane for azimuth
        let forward_comp = dx * fx + dy * fy + dz * fz;
        let right_comp = dx * rx + dy * ry + dz * rz;
        let azimuth = right_comp.atan2(forward_comp);

        // Elevation (angle above/below horizontal)
        let horizontal_dist = (dx * dx + dz * dz).sqrt();
        let elevation = dy.atan2(horizontal_dist);

        (azimuth, elevation)
    }

    /// Calculate ITD (Interaural Time Difference) for a source position.
    ///
    /// Returns the time difference in seconds (positive = right ear leads).
    pub fn calculate_itd(&self, source: Position3D, speed_of_sound: f32) -> f32 {
        let left_dist = self.left_ear_position().distance_to(&source);
        let right_dist = self.right_ear_position().distance_to(&source);

        // Time difference based on path length difference
        (left_dist - right_dist) / speed_of_sound
    }

    /// Calculate ILD (Interaural Level Difference) for a source.
    ///
    /// Returns the amplitude ratio (left/right) based on head shadow.
    pub fn calculate_ild(&self, source: Position3D) -> (f32, f32) {
        let (azimuth, _elevation) = self.angle_to_source(source);

        // Simple head shadow model
        // At 90° (source directly to the right), left ear is shadowed
        // At -90° (source directly to the left), right ear is shadowed

        let shadow_factor = 0.4; // Maximum shadow attenuation

        let left_gain;
        let right_gain;

        if azimuth >= 0.0 {
            // Source is to the right
            right_gain = 1.0;
            left_gain = 1.0 - shadow_factor * (azimuth / std::f32::consts::FRAC_PI_2).min(1.0);
        } else {
            // Source is to the left
            left_gain = 1.0;
            right_gain =
                1.0 - shadow_factor * ((-azimuth) / std::f32::consts::FRAC_PI_2).min(1.0);
        }

        (left_gain, right_gain)
    }
}

/// Binaural microphone for capturing stereo audio from the simulation.
pub struct BinauralMicrophone {
    /// Virtual head
    pub head: VirtualHead,
    /// Sample rate for output audio
    pub sample_rate: u32,
    /// Buffer for left ear samples
    left_buffer: VecDeque<f32>,
    /// Buffer for right ear samples
    right_buffer: VecDeque<f32>,
    /// Maximum buffer size
    buffer_size: usize,
    /// Interpolation factor (simulation steps per audio sample)
    interp_factor: f32,
    /// Accumulated simulation samples for interpolation
    accumulated_left: Vec<f32>,
    accumulated_right: Vec<f32>,
}

impl BinauralMicrophone {
    /// Create a new binaural microphone.
    ///
    /// # Arguments
    /// * `head` - Virtual head configuration
    /// * `sample_rate` - Output audio sample rate (e.g., 44100 Hz)
    /// * `simulation_dt` - Simulation time step in seconds
    pub fn new(head: VirtualHead, sample_rate: u32, simulation_dt: f32) -> Self {
        let buffer_size = sample_rate as usize * 2; // 2 seconds buffer

        // Calculate how many simulation steps per audio sample
        let audio_period = 1.0 / sample_rate as f32;
        let interp_factor = audio_period / simulation_dt;

        Self {
            head,
            sample_rate,
            left_buffer: VecDeque::with_capacity(buffer_size),
            right_buffer: VecDeque::with_capacity(buffer_size),
            buffer_size,
            interp_factor,
            accumulated_left: Vec::new(),
            accumulated_right: Vec::new(),
        }
    }

    /// Capture a sample from the simulation grid.
    ///
    /// Should be called once per simulation step.
    pub fn capture(&mut self, grid: &SimulationGrid3D) {
        let left_pos = self.head.left_ear_position();
        let right_pos = self.head.right_ear_position();

        // Sample pressure at ear positions
        let left_pressure = grid.sample_pressure(left_pos);
        let right_pressure = grid.sample_pressure(right_pos);

        // Accumulate samples for resampling
        self.accumulated_left.push(left_pressure);
        self.accumulated_right.push(right_pressure);

        // Downsample to audio rate when we have enough samples
        // Ensure we always remove at least 1 sample to prevent infinite loop
        let min_remove = self.interp_factor.max(1.0).ceil() as usize;
        while self.accumulated_left.len() >= min_remove {
            let n = min_remove;

            // Simple averaging for now (could use better filter)
            let left_sample: f32 =
                self.accumulated_left.iter().take(n).sum::<f32>() / n as f32;
            let right_sample: f32 =
                self.accumulated_right.iter().take(n).sum::<f32>() / n as f32;

            // Add to output buffers
            if self.left_buffer.len() < self.buffer_size {
                self.left_buffer.push_back(left_sample);
                self.right_buffer.push_back(right_sample);
            }

            // Remove processed samples (always at least 1)
            let to_remove = min_remove.max(1);
            self.accumulated_left.drain(0..to_remove.min(self.accumulated_left.len()));
            self.accumulated_right.drain(0..to_remove.min(self.accumulated_right.len()));
        }
    }

    /// Get available stereo samples.
    ///
    /// Returns (left_samples, right_samples).
    pub fn get_samples(&mut self, count: usize) -> (Vec<f32>, Vec<f32>) {
        let count = count.min(self.left_buffer.len());
        let left: Vec<f32> = self.left_buffer.drain(0..count).collect();
        let right: Vec<f32> = self.right_buffer.drain(0..count).collect();
        (left, right)
    }

    /// Get interleaved stereo samples (for audio output).
    pub fn get_interleaved(&mut self, frames: usize) -> Vec<f32> {
        let frames = frames.min(self.left_buffer.len());
        let mut output = Vec::with_capacity(frames * 2);

        for _ in 0..frames {
            if let (Some(left), Some(right)) =
                (self.left_buffer.pop_front(), self.right_buffer.pop_front())
            {
                output.push(left);
                output.push(right);
            }
        }

        output
    }

    /// Get the number of available samples.
    pub fn available_samples(&self) -> usize {
        self.left_buffer.len()
    }

    /// Clear all buffers.
    pub fn clear(&mut self) {
        self.left_buffer.clear();
        self.right_buffer.clear();
        self.accumulated_left.clear();
        self.accumulated_right.clear();
    }

    /// Update head position.
    pub fn set_head_position(&mut self, pos: Position3D) {
        self.head.set_position(pos);
    }

    /// Update head orientation.
    pub fn set_head_orientation(&mut self, orientation: Orientation3D) {
        self.head.set_orientation(orientation);
    }
}

/// Audio output handler using cpal.
#[cfg(feature = "audio-output")]
pub struct AudioOutput {
    stream: Option<cpal::Stream>,
    sample_rate: u32,
}

/// Simple delay line for ITD simulation.
pub struct DelayLine {
    buffer: VecDeque<f32>,
    max_delay_samples: usize,
}

impl DelayLine {
    /// Create a new delay line.
    pub fn new(max_delay_seconds: f32, sample_rate: u32) -> Self {
        let max_delay_samples = (max_delay_seconds * sample_rate as f32).ceil() as usize;
        Self {
            buffer: VecDeque::from(vec![0.0; max_delay_samples]),
            max_delay_samples,
        }
    }

    /// Process a sample with the given delay.
    pub fn process(&mut self, input: f32, delay_samples: f32) -> f32 {
        // Add new sample
        self.buffer.push_back(input);

        // Get delayed sample (with linear interpolation)
        let delay_int = delay_samples.floor() as usize;
        let delay_frac = delay_samples - delay_int as f32;

        let idx1 = self.buffer.len().saturating_sub(1 + delay_int);
        let idx2 = idx1.saturating_sub(1);

        let s1 = self.buffer.get(idx1).copied().unwrap_or(0.0);
        let s2 = self.buffer.get(idx2).copied().unwrap_or(0.0);

        // Remove old samples
        while self.buffer.len() > self.max_delay_samples {
            self.buffer.pop_front();
        }

        // Linear interpolation
        s1 * (1.0 - delay_frac) + s2 * delay_frac
    }

    /// Clear the delay line.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.buffer.resize(self.max_delay_samples, 0.0);
    }
}

/// HRTF-based binaural processor (simplified).
///
/// Applies ITD and ILD based on source position relative to head.
pub struct BinauralProcessor {
    /// Left ear delay line
    left_delay: DelayLine,
    /// Right ear delay line
    right_delay: DelayLine,
    /// Sample rate
    sample_rate: u32,
    /// Speed of sound for ITD calculation
    speed_of_sound: f32,
}

impl BinauralProcessor {
    /// Create a new binaural processor.
    pub fn new(sample_rate: u32, speed_of_sound: f32) -> Self {
        // Maximum ITD is about 0.7ms (ear spacing / speed of sound)
        let max_delay = 0.001;

        Self {
            left_delay: DelayLine::new(max_delay, sample_rate),
            right_delay: DelayLine::new(max_delay, sample_rate),
            sample_rate,
            speed_of_sound,
        }
    }

    /// Process a mono sample into stereo based on source and head position.
    pub fn process(&mut self, input: f32, head: &VirtualHead, source: Position3D) -> (f32, f32) {
        // Calculate ITD
        let itd = head.calculate_itd(source, self.speed_of_sound);
        let itd_samples = itd * self.sample_rate as f32;

        // Calculate ILD
        let (left_gain, right_gain) = head.calculate_ild(source);

        // Apply delays
        let (left_delay, right_delay) = if itd_samples >= 0.0 {
            // Right ear leads (source to the right)
            (itd_samples, 0.0)
        } else {
            // Left ear leads (source to the left)
            (0.0, -itd_samples)
        };

        let left = self.left_delay.process(input * left_gain, left_delay);
        let right = self.right_delay.process(input * right_gain, right_delay);

        (left, right)
    }

    /// Clear all delay buffers.
    pub fn clear(&mut self) {
        self.left_delay.clear();
        self.right_delay.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_virtual_head_ear_positions() {
        let head = VirtualHead::new(Position3D::new(0.0, 0.0, 0.0));

        let left = head.left_ear_position();
        let right = head.right_ear_position();

        // Ears should be symmetric
        assert!((left.x + right.x).abs() < 0.001);

        // Distance between ears should be ear_spacing
        let distance = left.distance_to(&right);
        assert!((distance - head.ear_spacing).abs() < 0.001);
    }

    #[test]
    fn test_itd_calculation() {
        let head = VirtualHead::new(Position3D::new(0.0, 0.0, 0.0));
        let speed = 343.0;

        // Source directly to the right
        let source_right = Position3D::new(10.0, 0.0, 0.0);
        let itd_right = head.calculate_itd(source_right, speed);

        // Source directly to the left
        let source_left = Position3D::new(-10.0, 0.0, 0.0);
        let itd_left = head.calculate_itd(source_left, speed);

        // ITD should be opposite for left vs right sources
        assert!(itd_right > 0.0);
        assert!(itd_left < 0.0);
        assert!((itd_right + itd_left).abs() < 0.001);
    }

    #[test]
    fn test_ild_calculation() {
        let head = VirtualHead::new(Position3D::new(0.0, 0.0, 0.0));

        // Source directly in front
        let source_front = Position3D::new(0.0, 0.0, 10.0);
        let (left_front, right_front) = head.calculate_ild(source_front);

        // Should be equal for source in front
        assert!((left_front - right_front).abs() < 0.1);

        // Source to the right
        let source_right = Position3D::new(10.0, 0.0, 0.0);
        let (left_right, right_right) = head.calculate_ild(source_right);

        // Right ear should have more gain
        assert!(right_right > left_right);
    }

    #[test]
    fn test_delay_line() {
        let mut delay = DelayLine::new(0.01, 44100);

        // Feed impulse
        delay.process(1.0, 0.0);
        for _ in 0..100 {
            delay.process(0.0, 0.0);
        }

        // Delayed output should appear after delay
        let output = delay.process(0.0, 50.0);
        // The impulse should have passed through
        assert!(output.abs() < 0.1);
    }

    #[test]
    fn test_binaural_processor() {
        let mut processor = BinauralProcessor::new(44100, 343.0);
        let head = VirtualHead::new(Position3D::new(0.0, 0.0, 0.0));

        // Source to the right
        let source = Position3D::new(5.0, 0.0, 0.0);

        // Process a burst
        let mut left_energy = 0.0;
        let mut right_energy = 0.0;

        for i in 0..100 {
            let input = if i < 10 { 1.0 } else { 0.0 };
            let (left, right) = processor.process(input, &head, source);
            left_energy += left * left;
            right_energy += right * right;
        }

        // Right ear should have more energy (source is to the right)
        assert!(
            right_energy > left_energy,
            "Right energy {} should be > left {}",
            right_energy,
            left_energy
        );
    }

    #[test]
    fn test_head_rotation() {
        let mut head = VirtualHead::new(Position3D::new(0.0, 0.0, 0.0));

        // Face forward, source to the right
        let source = Position3D::new(5.0, 0.0, 0.0);
        let (left1, right1) = head.calculate_ild(source);

        // Rotate 90° right (now facing the source)
        head.set_orientation(Orientation3D::new(std::f32::consts::FRAC_PI_2, 0.0, 0.0));
        let (left2, right2) = head.calculate_ild(source);

        // After rotation, ILD should be more balanced
        let diff1 = (right1 - left1).abs();
        let diff2 = (right2 - left2).abs();
        assert!(
            diff2 < diff1,
            "After rotation, ILD diff {} should be < {}",
            diff2,
            diff1
        );
    }
}
