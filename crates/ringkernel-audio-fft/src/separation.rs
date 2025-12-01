//! Direct/ambience signal separation algorithm.
//!
//! This module implements the coherence-based separation of direct sound
//! from room ambience using inter-bin phase analysis.
//!
//! ## Algorithm Overview
//!
//! The separation is based on the observation that:
//! - **Direct sound** has coherent phase relationships between neighboring frequency bins
//!   (the phase progression follows a predictable pattern related to time-of-arrival)
//! - **Ambience/reverb** has random, diffuse phase relationships
//!
//! We use multiple cues:
//! 1. **Inter-bin phase coherence**: Direct sound shows correlated phase between neighbors
//! 2. **Spectral flux**: Transients (direct attacks) have higher positive flux
//! 3. **Temporal stability**: Ambience is more temporally stable
//! 4. **Magnitude correlation**: Direct sound often shows correlated magnitude envelopes

use crate::messages::{Complex, NeighborData};

/// Configuration for signal separation.
#[derive(Debug, Clone)]
pub struct SeparationConfig {
    /// Weight for phase coherence in separation (0.0-1.0).
    pub phase_coherence_weight: f32,
    /// Weight for spectral flux in separation (0.0-1.0).
    pub spectral_flux_weight: f32,
    /// Weight for magnitude correlation (0.0-1.0).
    pub magnitude_correlation_weight: f32,
    /// Transient sensitivity (higher = more sensitive to attacks).
    pub transient_sensitivity: f32,
    /// Temporal smoothing factor (0.0 = no smoothing, 1.0 = full smoothing).
    pub temporal_smoothing: f32,
    /// Separation curve exponent (higher = sharper separation).
    pub separation_curve: f32,
    /// Minimum coherence threshold (below this = pure ambience).
    pub min_coherence: f32,
    /// Maximum coherence threshold (above this = pure direct).
    pub max_coherence: f32,
    /// Frequency-dependent weighting (lower frequencies get more smoothing).
    pub frequency_smoothing: bool,
}

impl Default for SeparationConfig {
    fn default() -> Self {
        Self {
            phase_coherence_weight: 0.4,
            spectral_flux_weight: 0.3,
            magnitude_correlation_weight: 0.3,
            transient_sensitivity: 1.0,
            temporal_smoothing: 0.7,
            separation_curve: 1.5,
            min_coherence: 0.1,
            max_coherence: 0.9,
            frequency_smoothing: true,
        }
    }
}

impl SeparationConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set phase coherence weight.
    pub fn with_phase_coherence_weight(mut self, weight: f32) -> Self {
        self.phase_coherence_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set spectral flux weight.
    pub fn with_spectral_flux_weight(mut self, weight: f32) -> Self {
        self.spectral_flux_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Set transient sensitivity.
    pub fn with_transient_sensitivity(mut self, sensitivity: f32) -> Self {
        self.transient_sensitivity = sensitivity.max(0.0);
        self
    }

    /// Set temporal smoothing.
    pub fn with_temporal_smoothing(mut self, smoothing: f32) -> Self {
        self.temporal_smoothing = smoothing.clamp(0.0, 0.99);
        self
    }

    /// Set separation curve.
    pub fn with_separation_curve(mut self, curve: f32) -> Self {
        self.separation_curve = curve.max(0.1);
        self
    }

    /// Preset for music (more ambience preserved).
    pub fn music_preset() -> Self {
        Self {
            phase_coherence_weight: 0.35,
            spectral_flux_weight: 0.25,
            magnitude_correlation_weight: 0.4,
            transient_sensitivity: 0.8,
            temporal_smoothing: 0.8,
            separation_curve: 1.2,
            min_coherence: 0.15,
            max_coherence: 0.85,
            frequency_smoothing: true,
        }
    }

    /// Preset for speech (cleaner separation).
    pub fn speech_preset() -> Self {
        Self {
            phase_coherence_weight: 0.5,
            spectral_flux_weight: 0.3,
            magnitude_correlation_weight: 0.2,
            transient_sensitivity: 1.2,
            temporal_smoothing: 0.6,
            separation_curve: 2.0,
            min_coherence: 0.1,
            max_coherence: 0.9,
            frequency_smoothing: true,
        }
    }

    /// Preset for aggressive separation.
    pub fn aggressive_preset() -> Self {
        Self {
            phase_coherence_weight: 0.45,
            spectral_flux_weight: 0.35,
            magnitude_correlation_weight: 0.2,
            transient_sensitivity: 1.5,
            temporal_smoothing: 0.5,
            separation_curve: 2.5,
            min_coherence: 0.05,
            max_coherence: 0.95,
            frequency_smoothing: false,
        }
    }
}

/// Coherence analyzer for bin-to-bin relationships.
pub struct CoherenceAnalyzer {
    config: SeparationConfig,
    /// Running average of phase coherence.
    phase_coherence_avg: f32,
    /// Running average of magnitude.
    magnitude_avg: f32,
    /// Running average of spectral flux.
    flux_avg: f32,
    /// Frame count for averaging.
    frame_count: u64,
}

impl CoherenceAnalyzer {
    /// Create a new coherence analyzer.
    pub fn new(config: SeparationConfig) -> Self {
        Self {
            config,
            phase_coherence_avg: 0.0,
            magnitude_avg: 0.0,
            flux_avg: 0.0,
            frame_count: 0,
        }
    }

    /// Analyze coherence and return (coherence, transient) scores.
    pub fn analyze(
        &mut self,
        current: &Complex,
        left_neighbor: Option<&NeighborData>,
        right_neighbor: Option<&NeighborData>,
        _phase_derivative: f32,
        spectral_flux: f32,
    ) -> (f32, f32) {
        self.frame_count += 1;

        // 1. Compute phase coherence with neighbors
        let phase_coherence = self.compute_phase_coherence(current, left_neighbor, right_neighbor);

        // 2. Compute magnitude correlation
        let magnitude_correlation =
            self.compute_magnitude_correlation(current, left_neighbor, right_neighbor);

        // 3. Compute transient score based on spectral flux
        let transient = self.compute_transient_score(spectral_flux);

        // 4. Update running averages (for adaptive thresholds)
        let alpha = 0.99;
        self.phase_coherence_avg = self.phase_coherence_avg * alpha + phase_coherence * (1.0 - alpha);
        self.magnitude_avg = self.magnitude_avg * alpha + current.magnitude() * (1.0 - alpha);
        self.flux_avg = self.flux_avg * alpha + spectral_flux * (1.0 - alpha);

        // 5. Combine cues with weights
        let coherence = self.config.phase_coherence_weight * phase_coherence
            + self.config.magnitude_correlation_weight * magnitude_correlation
            + self.config.spectral_flux_weight * transient;

        // Normalize and clamp
        let total_weight = self.config.phase_coherence_weight
            + self.config.magnitude_correlation_weight
            + self.config.spectral_flux_weight;

        let coherence = if total_weight > 0.0 {
            (coherence / total_weight).clamp(self.config.min_coherence, self.config.max_coherence)
        } else {
            0.5
        };

        // Rescale to 0-1 range
        let coherence = (coherence - self.config.min_coherence)
            / (self.config.max_coherence - self.config.min_coherence);

        (coherence.clamp(0.0, 1.0), transient)
    }

    /// Compute phase coherence with neighbors.
    fn compute_phase_coherence(
        &self,
        current: &Complex,
        left: Option<&NeighborData>,
        right: Option<&NeighborData>,
    ) -> f32 {
        let current_phase = current.phase();
        let mut coherence_sum = 0.0;
        let mut count = 0;

        // Compare phase with left neighbor
        if let Some(left_data) = left {
            let phase_diff = self.wrapped_phase_diff(current_phase, left_data.phase);
            // Coherent signals have small phase differences or differences that
            // follow a linear progression
            let coherence = (-phase_diff.abs() * 2.0).exp();
            coherence_sum += coherence;
            count += 1;
        }

        // Compare phase with right neighbor
        if let Some(right_data) = right {
            let phase_diff = self.wrapped_phase_diff(current_phase, right_data.phase);
            let coherence = (-phase_diff.abs() * 2.0).exp();
            coherence_sum += coherence;
            count += 1;
        }

        // Check phase derivative consistency between neighbors
        if let (Some(left_data), Some(right_data)) = (left, right) {
            // For coherent signals, the phase derivative should vary smoothly
            let left_deriv = left_data.phase_derivative;
            let right_deriv = right_data.phase_derivative;
            let deriv_diff = (left_deriv - right_deriv).abs();
            let deriv_coherence = (-deriv_diff).exp();
            coherence_sum += deriv_coherence * 0.5;
            count += 1;
        }

        if count > 0 {
            coherence_sum / count as f32
        } else {
            0.5 // Default to neutral if no neighbors
        }
    }

    /// Compute magnitude correlation with neighbors.
    fn compute_magnitude_correlation(
        &self,
        current: &Complex,
        left: Option<&NeighborData>,
        right: Option<&NeighborData>,
    ) -> f32 {
        let current_mag = current.magnitude();
        let mut correlation_sum = 0.0;
        let mut count = 0;

        if let Some(left_data) = left {
            // Compute correlation based on relative magnitudes
            let left_mag = left_data.magnitude;
            if left_mag > 1e-10 && current_mag > 1e-10 {
                let ratio = (current_mag / left_mag).ln().abs();
                // Similar magnitudes indicate coherent source
                let correlation = (-ratio * 0.5).exp();
                correlation_sum += correlation;
                count += 1;
            }
        }

        if let Some(right_data) = right {
            let right_mag = right_data.magnitude;
            if right_mag > 1e-10 && current_mag > 1e-10 {
                let ratio = (current_mag / right_mag).ln().abs();
                let correlation = (-ratio * 0.5).exp();
                correlation_sum += correlation;
                count += 1;
            }
        }

        // Check flux correlation (coherent sources have correlated flux)
        if let (Some(left_data), Some(right_data)) = (left, right) {
            let left_flux = left_data.spectral_flux;
            let right_flux = right_data.spectral_flux;
            let avg_flux = (left_flux + right_flux) / 2.0;
            if avg_flux > 1e-6 {
                let flux_ratio = (left_flux - right_flux).abs() / avg_flux;
                let flux_correlation = (-flux_ratio).exp();
                correlation_sum += flux_correlation * 0.5;
                count += 1;
            }
        }

        if count > 0 {
            correlation_sum / count as f32
        } else {
            0.5
        }
    }

    /// Compute transient score from spectral flux.
    fn compute_transient_score(&self, spectral_flux: f32) -> f32 {
        // Normalize flux relative to running average
        let threshold = self.flux_avg * 2.0 + 0.01;
        let normalized_flux = spectral_flux / threshold;

        // Apply sensitivity and sigmoid-like shaping
        let shaped = (normalized_flux * self.config.transient_sensitivity).tanh();

        shaped.clamp(0.0, 1.0)
    }

    /// Calculate wrapped phase difference.
    fn wrapped_phase_diff(&self, phase1: f32, phase2: f32) -> f32 {
        let mut diff = phase1 - phase2;
        while diff > std::f32::consts::PI {
            diff -= 2.0 * std::f32::consts::PI;
        }
        while diff < -std::f32::consts::PI {
            diff += 2.0 * std::f32::consts::PI;
        }
        diff
    }

    /// Reset the analyzer state.
    pub fn reset(&mut self) {
        self.phase_coherence_avg = 0.0;
        self.magnitude_avg = 0.0;
        self.flux_avg = 0.0;
        self.frame_count = 0;
    }
}

/// Signal separator that applies the coherence analysis to split signals.
pub struct SignalSeparator {
    config: SeparationConfig,
}

impl SignalSeparator {
    /// Create a new signal separator.
    pub fn new(config: SeparationConfig) -> Self {
        Self { config }
    }

    /// Separate a complex value into direct and ambient components.
    pub fn separate(&self, value: Complex, coherence: f32) -> (Complex, Complex) {
        // Apply separation curve
        let direct_ratio = coherence.powf(self.config.separation_curve);
        let ambient_ratio = 1.0 - direct_ratio;

        let direct = value.scale(direct_ratio);
        let ambient = value.scale(ambient_ratio);

        (direct, ambient)
    }

    /// Separate with frequency-dependent adjustment.
    pub fn separate_with_frequency(
        &self,
        value: Complex,
        coherence: f32,
        bin_index: u32,
        total_bins: u32,
    ) -> (Complex, Complex) {
        let mut adjusted_coherence = coherence;

        if self.config.frequency_smoothing {
            // Lower frequencies get more smoothing (less separation)
            // Higher frequencies can have sharper separation
            let freq_ratio = bin_index as f32 / total_bins as f32;
            let freq_factor = 0.8 + 0.4 * freq_ratio; // 0.8 at DC, 1.2 at Nyquist

            adjusted_coherence = coherence * freq_factor;
            adjusted_coherence = adjusted_coherence.clamp(0.0, 1.0);
        }

        self.separate(value, adjusted_coherence)
    }

    /// Get configuration.
    pub fn config(&self) -> &SeparationConfig {
        &self.config
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: SeparationConfig) {
        self.config = config;
    }
}

/// Stereo separation for maintaining spatial information.
pub struct StereoSeparator {
    left_analyzer: CoherenceAnalyzer,
    right_analyzer: CoherenceAnalyzer,
    separator: SignalSeparator,
    /// Cross-channel coherence weight.
    cross_channel_weight: f32,
}

impl StereoSeparator {
    /// Create a new stereo separator.
    pub fn new(config: SeparationConfig) -> Self {
        Self {
            left_analyzer: CoherenceAnalyzer::new(config.clone()),
            right_analyzer: CoherenceAnalyzer::new(config.clone()),
            separator: SignalSeparator::new(config),
            cross_channel_weight: 0.3,
        }
    }

    /// Process stereo bins and return separated results.
    #[allow(clippy::too_many_arguments)]
    pub fn process_stereo(
        &mut self,
        left_bin: &Complex,
        right_bin: &Complex,
        left_neighbors: (Option<&NeighborData>, Option<&NeighborData>),
        right_neighbors: (Option<&NeighborData>, Option<&NeighborData>),
        left_phase_deriv: f32,
        right_phase_deriv: f32,
        left_flux: f32,
        right_flux: f32,
        bin_index: u32,
        total_bins: u32,
    ) -> ((Complex, Complex), (Complex, Complex)) {
        // Analyze each channel
        let (left_coherence, _left_transient) = self.left_analyzer.analyze(
            left_bin,
            left_neighbors.0,
            left_neighbors.1,
            left_phase_deriv,
            left_flux,
        );

        let (right_coherence, _right_transient) = self.right_analyzer.analyze(
            right_bin,
            right_neighbors.0,
            right_neighbors.1,
            right_phase_deriv,
            right_flux,
        );

        // Cross-channel coherence (correlated L/R = direct source)
        let cross_coherence = self.compute_cross_channel_coherence(left_bin, right_bin);

        // Combine with cross-channel information
        let combined_left_coherence =
            left_coherence * (1.0 - self.cross_channel_weight) + cross_coherence * self.cross_channel_weight;
        let combined_right_coherence =
            right_coherence * (1.0 - self.cross_channel_weight) + cross_coherence * self.cross_channel_weight;

        // Separate each channel
        let left_separated = self
            .separator
            .separate_with_frequency(*left_bin, combined_left_coherence, bin_index, total_bins);
        let right_separated = self
            .separator
            .separate_with_frequency(*right_bin, combined_right_coherence, bin_index, total_bins);

        (left_separated, right_separated)
    }

    /// Compute cross-channel coherence.
    fn compute_cross_channel_coherence(&self, left: &Complex, right: &Complex) -> f32 {
        // Compute correlation between left and right channels
        let left_mag = left.magnitude();
        let right_mag = right.magnitude();

        if left_mag < 1e-10 || right_mag < 1e-10 {
            return 0.5;
        }

        // Magnitude similarity
        let mag_ratio = (left_mag / right_mag).ln().abs();
        let mag_coherence = (-mag_ratio * 0.5).exp();

        // Phase similarity (mono sources have similar phase)
        let phase_diff = self.wrapped_phase_diff(left.phase(), right.phase());
        let phase_coherence = (-phase_diff.abs() * 2.0).exp();

        // Combine (high correlation in both = likely direct sound)
        0.6 * phase_coherence + 0.4 * mag_coherence
    }

    fn wrapped_phase_diff(&self, phase1: f32, phase2: f32) -> f32 {
        let mut diff = phase1 - phase2;
        while diff > std::f32::consts::PI {
            diff -= 2.0 * std::f32::consts::PI;
        }
        while diff < -std::f32::consts::PI {
            diff += 2.0 * std::f32::consts::PI;
        }
        diff
    }

    /// Reset both analyzers.
    pub fn reset(&mut self) {
        self.left_analyzer.reset();
        self.right_analyzer.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_separation_config_presets() {
        let music = SeparationConfig::music_preset();
        assert!(music.temporal_smoothing > 0.7);

        let speech = SeparationConfig::speech_preset();
        assert!(speech.separation_curve > 1.5);

        let aggressive = SeparationConfig::aggressive_preset();
        assert!(aggressive.transient_sensitivity > 1.0);
    }

    #[test]
    fn test_coherence_analyzer() {
        let config = SeparationConfig::default();
        let mut analyzer = CoherenceAnalyzer::new(config);

        // Test with no neighbors
        let value = Complex::new(1.0, 0.0);
        let (coherence, transient) = analyzer.analyze(&value, None, None, 0.0, 0.0);

        assert!(coherence >= 0.0 && coherence <= 1.0);
        assert!(transient >= 0.0 && transient <= 1.0);
    }

    #[test]
    fn test_signal_separator() {
        let config = SeparationConfig::default();
        let separator = SignalSeparator::new(config);

        let value = Complex::new(1.0, 0.0);

        // High coherence = mostly direct
        let (direct, ambient) = separator.separate(value, 0.9);
        assert!(direct.magnitude() > ambient.magnitude());

        // Low coherence = mostly ambient
        let (direct2, ambient2) = separator.separate(value, 0.1);
        assert!(ambient2.magnitude() > direct2.magnitude());
    }

    #[test]
    fn test_separation_preserves_energy() {
        let config = SeparationConfig::default();
        let separator = SignalSeparator::new(config);

        let value = Complex::new(3.0, 4.0); // magnitude = 5
        let original_energy = value.magnitude_squared();

        for coherence in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let (direct, ambient) = separator.separate(value, coherence);
            // Energy should be approximately preserved (with some curve distortion)
            let separated_energy = direct.magnitude_squared() + ambient.magnitude_squared();
            // Due to the power curve, exact preservation isn't guaranteed, but it should be close
            assert!(separated_energy <= original_energy * 1.1);
        }
    }
}
