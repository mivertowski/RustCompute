//! Dry/wet mixer with gain control.
//!
//! This module provides mixing functionality to combine the separated
//! direct and ambient signals with user-controllable parameters.

use crate::messages::{Complex, SeparatedBin};

/// Configuration for the mixer.
#[derive(Debug, Clone)]
pub struct MixerConfig {
    /// Dry/wet mix (0.0 = all direct, 1.0 = all ambience).
    pub dry_wet: f32,
    /// Direct signal gain (1.0 = unity).
    pub direct_gain: f32,
    /// Ambience signal gain (1.0 = unity).
    pub ambience_gain: f32,
    /// Output gain (applied to final mix).
    pub output_gain: f32,
    /// Soft clip threshold (None = no limiting).
    pub soft_clip_threshold: Option<f32>,
}

impl Default for MixerConfig {
    fn default() -> Self {
        Self {
            dry_wet: 0.5,
            direct_gain: 1.0,
            ambience_gain: 1.0,
            output_gain: 1.0,
            soft_clip_threshold: Some(0.95),
        }
    }
}

impl MixerConfig {
    /// Create a new mixer configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the dry/wet mix.
    pub fn with_dry_wet(mut self, dry_wet: f32) -> Self {
        self.dry_wet = dry_wet.clamp(0.0, 1.0);
        self
    }

    /// Set the direct gain.
    pub fn with_direct_gain(mut self, gain: f32) -> Self {
        self.direct_gain = gain.max(0.0);
        self
    }

    /// Set the ambience gain.
    pub fn with_ambience_gain(mut self, gain: f32) -> Self {
        self.ambience_gain = gain.max(0.0);
        self
    }

    /// Set the output gain.
    pub fn with_output_gain(mut self, gain: f32) -> Self {
        self.output_gain = gain.max(0.0);
        self
    }

    /// Set soft clip threshold.
    pub fn with_soft_clip(mut self, threshold: Option<f32>) -> Self {
        self.soft_clip_threshold = threshold.map(|t| t.clamp(0.1, 1.0));
        self
    }

    /// Preset for direct-only output.
    pub fn direct_only() -> Self {
        Self {
            dry_wet: 0.0,
            direct_gain: 1.0,
            ambience_gain: 0.0,
            output_gain: 1.0,
            soft_clip_threshold: Some(0.95),
        }
    }

    /// Preset for ambience-only output.
    pub fn ambience_only() -> Self {
        Self {
            dry_wet: 1.0,
            direct_gain: 0.0,
            ambience_gain: 1.0,
            output_gain: 1.0,
            soft_clip_threshold: Some(0.95),
        }
    }

    /// Preset for balanced mix with boost.
    pub fn balanced_with_boost(boost_db: f32) -> Self {
        let linear_gain = 10.0_f32.powf(boost_db / 20.0);
        Self {
            dry_wet: 0.5,
            direct_gain: 1.0,
            ambience_gain: 1.0,
            output_gain: linear_gain,
            soft_clip_threshold: Some(0.95),
        }
    }
}

/// Dry/wet mixer for separated signals.
pub struct DryWetMixer {
    config: MixerConfig,
    /// Peak level tracking for direct signal.
    direct_peak: f32,
    /// Peak level tracking for ambience signal.
    ambience_peak: f32,
    /// Peak level tracking for output.
    output_peak: f32,
}

impl DryWetMixer {
    /// Create a new mixer with default configuration.
    pub fn new() -> Self {
        Self::with_config(MixerConfig::default())
    }

    /// Create a new mixer with specific configuration.
    pub fn with_config(config: MixerConfig) -> Self {
        Self {
            config,
            direct_peak: 0.0,
            ambience_peak: 0.0,
            output_peak: 0.0,
        }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &MixerConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: MixerConfig) {
        self.config = config;
    }

    /// Set dry/wet mix (0.0 = all direct, 1.0 = all ambience).
    pub fn set_dry_wet(&mut self, dry_wet: f32) {
        self.config.dry_wet = dry_wet.clamp(0.0, 1.0);
    }

    /// Set direct signal gain.
    pub fn set_direct_gain(&mut self, gain: f32) {
        self.config.direct_gain = gain.max(0.0);
    }

    /// Set ambience signal gain.
    pub fn set_ambience_gain(&mut self, gain: f32) {
        self.config.ambience_gain = gain.max(0.0);
    }

    /// Set output gain.
    pub fn set_output_gain(&mut self, gain: f32) {
        self.config.output_gain = gain.max(0.0);
    }

    /// Set output gain in dB.
    pub fn set_output_gain_db(&mut self, gain_db: f32) {
        self.config.output_gain = 10.0_f32.powf(gain_db / 20.0);
    }

    /// Mix a separated bin and return the combined value.
    pub fn mix_bin(&mut self, bin: &SeparatedBin) -> Complex {
        // Apply gains to each component
        let direct = bin.direct.scale(self.config.direct_gain);
        let ambience = bin.ambience.scale(self.config.ambience_gain);

        // Track peaks
        self.direct_peak = self.direct_peak.max(direct.magnitude());
        self.ambience_peak = self.ambience_peak.max(ambience.magnitude());

        // Mix based on dry/wet
        let dry_amount = 1.0 - self.config.dry_wet;
        let wet_amount = self.config.dry_wet;

        let mixed = Complex {
            re: direct.re * dry_amount + ambience.re * wet_amount,
            im: direct.im * dry_amount + ambience.im * wet_amount,
        };

        // Apply output gain
        let output = mixed.scale(self.config.output_gain);

        // Track output peak
        self.output_peak = self.output_peak.max(output.magnitude());

        // Apply soft clipping if enabled
        if let Some(threshold) = self.config.soft_clip_threshold {
            self.soft_clip(output, threshold)
        } else {
            output
        }
    }

    /// Mix multiple separated bins.
    pub fn mix_frame(&mut self, bins: &[SeparatedBin]) -> Vec<Complex> {
        bins.iter().map(|bin| self.mix_bin(bin)).collect()
    }

    /// Get only the direct component (with gain).
    pub fn direct_only(&self, bin: &SeparatedBin) -> Complex {
        bin.direct.scale(self.config.direct_gain * self.config.output_gain)
    }

    /// Get only the ambience component (with gain).
    pub fn ambience_only(&self, bin: &SeparatedBin) -> Complex {
        bin.ambience.scale(self.config.ambience_gain * self.config.output_gain)
    }

    /// Extract direct bins from separated data.
    pub fn extract_direct(&self, bins: &[SeparatedBin]) -> Vec<Complex> {
        bins.iter().map(|bin| self.direct_only(bin)).collect()
    }

    /// Extract ambience bins from separated data.
    pub fn extract_ambience(&self, bins: &[SeparatedBin]) -> Vec<Complex> {
        bins.iter().map(|bin| self.ambience_only(bin)).collect()
    }

    /// Apply soft clipping to a complex value.
    fn soft_clip(&self, value: Complex, threshold: f32) -> Complex {
        let magnitude = value.magnitude();

        if magnitude <= threshold {
            return value;
        }

        // Soft knee compression above threshold
        let overshoot = magnitude - threshold;
        let compressed = threshold + overshoot.tanh() * (1.0 - threshold);

        // Scale to new magnitude while preserving phase
        if magnitude > 1e-10 {
            value.scale(compressed / magnitude)
        } else {
            value
        }
    }

    /// Get peak levels (direct, ambience, output).
    pub fn peak_levels(&self) -> (f32, f32, f32) {
        (self.direct_peak, self.ambience_peak, self.output_peak)
    }

    /// Get peak levels in dB.
    pub fn peak_levels_db(&self) -> (f32, f32, f32) {
        let to_db = |level: f32| {
            if level > 1e-10 {
                20.0 * level.log10()
            } else {
                -200.0
            }
        };

        (
            to_db(self.direct_peak),
            to_db(self.ambience_peak),
            to_db(self.output_peak),
        )
    }

    /// Reset peak tracking.
    pub fn reset_peaks(&mut self) {
        self.direct_peak = 0.0;
        self.ambience_peak = 0.0;
        self.output_peak = 0.0;
    }
}

impl Default for DryWetMixer {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of mixing a frame.
#[derive(Debug, Clone)]
pub struct MixedFrame {
    /// Mixed frequency bins.
    pub bins: Vec<Complex>,
    /// Direct component only.
    pub direct_bins: Vec<Complex>,
    /// Ambience component only.
    pub ambience_bins: Vec<Complex>,
    /// Frame ID.
    pub frame_id: u64,
}

impl MixedFrame {
    /// Create a new mixed frame.
    pub fn new(bins: Vec<Complex>, direct: Vec<Complex>, ambience: Vec<Complex>, frame_id: u64) -> Self {
        Self {
            bins,
            direct_bins: direct,
            ambience_bins: ambience,
            frame_id,
        }
    }
}

/// Full frame mixer that produces all output variants.
pub struct FrameMixer {
    mixer: DryWetMixer,
}

impl FrameMixer {
    /// Create a new frame mixer.
    pub fn new(config: MixerConfig) -> Self {
        Self {
            mixer: DryWetMixer::with_config(config),
        }
    }

    /// Get the underlying mixer.
    pub fn mixer(&self) -> &DryWetMixer {
        &self.mixer
    }

    /// Get mutable reference to the mixer.
    pub fn mixer_mut(&mut self) -> &mut DryWetMixer {
        &mut self.mixer
    }

    /// Process a frame of separated bins.
    pub fn process(&mut self, bins: &[SeparatedBin]) -> MixedFrame {
        let frame_id = bins.first().map(|b| b.frame_id).unwrap_or(0);

        let mixed = self.mixer.mix_frame(bins);
        let direct = self.mixer.extract_direct(bins);
        let ambience = self.mixer.extract_ambience(bins);

        MixedFrame::new(mixed, direct, ambience, frame_id)
    }

    /// Update the dry/wet mix.
    pub fn set_dry_wet(&mut self, dry_wet: f32) {
        self.mixer.set_dry_wet(dry_wet);
    }

    /// Update the output gain in dB.
    pub fn set_gain_db(&mut self, gain_db: f32) {
        self.mixer.set_output_gain_db(gain_db);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixer_config() {
        let config = MixerConfig::new()
            .with_dry_wet(0.3)
            .with_direct_gain(1.2)
            .with_ambience_gain(0.8);

        assert!((config.dry_wet - 0.3).abs() < 1e-6);
        assert!((config.direct_gain - 1.2).abs() < 1e-6);
    }

    #[test]
    fn test_dry_wet_mix() {
        // Disable soft clipping for predictable test results
        let config = MixerConfig::new().with_soft_clip(None);
        let mut mixer = DryWetMixer::with_config(config);

        let bin = SeparatedBin::new(
            0,
            0,
            Complex::new(1.0, 0.0),  // Direct
            Complex::new(0.5, 0.0),  // Ambience
            0.6,
            0.0,
        );

        // All direct (dry = 1.0)
        mixer.set_dry_wet(0.0);
        let result = mixer.mix_bin(&bin);
        assert!((result.re - 1.0).abs() < 1e-6);

        // All ambience (wet = 1.0)
        mixer.set_dry_wet(1.0);
        let result = mixer.mix_bin(&bin);
        assert!((result.re - 0.5).abs() < 1e-6);

        // 50/50 mix
        mixer.set_dry_wet(0.5);
        let result = mixer.mix_bin(&bin);
        assert!((result.re - 0.75).abs() < 1e-6); // 0.5 * 1.0 + 0.5 * 0.5
    }

    #[test]
    fn test_gain_application() {
        let config = MixerConfig::new()
            .with_dry_wet(0.0) // All direct
            .with_direct_gain(2.0)
            .with_output_gain(0.5)
            .with_soft_clip(None); // Disable soft clipping

        let mut mixer = DryWetMixer::with_config(config);

        let bin = SeparatedBin::new(
            0,
            0,
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            1.0,
            0.0,
        );

        let result = mixer.mix_bin(&bin);
        // 1.0 * 2.0 (direct gain) * 0.5 (output gain) = 1.0
        assert!((result.re - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_soft_clipping() {
        let config = MixerConfig::new()
            .with_output_gain(10.0) // High gain to trigger clipping
            .with_soft_clip(Some(0.9));

        let mut mixer = DryWetMixer::with_config(config);

        let bin = SeparatedBin::new(
            0,
            0,
            Complex::new(0.5, 0.0),
            Complex::new(0.0, 0.0),
            1.0,
            0.0,
        );

        let result = mixer.mix_bin(&bin);
        // Should be soft-clipped below 1.0
        assert!(result.magnitude() < 1.0);
    }

    #[test]
    fn test_peak_tracking() {
        let mut mixer = DryWetMixer::new();

        let bin = SeparatedBin::new(
            0,
            0,
            Complex::new(0.8, 0.0),
            Complex::new(0.3, 0.0),
            0.5,
            0.0,
        );

        mixer.mix_bin(&bin);
        let (direct, ambience, _output) = mixer.peak_levels();

        assert!((direct - 0.8).abs() < 1e-6);
        assert!((ambience - 0.3).abs() < 1e-6);

        mixer.reset_peaks();
        let (direct, ambience, output) = mixer.peak_levels();
        assert_eq!(direct, 0.0);
        assert_eq!(ambience, 0.0);
        assert_eq!(output, 0.0);
    }
}
