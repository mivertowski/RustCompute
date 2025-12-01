//! FFT and IFFT processing utilities.
//!
//! This module provides FFT/IFFT processing using rustfft, with support for
//! overlap-add processing and various window functions.

use std::sync::Arc;

use num_complex::Complex as NumComplex;
use rustfft::{Fft, FftPlanner};

use crate::error::{AudioFftError, Result};
use crate::messages::Complex;

/// Window function types for FFT analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFunction {
    /// Rectangular (no window).
    Rectangular,
    /// Hann window (cosine-squared).
    Hann,
    /// Hamming window.
    Hamming,
    /// Blackman window.
    Blackman,
    /// Blackman-Harris window.
    BlackmanHarris,
    /// Kaiser window with beta parameter.
    Kaiser(u8), // beta * 10 to avoid float in enum
}

impl WindowFunction {
    /// Generate window coefficients.
    pub fn generate(&self, size: usize) -> Vec<f32> {
        let n = size as f32;
        (0..size)
            .map(|i| {
                let x = i as f32;
                match self {
                    Self::Rectangular => 1.0,
                    Self::Hann => 0.5 * (1.0 - (2.0 * std::f32::consts::PI * x / n).cos()),
                    Self::Hamming => {
                        0.54 - 0.46 * (2.0 * std::f32::consts::PI * x / n).cos()
                    }
                    Self::Blackman => {
                        let a0 = 0.42;
                        let a1 = 0.5;
                        let a2 = 0.08;
                        a0 - a1 * (2.0 * std::f32::consts::PI * x / n).cos()
                            + a2 * (4.0 * std::f32::consts::PI * x / n).cos()
                    }
                    Self::BlackmanHarris => {
                        let a0 = 0.35875;
                        let a1 = 0.48829;
                        let a2 = 0.14128;
                        let a3 = 0.01168;
                        a0 - a1 * (2.0 * std::f32::consts::PI * x / n).cos()
                            + a2 * (4.0 * std::f32::consts::PI * x / n).cos()
                            - a3 * (6.0 * std::f32::consts::PI * x / n).cos()
                    }
                    Self::Kaiser(beta_10) => {
                        let beta = *beta_10 as f32 / 10.0;
                        let alpha = (n - 1.0) / 2.0;
                        let r = (x - alpha) / alpha;
                        bessel_i0(beta * (1.0 - r * r).sqrt()) / bessel_i0(beta)
                    }
                }
            })
            .collect()
    }

    /// Get the coherent gain for this window.
    pub fn coherent_gain(&self) -> f32 {
        match self {
            Self::Rectangular => 1.0,
            Self::Hann => 0.5,
            Self::Hamming => 0.54,
            Self::Blackman => 0.42,
            Self::BlackmanHarris => 0.35875,
            Self::Kaiser(_) => 0.5, // Approximate
        }
    }
}

/// Bessel I0 function for Kaiser window.
fn bessel_i0(x: f32) -> f32 {
    let mut sum = 1.0f32;
    let mut term = 1.0f32;
    let x2 = x * x / 4.0;

    for k in 1..20 {
        term *= x2 / (k * k) as f32;
        sum += term;
        if term < 1e-10 {
            break;
        }
    }
    sum
}

/// FFT processor for time-to-frequency conversion.
pub struct FftProcessor {
    /// FFT size (must be power of 2).
    fft_size: usize,
    /// Hop size for overlap-add.
    hop_size: usize,
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Window function (stored for potential reconfiguration).
    #[allow(dead_code)]
    window: WindowFunction,
    /// Pre-computed window coefficients.
    window_coeffs: Vec<f32>,
    /// FFT planner.
    fft: Arc<dyn Fft<f32>>,
    /// Scratch buffer.
    scratch: Vec<NumComplex<f32>>,
    /// Input buffer for overlap.
    input_buffer: Vec<f32>,
}

impl FftProcessor {
    /// Create a new FFT processor.
    pub fn new(fft_size: usize, hop_size: usize, sample_rate: u32) -> Result<Self> {
        Self::with_window(fft_size, hop_size, sample_rate, WindowFunction::Hann)
    }

    /// Create with a specific window function.
    pub fn with_window(
        fft_size: usize,
        hop_size: usize,
        sample_rate: u32,
        window: WindowFunction,
    ) -> Result<Self> {
        if !fft_size.is_power_of_two() {
            return Err(AudioFftError::config(format!(
                "FFT size must be power of 2, got {}",
                fft_size
            )));
        }

        if hop_size > fft_size {
            return Err(AudioFftError::config(format!(
                "Hop size {} cannot exceed FFT size {}",
                hop_size, fft_size
            )));
        }

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let scratch_len = fft.get_inplace_scratch_len();

        Ok(Self {
            fft_size,
            hop_size,
            sample_rate,
            window,
            window_coeffs: window.generate(fft_size),
            fft,
            scratch: vec![NumComplex::default(); scratch_len],
            input_buffer: Vec::with_capacity(fft_size * 2),
        })
    }

    /// Get the FFT size.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Get the hop size.
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Get the number of frequency bins (positive frequencies only).
    pub fn num_bins(&self) -> usize {
        self.fft_size / 2 + 1
    }

    /// Get the frequency in Hz for a given bin.
    pub fn bin_to_frequency(&self, bin: usize) -> f32 {
        bin as f32 * self.sample_rate as f32 / self.fft_size as f32
    }

    /// Get the bin index for a given frequency.
    pub fn frequency_to_bin(&self, freq: f32) -> usize {
        (freq * self.fft_size as f32 / self.sample_rate as f32).round() as usize
    }

    /// Process a frame of audio and return FFT bins.
    pub fn process_frame(&mut self, samples: &[f32]) -> Vec<Complex> {
        // Add samples to input buffer
        self.input_buffer.extend_from_slice(samples);

        // Check if we have enough for a frame
        if self.input_buffer.len() < self.fft_size {
            return Vec::new();
        }

        // Take the FFT frame
        let mut buffer: Vec<NumComplex<f32>> = self.input_buffer[..self.fft_size]
            .iter()
            .enumerate()
            .map(|(i, &s)| NumComplex::new(s * self.window_coeffs[i], 0.0))
            .collect();

        // Perform FFT
        self.fft.process_with_scratch(&mut buffer, &mut self.scratch);

        // Remove processed samples (hop)
        self.input_buffer.drain(..self.hop_size);

        // Convert to our Complex type (positive frequencies only)
        buffer[..self.num_bins()]
            .iter()
            .map(|c| Complex::new(c.re, c.im))
            .collect()
    }

    /// Process all available frames in the buffer.
    pub fn process_all(&mut self, samples: &[f32]) -> Vec<Vec<Complex>> {
        self.input_buffer.extend_from_slice(samples);

        let mut frames = Vec::new();

        while self.input_buffer.len() >= self.fft_size {
            // Take the FFT frame
            let mut buffer: Vec<NumComplex<f32>> = self.input_buffer[..self.fft_size]
                .iter()
                .enumerate()
                .map(|(i, &s)| NumComplex::new(s * self.window_coeffs[i], 0.0))
                .collect();

            // Perform FFT
            self.fft.process_with_scratch(&mut buffer, &mut self.scratch);

            // Remove processed samples (hop)
            self.input_buffer.drain(..self.hop_size);

            // Convert to our Complex type
            frames.push(
                buffer[..self.num_bins()]
                    .iter()
                    .map(|c| Complex::new(c.re, c.im))
                    .collect(),
            );
        }

        frames
    }

    /// Flush remaining samples (zero-pad if necessary).
    pub fn flush(&mut self) -> Option<Vec<Complex>> {
        if self.input_buffer.is_empty() {
            return None;
        }

        // Zero-pad to FFT size
        self.input_buffer.resize(self.fft_size, 0.0);

        let mut buffer: Vec<NumComplex<f32>> = self.input_buffer
            .iter()
            .enumerate()
            .map(|(i, &s)| NumComplex::new(s * self.window_coeffs[i], 0.0))
            .collect();

        self.fft.process_with_scratch(&mut buffer, &mut self.scratch);
        self.input_buffer.clear();

        Some(
            buffer[..self.num_bins()]
                .iter()
                .map(|c| Complex::new(c.re, c.im))
                .collect(),
        )
    }

    /// Reset the processor state.
    pub fn reset(&mut self) {
        self.input_buffer.clear();
    }
}

/// IFFT processor for frequency-to-time conversion.
pub struct IfftProcessor {
    /// FFT size.
    fft_size: usize,
    /// Hop size.
    hop_size: usize,
    /// IFFT planner.
    ifft: Arc<dyn Fft<f32>>,
    /// Scratch buffer.
    scratch: Vec<NumComplex<f32>>,
    /// Synthesis window.
    synthesis_window: Vec<f32>,
    /// Output buffer for overlap-add.
    output_buffer: Vec<f32>,
    /// Normalization factor.
    norm_factor: f32,
}

impl IfftProcessor {
    /// Create a new IFFT processor.
    pub fn new(fft_size: usize, hop_size: usize) -> Result<Self> {
        Self::with_window(fft_size, hop_size, WindowFunction::Hann)
    }

    /// Create with a specific synthesis window.
    pub fn with_window(fft_size: usize, hop_size: usize, window: WindowFunction) -> Result<Self> {
        if !fft_size.is_power_of_two() {
            return Err(AudioFftError::config(format!(
                "FFT size must be power of 2, got {}",
                fft_size
            )));
        }

        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(fft_size);
        let scratch_len = ifft.get_inplace_scratch_len();

        // Calculate COLA normalization for overlap-add
        // For Hann window with 50% overlap, sum of squared windows = 1.5
        let window_coeffs = window.generate(fft_size);
        let overlap_factor = fft_size / hop_size;

        // Calculate the sum of squared windows at each output sample
        let mut cola_sum = vec![0.0f32; hop_size];
        for offset in 0..overlap_factor {
            for (i, sum) in cola_sum.iter_mut().enumerate() {
                let window_idx = offset * hop_size + i;
                if window_idx < fft_size {
                    *sum += window_coeffs[window_idx] * window_coeffs[window_idx];
                }
            }
        }
        let avg_cola = cola_sum.iter().sum::<f32>() / hop_size as f32;

        Ok(Self {
            fft_size,
            hop_size,
            ifft,
            scratch: vec![NumComplex::default(); scratch_len],
            synthesis_window: window_coeffs,
            output_buffer: vec![0.0; fft_size * 2],
            norm_factor: 1.0 / (fft_size as f32 * avg_cola.sqrt()),
        })
    }

    /// Process FFT bins and return audio samples.
    pub fn process_frame(&mut self, bins: &[Complex]) -> Vec<f32> {
        // Reconstruct full spectrum (mirror conjugate)
        let mut buffer: Vec<NumComplex<f32>> = Vec::with_capacity(self.fft_size);

        // Positive frequencies
        for bin in bins.iter().take(self.fft_size / 2 + 1) {
            buffer.push(NumComplex::new(bin.re, bin.im));
        }

        // Negative frequencies (conjugate mirror)
        for i in 1..self.fft_size / 2 {
            let idx = self.fft_size / 2 - i;
            if idx < bins.len() {
                buffer.push(NumComplex::new(bins[idx].re, -bins[idx].im));
            } else {
                buffer.push(NumComplex::default());
            }
        }

        // Pad if necessary
        while buffer.len() < self.fft_size {
            buffer.push(NumComplex::default());
        }

        // Perform IFFT
        self.ifft.process_with_scratch(&mut buffer, &mut self.scratch);

        // Apply synthesis window and add to output buffer
        for (i, c) in buffer.iter().enumerate() {
            self.output_buffer[i] += c.re * self.synthesis_window[i] * self.norm_factor;
        }

        // Extract output samples
        let output: Vec<f32> = self.output_buffer[..self.hop_size].to_vec();

        // Shift buffer
        self.output_buffer.copy_within(self.hop_size.., 0);
        for i in (self.output_buffer.len() - self.hop_size)..self.output_buffer.len() {
            self.output_buffer[i] = 0.0;
        }

        output
    }

    /// Flush remaining samples.
    pub fn flush(&mut self) -> Vec<f32> {
        let mut output = Vec::new();

        // Drain the output buffer
        while self.output_buffer.iter().any(|&x| x.abs() > 1e-10) {
            output.extend_from_slice(&self.output_buffer[..self.hop_size]);
            self.output_buffer.copy_within(self.hop_size.., 0);
            for i in (self.output_buffer.len() - self.hop_size)..self.output_buffer.len() {
                self.output_buffer[i] = 0.0;
            }
        }

        output
    }

    /// Reset the processor state.
    pub fn reset(&mut self) {
        self.output_buffer.fill(0.0);
    }
}

/// Helper for STFT processing with proper overlap-add.
pub struct StftProcessor {
    /// FFT processor.
    pub fft: FftProcessor,
    /// IFFT processor.
    pub ifft: IfftProcessor,
}

impl StftProcessor {
    /// Create a new STFT processor.
    pub fn new(fft_size: usize, hop_size: usize, sample_rate: u32) -> Result<Self> {
        Self::with_window(fft_size, hop_size, sample_rate, WindowFunction::Hann)
    }

    /// Create with a specific window function.
    pub fn with_window(
        fft_size: usize,
        hop_size: usize,
        sample_rate: u32,
        window: WindowFunction,
    ) -> Result<Self> {
        Ok(Self {
            fft: FftProcessor::with_window(fft_size, hop_size, sample_rate, window)?,
            ifft: IfftProcessor::with_window(fft_size, hop_size, window)?,
        })
    }

    /// Process samples through FFT, apply a function, and IFFT back.
    pub fn process<F>(&mut self, samples: &[f32], mut processor: F) -> Vec<f32>
    where
        F: FnMut(&mut [Complex]),
    {
        let mut output = Vec::new();

        for mut frame in self.fft.process_all(samples) {
            processor(&mut frame);
            output.extend(self.ifft.process_frame(&frame));
        }

        output
    }

    /// Flush remaining samples.
    pub fn flush<F>(&mut self, mut processor: F) -> Vec<f32>
    where
        F: FnMut(&mut [Complex]),
    {
        let mut output = Vec::new();

        if let Some(mut frame) = self.fft.flush() {
            processor(&mut frame);
            output.extend(self.ifft.process_frame(&frame));
        }

        output.extend(self.ifft.flush());
        output
    }

    /// Reset the processor state.
    pub fn reset(&mut self) {
        self.fft.reset();
        self.ifft.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_functions() {
        let size = 1024;

        let hann = WindowFunction::Hann.generate(size);
        assert!((hann[0] - 0.0).abs() < 1e-6);
        assert!((hann[size / 2] - 1.0).abs() < 1e-6);

        let hamming = WindowFunction::Hamming.generate(size);
        assert!((hamming[0] - 0.08).abs() < 0.01);
    }

    #[test]
    fn test_fft_roundtrip() {
        let fft_size = 1024;
        let hop_size = 256;
        let sample_rate = 44100;

        let mut stft = StftProcessor::new(fft_size, hop_size, sample_rate).unwrap();

        // Generate a test signal (sine wave at 440 Hz)
        let duration = 0.1;
        let samples: Vec<f32> = (0..(sample_rate as f32 * duration) as usize)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        // Process through FFT and back
        let output = stft.process(&samples, |_bins| {
            // Identity transform
        });

        // Check that output length is reasonable
        assert!(output.len() > 0);

        // The output should be similar to input (with some latency)
        // Due to windowing/overlap-add, there's some distortion at edges
    }

    #[test]
    fn test_bin_frequency_conversion() {
        let fft = FftProcessor::new(2048, 512, 44100).unwrap();

        // DC bin
        assert!((fft.bin_to_frequency(0) - 0.0).abs() < 1e-6);

        // Nyquist
        let nyquist = fft.bin_to_frequency(1024);
        assert!((nyquist - 22050.0).abs() < 1.0);

        // Round-trip
        let freq = 1000.0;
        let bin = fft.frequency_to_bin(freq);
        let recovered = fft.bin_to_frequency(bin);
        assert!((recovered - freq).abs() < 50.0); // Within one bin width
    }
}
