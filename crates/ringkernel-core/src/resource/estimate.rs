//! Memory estimation for resource planning.

/// Memory estimate for a workload.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryEstimate {
    /// Primary memory requirement (main data structures).
    pub primary_bytes: u64,
    /// Auxiliary memory (temporary buffers, working space).
    pub auxiliary_bytes: u64,
    /// Peak memory usage (during execution).
    pub peak_bytes: u64,
    /// Confidence level (0.0-1.0) of this estimate.
    pub confidence: f32,
}

impl Default for MemoryEstimate {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryEstimate {
    /// Creates a new empty memory estimate.
    #[must_use]
    pub fn new() -> Self {
        Self {
            primary_bytes: 0,
            auxiliary_bytes: 0,
            peak_bytes: 0,
            confidence: 1.0,
        }
    }

    /// Creates an estimate with only primary memory.
    #[must_use]
    pub fn primary(bytes: u64) -> Self {
        Self {
            primary_bytes: bytes,
            auxiliary_bytes: 0,
            peak_bytes: bytes,
            confidence: 1.0,
        }
    }

    /// Builder method to set primary memory.
    #[must_use]
    pub fn with_primary(mut self, bytes: u64) -> Self {
        self.primary_bytes = bytes;
        self.update_peak();
        self
    }

    /// Builder method to set auxiliary memory.
    #[must_use]
    pub fn with_auxiliary(mut self, bytes: u64) -> Self {
        self.auxiliary_bytes = bytes;
        self.update_peak();
        self
    }

    /// Builder method to set peak memory.
    #[must_use]
    pub fn with_peak(mut self, bytes: u64) -> Self {
        self.peak_bytes = bytes;
        self
    }

    /// Builder method to set confidence.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Total estimated memory in bytes.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.primary_bytes.saturating_add(self.auxiliary_bytes)
    }

    /// Updates peak to be at least total.
    fn update_peak(&mut self) {
        let total = self.total_bytes();
        if self.peak_bytes < total {
            self.peak_bytes = total;
        }
    }

    /// Returns a human-readable summary.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "primary={}, auxiliary={}, peak={}, confidence={:.0}%",
            format_bytes(self.primary_bytes),
            format_bytes(self.auxiliary_bytes),
            format_bytes(self.peak_bytes),
            self.confidence * 100.0
        )
    }

    /// Combines two estimates (for composite workloads).
    #[must_use]
    pub fn combine(&self, other: &MemoryEstimate) -> Self {
        Self {
            primary_bytes: self.primary_bytes.saturating_add(other.primary_bytes),
            auxiliary_bytes: self.auxiliary_bytes.saturating_add(other.auxiliary_bytes),
            peak_bytes: self.peak_bytes.saturating_add(other.peak_bytes),
            confidence: (self.confidence + other.confidence) / 2.0,
        }
    }

    /// Scales the estimate by a factor.
    #[must_use]
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            primary_bytes: (self.primary_bytes as f64 * factor) as u64,
            auxiliary_bytes: (self.auxiliary_bytes as f64 * factor) as u64,
            peak_bytes: (self.peak_bytes as f64 * factor) as u64,
            confidence: self.confidence,
        }
    }
}

/// Trait for types that can estimate their memory requirements.
pub trait MemoryEstimator: Send + Sync {
    /// Returns an estimate of memory required.
    fn estimate(&self) -> MemoryEstimate;

    /// Returns the name of this estimator (for logging).
    fn name(&self) -> &str;

    /// Returns an estimate scaled for a specific element count.
    fn estimate_for(&self, element_count: usize) -> MemoryEstimate {
        let _ = element_count;
        self.estimate()
    }
}

/// A linear memory estimator (bytes_per_element * count + overhead).
#[derive(Debug, Clone)]
pub struct LinearEstimator {
    /// Name of this estimator.
    pub name: String,
    /// Bytes per element.
    pub bytes_per_element: usize,
    /// Fixed overhead bytes.
    pub fixed_overhead: usize,
    /// Auxiliary bytes per element.
    pub auxiliary_per_element: usize,
}

impl LinearEstimator {
    /// Creates a new linear estimator.
    #[must_use]
    pub fn new(name: impl Into<String>, bytes_per_element: usize) -> Self {
        Self {
            name: name.into(),
            bytes_per_element,
            fixed_overhead: 0,
            auxiliary_per_element: 0,
        }
    }

    /// Builder method to set fixed overhead.
    #[must_use]
    pub fn with_overhead(mut self, bytes: usize) -> Self {
        self.fixed_overhead = bytes;
        self
    }

    /// Builder method to set auxiliary bytes per element.
    #[must_use]
    pub fn with_auxiliary(mut self, bytes_per_element: usize) -> Self {
        self.auxiliary_per_element = bytes_per_element;
        self
    }
}

impl MemoryEstimator for LinearEstimator {
    fn estimate(&self) -> MemoryEstimate {
        MemoryEstimate::new()
            .with_primary(self.fixed_overhead as u64)
            .with_confidence(0.9)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn estimate_for(&self, element_count: usize) -> MemoryEstimate {
        let primary = self.fixed_overhead + (self.bytes_per_element * element_count);
        let auxiliary = self.auxiliary_per_element * element_count;

        MemoryEstimate::new()
            .with_primary(primary as u64)
            .with_auxiliary(auxiliary as u64)
            .with_confidence(0.9)
    }
}

/// Formats bytes as human-readable string.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_estimate_new() {
        let estimate = MemoryEstimate::new();
        assert_eq!(estimate.total_bytes(), 0);
        assert!((estimate.confidence - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_memory_estimate_primary() {
        let estimate = MemoryEstimate::primary(1024);
        assert_eq!(estimate.primary_bytes, 1024);
        assert_eq!(estimate.total_bytes(), 1024);
        assert_eq!(estimate.peak_bytes, 1024);
    }

    #[test]
    fn test_memory_estimate_builder() {
        let estimate = MemoryEstimate::new()
            .with_primary(1024)
            .with_auxiliary(512)
            .with_confidence(0.8);

        assert_eq!(estimate.primary_bytes, 1024);
        assert_eq!(estimate.auxiliary_bytes, 512);
        assert_eq!(estimate.total_bytes(), 1536);
        assert!((estimate.confidence - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_memory_estimate_combine() {
        let a = MemoryEstimate::new().with_primary(1000).with_auxiliary(500);
        let b = MemoryEstimate::new()
            .with_primary(2000)
            .with_auxiliary(1000);

        let combined = a.combine(&b);
        assert_eq!(combined.primary_bytes, 3000);
        assert_eq!(combined.auxiliary_bytes, 1500);
    }

    #[test]
    fn test_memory_estimate_scale() {
        let estimate = MemoryEstimate::new().with_primary(1000);
        let scaled = estimate.scale(2.0);
        assert_eq!(scaled.primary_bytes, 2000);
    }

    #[test]
    fn test_linear_estimator() {
        let estimator = LinearEstimator::new("test", 64)
            .with_overhead(1024)
            .with_auxiliary(16);

        let estimate = estimator.estimate_for(100);
        assert_eq!(estimate.primary_bytes, 1024 + 64 * 100);
        assert_eq!(estimate.auxiliary_bytes, 16 * 100);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1_500_000), "1.43 MB");
    }
}
