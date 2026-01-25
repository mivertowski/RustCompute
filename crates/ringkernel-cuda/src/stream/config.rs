//! Stream configuration.

/// Configuration for the stream manager.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Number of compute streams.
    pub num_compute_streams: usize,
    /// Whether to use a dedicated transfer stream.
    pub use_transfer_stream: bool,
    /// Stream priority (0 = default, negative = higher priority).
    pub compute_priority: i32,
    /// Transfer stream priority.
    pub transfer_priority: i32,
    /// Whether to enable stream capturing for CUDA graphs.
    pub enable_graph_capture: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            num_compute_streams: 4,
            use_transfer_stream: true,
            compute_priority: 0,
            transfer_priority: -1, // Higher priority for transfers
            enable_graph_capture: false,
        }
    }
}

impl StreamConfig {
    /// Creates a minimal configuration (single compute stream).
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            num_compute_streams: 1,
            use_transfer_stream: false,
            compute_priority: 0,
            transfer_priority: 0,
            enable_graph_capture: false,
        }
    }

    /// Creates a performance configuration (4 compute streams + transfer).
    #[must_use]
    pub fn performance() -> Self {
        Self {
            num_compute_streams: 4,
            use_transfer_stream: true,
            compute_priority: 0,
            transfer_priority: -1,
            enable_graph_capture: true,
        }
    }

    /// Creates a configuration optimized for simulation workloads.
    #[must_use]
    pub fn for_simulation() -> Self {
        Self {
            num_compute_streams: 2,
            use_transfer_stream: true,
            compute_priority: 0,
            transfer_priority: -1,
            enable_graph_capture: false,
        }
    }
}

/// Builder for StreamConfig.
#[derive(Debug, Default)]
pub struct StreamConfigBuilder {
    config: StreamConfig,
}

impl StreamConfigBuilder {
    /// Creates a new builder with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a minimal configuration builder.
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            config: StreamConfig::minimal(),
        }
    }

    /// Creates a performance configuration builder.
    #[must_use]
    pub fn performance() -> Self {
        Self {
            config: StreamConfig::performance(),
        }
    }

    /// Sets the number of compute streams.
    #[must_use]
    pub fn with_compute_streams(mut self, count: usize) -> Self {
        self.config.num_compute_streams = count;
        self
    }

    /// Enables or disables the transfer stream.
    #[must_use]
    pub fn with_transfer_stream(mut self, enabled: bool) -> Self {
        self.config.use_transfer_stream = enabled;
        self
    }

    /// Sets compute stream priority.
    #[must_use]
    pub fn with_compute_priority(mut self, priority: i32) -> Self {
        self.config.compute_priority = priority;
        self
    }

    /// Sets transfer stream priority.
    #[must_use]
    pub fn with_transfer_priority(mut self, priority: i32) -> Self {
        self.config.transfer_priority = priority;
        self
    }

    /// Enables or disables CUDA graph capture.
    #[must_use]
    pub fn with_graph_capture(mut self, enabled: bool) -> Self {
        self.config.enable_graph_capture = enabled;
        self
    }

    /// Builds the configuration.
    #[must_use]
    pub fn build(self) -> StreamConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_config_defaults() {
        let config = StreamConfig::default();
        assert_eq!(config.num_compute_streams, 4);
        assert!(config.use_transfer_stream);
        assert_eq!(config.compute_priority, 0);
        assert_eq!(config.transfer_priority, -1);
    }

    #[test]
    fn test_stream_config_minimal() {
        let config = StreamConfig::minimal();
        assert_eq!(config.num_compute_streams, 1);
        assert!(!config.use_transfer_stream);
    }

    #[test]
    fn test_stream_config_performance() {
        let config = StreamConfig::performance();
        assert_eq!(config.num_compute_streams, 4);
        assert!(config.use_transfer_stream);
        assert!(config.enable_graph_capture);
    }

    #[test]
    fn test_stream_config_builder() {
        let config = StreamConfigBuilder::new()
            .with_compute_streams(8)
            .with_transfer_stream(true)
            .with_compute_priority(-1)
            .with_graph_capture(true)
            .build();

        assert_eq!(config.num_compute_streams, 8);
        assert!(config.use_transfer_stream);
        assert_eq!(config.compute_priority, -1);
        assert!(config.enable_graph_capture);
    }
}
