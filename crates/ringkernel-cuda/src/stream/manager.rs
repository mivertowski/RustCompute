//! Stream manager implementation.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::CudaStream;

use crate::CudaDevice;

use super::config::StreamConfig;

/// Stream identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamId {
    /// Compute stream for kernel execution.
    Compute(usize),
    /// Dedicated transfer stream for host-GPU copies.
    Transfer,
    /// Default stream (stream 0).
    Default,
}

impl StreamId {
    /// Returns the numeric index for this stream.
    #[must_use]
    pub fn index(self) -> usize {
        match self {
            Self::Compute(i) => i,
            Self::Transfer => usize::MAX - 1,
            Self::Default => usize::MAX,
        }
    }
}

/// Errors from stream operations.
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    /// Stream creation failed.
    #[error("Failed to create stream: {0}")]
    CreationFailed(String),

    /// Stream synchronization failed.
    #[error("Stream synchronization failed: {0}")]
    SyncFailed(String),

    /// Event operation failed.
    #[error("Event operation failed: {0}")]
    EventFailed(String),

    /// Invalid stream ID.
    #[error("Invalid stream ID: {id:?}")]
    InvalidStream { id: StreamId },

    /// CUDA not available.
    #[error("CUDA not available")]
    CudaNotAvailable,
}

/// Result type for stream operations.
pub type StreamResult<T> = Result<T, StreamError>;

/// Multi-stream execution manager.
///
/// Manages multiple CUDA streams for concurrent kernel execution and
/// overlapped compute/transfer operations.
pub struct StreamManager {
    /// CUDA device.
    device: CudaDevice,

    /// Compute streams.
    compute_streams: Vec<Arc<CudaStream>>,

    /// Transfer stream (if enabled).
    transfer_stream: Option<Arc<CudaStream>>,

    /// Named events for synchronization timing.
    events: HashMap<String, std::time::Instant>,

    /// Configuration.
    config: StreamConfig,

    /// Whether manager is initialized.
    initialized: bool,
}

impl StreamManager {
    /// Creates a new stream manager.
    pub fn new(device: &CudaDevice, config: StreamConfig) -> StreamResult<Self> {
        let mut compute_streams = Vec::with_capacity(config.num_compute_streams);

        // Create compute streams
        for _i in 0..config.num_compute_streams {
            let stream = device
                .inner()
                .new_stream()
                .map_err(|e| StreamError::CreationFailed(e.to_string()))?;
            compute_streams.push(stream);
        }

        // Create transfer stream if enabled
        let transfer_stream = if config.use_transfer_stream {
            let stream = device
                .inner()
                .new_stream()
                .map_err(|e| StreamError::CreationFailed(e.to_string()))?;
            Some(stream)
        } else {
            None
        };

        tracing::info!(
            num_compute = config.num_compute_streams,
            has_transfer = config.use_transfer_stream,
            "Created stream manager"
        );

        Ok(Self {
            device: device.clone(),
            compute_streams,
            transfer_stream,
            events: HashMap::new(),
            config,
            initialized: true,
        })
    }

    /// Creates a stream manager with default configuration.
    pub fn with_defaults(device: &CudaDevice) -> StreamResult<Self> {
        Self::new(device, StreamConfig::default())
    }

    /// Returns a reference to the CUDA stream for the given stream ID.
    pub fn stream(&self, id: StreamId) -> StreamResult<&Arc<CudaStream>> {
        match id {
            StreamId::Compute(i) => self
                .compute_streams
                .get(i)
                .ok_or(StreamError::InvalidStream { id }),
            StreamId::Transfer => self
                .transfer_stream
                .as_ref()
                .ok_or(StreamError::InvalidStream { id }),
            StreamId::Default => {
                // Return the device's default stream
                Ok(self.device.stream())
            }
        }
    }

    /// Returns the number of compute streams.
    #[must_use]
    pub fn num_compute_streams(&self) -> usize {
        self.compute_streams.len()
    }

    /// Synchronizes a specific stream.
    pub fn sync_stream(&self, id: StreamId) -> StreamResult<()> {
        let stream = self.stream(id)?;
        stream
            .synchronize()
            .map_err(|e| StreamError::SyncFailed(e.to_string()))
    }

    /// Synchronizes all streams.
    pub fn sync_all(&self) -> StreamResult<()> {
        for stream in &self.compute_streams {
            stream
                .synchronize()
                .map_err(|e| StreamError::SyncFailed(e.to_string()))?;
        }

        if let Some(ref transfer) = self.transfer_stream {
            transfer
                .synchronize()
                .map_err(|e| StreamError::SyncFailed(e.to_string()))?;
        }

        Ok(())
    }

    /// Records a timestamp event for profiling.
    ///
    /// Note: This uses software timing. For hardware-accurate timing,
    /// use the profiling module's CudaEvent API.
    pub fn record_event(&mut self, name: &str) {
        self.events
            .insert(name.to_string(), std::time::Instant::now());
    }

    /// Gets elapsed time in milliseconds between two events.
    ///
    /// Returns None if either event is not found.
    pub fn event_elapsed_ms(&self, start_name: &str, end_name: &str) -> Option<f64> {
        let start = self.events.get(start_name)?;
        let end = self.events.get(end_name)?;

        if end >= start {
            Some(end.duration_since(*start).as_secs_f64() * 1000.0)
        } else {
            Some(0.0)
        }
    }

    /// Synchronizes the transfer stream.
    pub fn sync_transfer(&self) -> StreamResult<()> {
        if let Some(ref stream) = self.transfer_stream {
            stream
                .synchronize()
                .map_err(|e| StreamError::SyncFailed(e.to_string()))
        } else {
            Ok(())
        }
    }

    /// Returns whether the manager is initialized.
    #[must_use]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &StreamConfig {
        &self.config
    }

    /// Returns the device reference.
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Assigns algorithms to streams in round-robin fashion.
    ///
    /// Returns the stream ID for the given algorithm index.
    #[must_use]
    pub fn stream_for_workload(&self, workload_index: usize) -> StreamId {
        StreamId::Compute(workload_index % self.compute_streams.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_id_index() {
        assert_eq!(StreamId::Compute(0).index(), 0);
        assert_eq!(StreamId::Compute(3).index(), 3);
        assert_eq!(StreamId::Transfer.index(), usize::MAX - 1);
        assert_eq!(StreamId::Default.index(), usize::MAX);
    }

    #[test]
    fn test_stream_id_equality() {
        assert_eq!(StreamId::Compute(0), StreamId::Compute(0));
        assert_ne!(StreamId::Compute(0), StreamId::Compute(1));
        assert_ne!(StreamId::Compute(0), StreamId::Transfer);
    }
}
