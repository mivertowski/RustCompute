//! Actor-based ring kernel backend for streaming transaction monitoring.
//!
//! This backend uses persistent GPU kernels that continuously process
//! transaction streams with HLC timestamps and K2K messaging support.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                         GPU Device                               │
//! │  ┌──────────────────────────────────────────────────────────┐  │
//! │  │              Ring Kernel (Persistent)                     │  │
//! │  │  ┌─────────────────────────────────────────────────────┐ │  │
//! │  │  │                 Control Block                        │ │  │
//! │  │  │  is_active | should_terminate | messages_processed   │ │  │
//! │  │  │  input_head/tail | output_head/tail | hlc_state      │ │  │
//! │  │  └─────────────────────────────────────────────────────┘ │  │
//! │  │                                                           │  │
//! │  │  ┌──────────────┐     ┌──────────────┐                   │  │
//! │  │  │ Input Queue  │────>│ Thread Block │────>│ Output Queue│  │
//! │  │  │ (Lock-free)  │     │  (Handler)   │     │ (Lock-free) │  │
//! │  │  └──────────────┘     └──────────────┘     └─────────────┘  │
//! │  │                              │                              │  │
//! │  │                              v                              │  │
//! │  │                     ┌──────────────┐                        │  │
//! │  │                     │  K2K Routes  │───> Other Kernels      │  │
//! │  │                     └──────────────┘                        │  │
//! │  └──────────────────────────────────────────────────────────┘  │
//! │                                                                  │
//! │  ┌──────────────────┐     ┌──────────────────┐                  │
//! │  │ Alert Aggregator │     │ Pattern Detector │                  │
//! │  │   Ring Kernel    │<--->│   Ring Kernel    │                  │
//! │  └──────────────────┘     └──────────────────┘                  │
//! └─────────────────────────────────────────────────────────────────┘
//!          ^                                      |
//!          |         Host-GPU Communication      v
//! ┌────────┴───────────────────────────────────────────┐
//! │  Host Application                                   │
//! │  - Enqueue transactions to input queue              │
//! │  - Dequeue alerts from output queue                 │
//! │  - Control kernel lifecycle                         │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! ## Features
//!
//! - **Persistent Execution**: Kernel stays resident on GPU, avoiding launch overhead
//! - **HLC Timestamps**: Hybrid Logical Clocks for causal ordering across kernels
//! - **K2K Messaging**: Direct kernel-to-kernel communication without host
//! - **Lock-free Queues**: High-performance message passing
//!
//! ## Use Cases
//!
//! - Real-time streaming with low latency requirements
//! - Multi-stage processing pipelines
//! - Complex inter-kernel coordination

use bytemuck::Zeroable;

/// Configuration for ring kernel backend.
#[derive(Debug, Clone)]
pub struct RingKernelConfig {
    /// Number of threads per block.
    pub block_size: u32,
    /// Input queue capacity (must be power of 2).
    pub queue_capacity: u32,
    /// Enable Hybrid Logical Clocks.
    pub enable_hlc: bool,
    /// Enable Kernel-to-Kernel messaging.
    pub enable_k2k: bool,
    /// Maximum K2K connections.
    pub max_k2k_routes: u32,
    /// Polling interval in nanoseconds when queue is empty.
    pub idle_poll_ns: u32,
}

impl Default for RingKernelConfig {
    fn default() -> Self {
        Self {
            block_size: 128,
            queue_capacity: 4096,
            enable_hlc: true,
            enable_k2k: true,
            max_k2k_routes: 8,
            idle_poll_ns: 1000,
        }
    }
}

/// Control block state matching GPU ControlBlock struct.
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy)]
pub struct ControlBlock {
    pub is_active: u32,
    pub should_terminate: u32,
    pub has_terminated: u32,
    pub _padding1: u32,
    pub messages_processed: u64,
    pub messages_in_flight: u64,
    pub input_head: u64,
    pub input_tail: u64,
    pub output_head: u64,
    pub output_tail: u64,
    pub input_mask: u32,
    pub output_mask: u32,
    pub hlc_physical: u64,
    pub hlc_logical: u64,
    pub last_error: u32,
    pub error_count: u32,
    pub _reserved: [u8; 32], // 128 - 96 = 32 bytes reserved
}

unsafe impl Zeroable for ControlBlock {}

impl Default for ControlBlock {
    fn default() -> Self {
        Self::zeroed()
    }
}

const _: () = assert!(std::mem::size_of::<ControlBlock>() == 128);

/// HLC timestamp from GPU.
#[derive(Debug, Clone, Copy, Default)]
pub struct HlcTimestamp {
    pub physical: u64,
    pub logical: u64,
}

impl HlcTimestamp {
    pub fn new(physical: u64, logical: u64) -> Self {
        Self { physical, logical }
    }

    /// Compare timestamps for causal ordering.
    pub fn happens_before(&self, other: &HlcTimestamp) -> bool {
        self.physical < other.physical
            || (self.physical == other.physical && self.logical < other.logical)
    }
}

/// Ring kernel handle for managing kernel lifecycle.
pub struct RingKernelHandle {
    pub kernel_id: String,
    pub config: RingKernelConfig,
    control: ControlBlock,
    state: KernelState,
}

/// Kernel lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelState {
    Created,
    Active,
    Paused,
    Terminating,
    Terminated,
}

impl RingKernelHandle {
    /// Create a new ring kernel handle.
    pub fn new(kernel_id: String, config: RingKernelConfig) -> Self {
        let control = ControlBlock {
            input_mask: config.queue_capacity - 1,
            output_mask: config.queue_capacity - 1,
            ..Default::default()
        };

        Self {
            kernel_id,
            config,
            control,
            state: KernelState::Created,
        }
    }

    /// Activate the kernel.
    pub fn activate(&mut self) -> Result<(), String> {
        match self.state {
            KernelState::Created | KernelState::Paused => {
                self.control.is_active = 1;
                self.state = KernelState::Active;
                Ok(())
            }
            KernelState::Active => Err("Kernel already active".to_string()),
            KernelState::Terminating | KernelState::Terminated => {
                Err("Cannot activate terminated kernel".to_string())
            }
        }
    }

    /// Pause the kernel (stops processing but doesn't terminate).
    pub fn pause(&mut self) -> Result<(), String> {
        if self.state == KernelState::Active {
            self.control.is_active = 0;
            self.state = KernelState::Paused;
            Ok(())
        } else {
            Err("Kernel not active".to_string())
        }
    }

    /// Request kernel termination.
    pub fn terminate(&mut self) -> Result<(), String> {
        match self.state {
            KernelState::Terminated => Ok(()),
            _ => {
                self.control.should_terminate = 1;
                self.state = KernelState::Terminating;
                Ok(())
            }
        }
    }

    /// Check if kernel has terminated.
    pub fn is_terminated(&self) -> bool {
        self.control.has_terminated != 0
    }

    /// Get current HLC timestamp.
    pub fn hlc_now(&self) -> HlcTimestamp {
        HlcTimestamp {
            physical: self.control.hlc_physical,
            logical: self.control.hlc_logical,
        }
    }

    /// Get number of messages processed.
    pub fn messages_processed(&self) -> u64 {
        self.control.messages_processed
    }

    /// Get current state.
    pub fn state(&self) -> KernelState {
        self.state
    }

    /// Get input queue occupancy.
    pub fn input_queue_size(&self) -> u64 {
        self.control
            .input_head
            .wrapping_sub(self.control.input_tail)
    }

    /// Get output queue occupancy.
    pub fn output_queue_size(&self) -> u64 {
        self.control
            .output_head
            .wrapping_sub(self.control.output_tail)
    }

    /// Update control block from GPU state.
    pub fn update_from_gpu(&mut self, gpu_control: ControlBlock) {
        self.control = gpu_control;
        if self.control.has_terminated != 0 {
            self.state = KernelState::Terminated;
        }
    }
}

/// K2K route configuration.
#[derive(Debug, Clone)]
pub struct K2KRoute {
    pub source_kernel_id: String,
    pub target_kernel_id: String,
    pub route_id: u32,
}

/// Ring kernel backend manager.
///
/// Manages multiple ring kernels and their K2K connections.
pub struct RingKernelBackend {
    kernels: Vec<RingKernelHandle>,
    k2k_routes: Vec<K2KRoute>,
}

impl RingKernelBackend {
    pub fn new() -> Self {
        Self {
            kernels: Vec::new(),
            k2k_routes: Vec::new(),
        }
    }

    /// Create a new ring kernel.
    pub fn create_kernel(&mut self, kernel_id: String, config: RingKernelConfig) -> usize {
        let handle = RingKernelHandle::new(kernel_id, config);
        self.kernels.push(handle);
        self.kernels.len() - 1
    }

    /// Get a kernel handle by index.
    pub fn get_kernel(&self, index: usize) -> Option<&RingKernelHandle> {
        self.kernels.get(index)
    }

    /// Get a mutable kernel handle by index.
    pub fn get_kernel_mut(&mut self, index: usize) -> Option<&mut RingKernelHandle> {
        self.kernels.get_mut(index)
    }

    /// Add a K2K route between kernels.
    pub fn add_k2k_route(&mut self, source: &str, target: &str) -> Result<u32, String> {
        let route_id = self.k2k_routes.len() as u32;
        self.k2k_routes.push(K2KRoute {
            source_kernel_id: source.to_string(),
            target_kernel_id: target.to_string(),
            route_id,
        });
        Ok(route_id)
    }

    /// Get all kernels.
    pub fn kernels(&self) -> &[RingKernelHandle] {
        &self.kernels
    }

    /// Get active kernel count.
    pub fn active_kernel_count(&self) -> usize {
        self.kernels
            .iter()
            .filter(|k| k.state() == KernelState::Active)
            .count()
    }
}

impl Default for RingKernelBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Transaction monitoring pipeline using multiple ring kernels.
///
/// ## Pipeline Architecture
///
/// ```text
/// Input -> [Validator Kernel] -> [Pattern Detector] -> [Alert Aggregator] -> Output
///              |                       |                      |
///              +----------- K2K -------+------- K2K ----------+
/// ```
pub struct TxMonitorPipeline {
    backend: RingKernelBackend,
    validator_idx: usize,
    pattern_detector_idx: usize,
    aggregator_idx: usize,
}

impl TxMonitorPipeline {
    /// Create a new transaction monitoring pipeline.
    pub fn new() -> Self {
        let mut backend = RingKernelBackend::new();

        // Create validator kernel (first stage)
        let validator_config = RingKernelConfig {
            block_size: 256,
            queue_capacity: 8192,
            enable_hlc: true,
            enable_k2k: true,
            ..Default::default()
        };
        let validator_idx = backend.create_kernel("tx_validator".to_string(), validator_config);

        // Create pattern detector kernel (second stage)
        let pattern_config = RingKernelConfig {
            block_size: 128,
            queue_capacity: 4096,
            enable_hlc: true,
            enable_k2k: true,
            ..Default::default()
        };
        let pattern_detector_idx =
            backend.create_kernel("pattern_detector".to_string(), pattern_config);

        // Create alert aggregator kernel (final stage)
        let aggregator_config = RingKernelConfig {
            block_size: 64,
            queue_capacity: 2048,
            enable_hlc: true,
            enable_k2k: false, // Terminal kernel
            ..Default::default()
        };
        let aggregator_idx =
            backend.create_kernel("alert_aggregator".to_string(), aggregator_config);

        // Set up K2K routes
        backend
            .add_k2k_route("tx_validator", "pattern_detector")
            .unwrap();
        backend
            .add_k2k_route("pattern_detector", "alert_aggregator")
            .unwrap();

        Self {
            backend,
            validator_idx,
            pattern_detector_idx,
            aggregator_idx,
        }
    }

    /// Start the pipeline.
    pub fn start(&mut self) -> Result<(), String> {
        self.backend
            .get_kernel_mut(self.validator_idx)
            .unwrap()
            .activate()?;
        self.backend
            .get_kernel_mut(self.pattern_detector_idx)
            .unwrap()
            .activate()?;
        self.backend
            .get_kernel_mut(self.aggregator_idx)
            .unwrap()
            .activate()?;
        Ok(())
    }

    /// Stop the pipeline.
    pub fn stop(&mut self) -> Result<(), String> {
        self.backend
            .get_kernel_mut(self.validator_idx)
            .unwrap()
            .terminate()?;
        self.backend
            .get_kernel_mut(self.pattern_detector_idx)
            .unwrap()
            .terminate()?;
        self.backend
            .get_kernel_mut(self.aggregator_idx)
            .unwrap()
            .terminate()?;
        Ok(())
    }

    /// Get pipeline statistics.
    pub fn stats(&self) -> PipelineStats {
        let validator = self.backend.get_kernel(self.validator_idx).unwrap();
        let pattern = self.backend.get_kernel(self.pattern_detector_idx).unwrap();
        let aggregator = self.backend.get_kernel(self.aggregator_idx).unwrap();

        PipelineStats {
            validator_processed: validator.messages_processed(),
            pattern_processed: pattern.messages_processed(),
            aggregator_processed: aggregator.messages_processed(),
            validator_queue_size: validator.input_queue_size(),
            pattern_queue_size: pattern.input_queue_size(),
            aggregator_queue_size: aggregator.input_queue_size(),
        }
    }
}

impl Default for TxMonitorPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Pipeline statistics.
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    pub validator_processed: u64,
    pub pattern_processed: u64,
    pub aggregator_processed: u64,
    pub validator_queue_size: u64,
    pub pattern_queue_size: u64,
    pub aggregator_queue_size: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_block_size() {
        assert_eq!(std::mem::size_of::<ControlBlock>(), 128);
    }

    #[test]
    fn test_kernel_lifecycle() {
        let mut handle = RingKernelHandle::new("test".to_string(), RingKernelConfig::default());

        assert_eq!(handle.state(), KernelState::Created);

        handle.activate().unwrap();
        assert_eq!(handle.state(), KernelState::Active);

        handle.pause().unwrap();
        assert_eq!(handle.state(), KernelState::Paused);

        handle.activate().unwrap();
        assert_eq!(handle.state(), KernelState::Active);

        handle.terminate().unwrap();
        assert_eq!(handle.state(), KernelState::Terminating);
    }

    #[test]
    fn test_hlc_ordering() {
        let ts1 = HlcTimestamp::new(100, 0);
        let ts2 = HlcTimestamp::new(100, 1);
        let ts3 = HlcTimestamp::new(101, 0);

        assert!(ts1.happens_before(&ts2));
        assert!(ts2.happens_before(&ts3));
        assert!(ts1.happens_before(&ts3));
        assert!(!ts3.happens_before(&ts1));
    }

    #[test]
    fn test_pipeline_creation() {
        let pipeline = TxMonitorPipeline::new();
        assert_eq!(pipeline.backend.kernels().len(), 3);
    }

    #[test]
    fn test_pipeline_start_stop() {
        let mut pipeline = TxMonitorPipeline::new();
        pipeline.start().unwrap();

        assert_eq!(pipeline.backend.active_kernel_count(), 3);

        pipeline.stop().unwrap();
    }
}
