//! CUDA kernel management.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use cudarc::driver::{CudaFunction, LaunchAsync, LaunchConfig};
use parking_lot::RwLock;
use tokio::sync::Notify;

use ringkernel_core::error::{Result, RingKernelError};
use ringkernel_core::hlc::{HlcClock, HlcTimestamp};
use ringkernel_core::message::{CorrelationId, MessageEnvelope};
use ringkernel_core::runtime::{
    KernelHandleInner, KernelId, KernelState, KernelStatus, LaunchOptions,
};
use ringkernel_core::telemetry::KernelMetrics;
use ringkernel_core::types::KernelMode;

use crate::device::CudaDevice;
use crate::memory::{CudaControlBlock, CudaMessageQueue};

/// CUDA kernel handle.
pub struct CudaKernel {
    /// Kernel identifier.
    id: KernelId,
    /// Numeric ID for message routing.
    id_num: u64,
    /// Current state.
    state: RwLock<KernelState>,
    /// Launch options.
    options: LaunchOptions,
    /// Device reference.
    device: CudaDevice,
    /// Control block on device.
    control_block: RwLock<CudaControlBlock>,
    /// Input queue on device.
    #[allow(dead_code)]
    input_queue: CudaMessageQueue,
    /// Output queue on device.
    #[allow(dead_code)]
    output_queue: CudaMessageQueue,
    /// HLC clock for timestamps.
    clock: HlcClock,
    /// Metrics.
    metrics: RwLock<KernelMetrics>,
    /// Message counter.
    message_counter: AtomicU64,
    /// Created timestamp.
    created_at: Instant,
    /// Termination notifier.
    terminate_notify: Notify,
    /// Kernel function.
    #[allow(dead_code)]
    kernel_fn: Option<CudaFunction>,
    /// Whether the kernel has been launched on the GPU.
    gpu_launched: AtomicBool,
}

impl CudaKernel {
    /// Create a new CUDA kernel.
    pub fn new(id: &str, id_num: u64, device: CudaDevice, options: LaunchOptions) -> Result<Self> {
        let input_capacity = options.input_queue_capacity;
        let output_capacity = options.output_queue_capacity;

        let control_block = CudaControlBlock::new(&device)?;
        let input_queue = CudaMessageQueue::new(&device, input_capacity, 4096)?;
        let output_queue = CudaMessageQueue::new(&device, output_capacity, 4096)?;

        Ok(Self {
            id: KernelId::new(id),
            id_num,
            state: RwLock::new(KernelState::Created),
            options,
            device,
            control_block: RwLock::new(control_block),
            input_queue,
            output_queue,
            clock: HlcClock::new(id_num),
            metrics: RwLock::new(KernelMetrics::default()),
            message_counter: AtomicU64::new(0),
            created_at: Instant::now(),
            terminate_notify: Notify::new(),
            kernel_fn: None,
            gpu_launched: AtomicBool::new(false),
        })
    }

    /// Get the kernel ID.
    #[allow(dead_code)]
    pub fn kernel_id(&self) -> &KernelId {
        &self.id
    }

    /// Load and compile the kernel PTX.
    pub fn load_ptx(&mut self, ptx: &str) -> Result<()> {
        // Load PTX directly (PTX is already compiled assembly, not CUDA C)
        let ptx_code = cudarc::nvrtc::Ptx::from_src(ptx);

        self.device
            .inner()
            .load_ptx(ptx_code, "ring_kernel", &["ring_kernel_main"])
            .map_err(|e| RingKernelError::LaunchFailed(format!("PTX load failed: {}", e)))?;

        let kernel_fn = self
            .device
            .inner()
            .get_func("ring_kernel", "ring_kernel_main")
            .ok_or_else(|| {
                RingKernelError::LaunchFailed("Kernel function not found".to_string())
            })?;

        self.kernel_fn = Some(kernel_fn);
        *self.state.write() = KernelState::Launched;

        Ok(())
    }

    /// Launch the kernel on GPU.
    pub async fn launch_kernel(&self) -> Result<()> {
        let kernel_fn = self
            .kernel_fn
            .as_ref()
            .ok_or_else(|| RingKernelError::LaunchFailed("Kernel not loaded".to_string()))?;

        let control_ptr = self.control_block.read().device_ptr();
        let input_ptr = self.input_queue.headers_ptr();
        let output_ptr = self.output_queue.headers_ptr();
        let shared_ptr = 0u64; // Placeholder for shared state

        let grid_size = self.options.grid_size;
        let block_size = self.options.block_size;

        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        unsafe {
            kernel_fn
                .clone()
                .launch(config, (control_ptr, input_ptr, output_ptr, shared_ptr))
                .map_err(|e| {
                    RingKernelError::LaunchFailed(format!("Kernel launch failed: {}", e))
                })?;
        }

        // Mark that the GPU kernel has been launched
        self.gpu_launched.store(true, Ordering::Release);

        Ok(())
    }
}

#[async_trait]
impl KernelHandleInner for CudaKernel {
    fn kernel_id_num(&self) -> u64 {
        self.id_num
    }

    fn current_timestamp(&self) -> HlcTimestamp {
        self.clock.tick()
    }

    fn status(&self) -> KernelStatus {
        let state = *self.state.read();
        let cb = self.control_block.read().read().unwrap_or_default();
        KernelStatus {
            id: self.id.clone(),
            state,
            mode: KernelMode::Persistent, // CUDA kernels are persistent
            input_queue_depth: cb.input_queue_size() as usize,
            output_queue_depth: cb.output_queue_size() as usize,
            messages_processed: self.message_counter.load(Ordering::Relaxed),
            uptime: self.created_at.elapsed(),
        }
    }

    fn metrics(&self) -> KernelMetrics {
        self.metrics.read().clone()
    }

    async fn activate(&self) -> Result<()> {
        let current_state = *self.state.read();
        if current_state != KernelState::Launched && current_state != KernelState::Deactivated {
            return Err(RingKernelError::InvalidStateTransition {
                from: format!("{:?}", current_state),
                to: "Active".to_string(),
            });
        }

        // Set active flag in control block
        {
            let cb_lock = self.control_block.write();
            let mut cb = cb_lock.read()?;
            cb.is_active = 1;
            cb_lock.write(&cb)?;
        }

        // Launch the GPU kernel if this is the first activation
        if current_state == KernelState::Launched {
            self.launch_kernel().await?;
        }

        *self.state.write() = KernelState::Active;
        tracing::info!(kernel_id = %self.id, "CUDA kernel activated");

        Ok(())
    }

    async fn deactivate(&self) -> Result<()> {
        let current_state = *self.state.read();
        if current_state != KernelState::Active {
            return Err(RingKernelError::InvalidStateTransition {
                from: format!("{:?}", current_state),
                to: "Deactivated".to_string(),
            });
        }

        // Clear active flag
        {
            let cb_lock = self.control_block.write();
            let mut cb = cb_lock.read()?;
            cb.is_active = 0;
            cb_lock.write(&cb)?;
        }

        *self.state.write() = KernelState::Deactivated;
        tracing::info!(kernel_id = %self.id, "CUDA kernel deactivated");

        Ok(())
    }

    async fn terminate(&self) -> Result<()> {
        // Request termination via control block
        {
            let cb_lock = self.control_block.write();
            let mut cb = cb_lock.read()?;
            cb.should_terminate = 1;
            cb_lock.write(&cb)?;
        }

        *self.state.write() = KernelState::Terminating;

        // Only wait for kernel to terminate if it was actually launched on GPU
        if self.gpu_launched.load(Ordering::Acquire) {
            // Wait for kernel to terminate (with timeout)
            let start = Instant::now();
            let timeout = Duration::from_secs(5);

            while start.elapsed() < timeout {
                let cb = self.control_block.read().read()?;
                if cb.has_terminated() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }

            // Synchronize device
            self.device.synchronize()?;
        }

        // Mark as terminated
        {
            let cb_lock = self.control_block.write();
            let mut cb = cb_lock.read()?;
            cb.has_terminated = 1;
            cb_lock.write(&cb)?;
        }

        *self.state.write() = KernelState::Terminated;
        self.terminate_notify.notify_waiters();

        tracing::info!(kernel_id = %self.id, "CUDA kernel terminated");
        Ok(())
    }

    async fn send_envelope(&self, envelope: MessageEnvelope) -> Result<()> {
        // TODO: Copy envelope to input queue on GPU
        // For CUDA, this would involve:
        // 1. Write header to headers buffer
        // 2. Write payload to payloads buffer
        // 3. Update tail index in control block
        // 4. Signal kernel if needed

        let _ = envelope; // Suppress unused warning for now
        self.message_counter.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    async fn receive(&self) -> Result<MessageEnvelope> {
        // TODO: Read from output queue - for now just wait
        self.terminate_notify.notified().await;
        Err(RingKernelError::QueueEmpty)
    }

    async fn receive_timeout(&self, timeout: Duration) -> Result<MessageEnvelope> {
        // Try to receive with timeout
        match tokio::time::timeout(timeout, self.receive()).await {
            Ok(result) => result,
            Err(_) => Err(RingKernelError::Timeout(timeout)),
        }
    }

    fn try_receive(&self) -> Result<MessageEnvelope> {
        // TODO: Non-blocking read from output queue
        Err(RingKernelError::QueueEmpty)
    }

    async fn receive_correlated(
        &self,
        _correlation: CorrelationId,
        timeout: Duration,
    ) -> Result<MessageEnvelope> {
        // TODO: Implement correlation tracking
        self.receive_timeout(timeout).await
    }

    async fn wait(&self) -> Result<()> {
        // Wait for termination
        self.terminate_notify.notified().await;
        Ok(())
    }
}

impl Drop for CudaKernel {
    fn drop(&mut self) {
        // Ensure kernel is terminated - only do cleanup if not already terminated
        if *self.state.read() != KernelState::Terminated {
            // Request termination via control block
            if let Ok(mut cb) = self.control_block.get_mut().read() {
                cb.should_terminate = 1;
                let _ = self.control_block.get_mut().write(&cb);
            }
            // Best effort synchronize - don't block indefinitely
            let _ = self.device.synchronize();
        }
    }
}
