//! CUDA kernel management.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use cudarc::driver::{CudaFunction, CudaModule, LaunchAsync, LaunchConfig};
use parking_lot::RwLock;
use tokio::sync::{mpsc, Notify};

use ringkernel_core::error::{Result, RingKernelError};
use ringkernel_core::hlc::HlcClock;
use ringkernel_core::message::{CorrelationId, MessageEnvelope, RingMessage};
use ringkernel_core::runtime::{
    KernelHandle, KernelHandleInner, KernelId, KernelState, KernelStatus, LaunchOptions,
};
use ringkernel_core::telemetry::KernelMetrics;

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
    input_queue: CudaMessageQueue,
    /// Output queue on device.
    output_queue: CudaMessageQueue,
    /// HLC clock for timestamps.
    clock: HlcClock,
    /// Metrics.
    metrics: RwLock<KernelMetrics>,
    /// Pending responses (correlation_id -> sender).
    pending: RwLock<std::collections::HashMap<u64, tokio::sync::oneshot::Sender<MessageEnvelope>>>,
    /// Message counter.
    message_counter: AtomicU64,
    /// Created timestamp.
    created_at: Instant,
    /// Termination notifier.
    terminate_notify: Notify,
    /// CUDA module (compiled kernel).
    module: Option<Arc<CudaModule>>,
    /// Kernel function.
    kernel_fn: Option<CudaFunction>,
}

impl CudaKernel {
    /// Create a new CUDA kernel.
    pub fn new(
        id: &str,
        id_num: u64,
        device: CudaDevice,
        options: LaunchOptions,
    ) -> Result<Self> {
        let input_capacity = options.input_queue_capacity;
        let output_capacity = options.output_queue_capacity;

        let control_block = CudaControlBlock::new(&device)?;
        let input_queue = CudaMessageQueue::new(&device, input_capacity, 4096)?;
        let output_queue = CudaMessageQueue::new(&device, output_capacity, 4096)?;

        Ok(Self {
            id: KernelId::new(id),
            id_num,
            state: RwLock::new(KernelState::Initialized),
            options,
            device,
            control_block: RwLock::new(control_block),
            input_queue,
            output_queue,
            clock: HlcClock::new(id_num),
            metrics: RwLock::new(KernelMetrics::default()),
            pending: RwLock::new(std::collections::HashMap::new()),
            message_counter: AtomicU64::new(0),
            created_at: Instant::now(),
            terminate_notify: Notify::new(),
            module: None,
            kernel_fn: None,
        })
    }

    /// Load and compile the kernel PTX.
    pub fn load_ptx(&mut self, ptx: &str) -> Result<()> {
        let module = self
            .device
            .inner()
            .load_ptx(
                cudarc::nvrtc::compile_ptx(ptx).map_err(|e| {
                    RingKernelError::CompilationError(format!("PTX compilation failed: {}", e))
                })?,
                "ring_kernel",
                &["ring_kernel_main"],
            )
            .map_err(|e| {
                RingKernelError::CompilationError(format!("Module load failed: {}", e))
            })?;

        let kernel_fn = self
            .device
            .inner()
            .get_func("ring_kernel", "ring_kernel_main")
            .ok_or_else(|| {
                RingKernelError::CompilationError("Kernel function not found".to_string())
            })?;

        self.module = Some(Arc::new(module));
        self.kernel_fn = Some(kernel_fn);

        Ok(())
    }

    /// Launch the kernel on GPU.
    pub async fn launch(&self) -> Result<()> {
        let kernel_fn = self.kernel_fn.as_ref().ok_or_else(|| {
            RingKernelError::InvalidState("Kernel not loaded".to_string())
        })?;

        let control_ptr = self.control_block.read().device_ptr();
        let input_ptr = self.input_queue.headers_ptr();
        let output_ptr = self.output_queue.headers_ptr();
        let shared_ptr = 0u64; // Placeholder for shared state

        let grid_size = self.options.grid_size.unwrap_or(1);
        let block_size = self.options.block_size.unwrap_or(256);

        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        unsafe {
            kernel_fn
                .clone()
                .launch(config, (control_ptr, input_ptr, output_ptr, shared_ptr))
                .map_err(|e| {
                    RingKernelError::LaunchError(format!("Kernel launch failed: {}", e))
                })?;
        }

        Ok(())
    }

    /// Update metrics from control block.
    fn update_metrics(&self) -> Result<()> {
        let cb = self.control_block.read().read()?;
        let mut metrics = self.metrics.write();

        metrics.messages_processed = cb.messages_processed;
        metrics.uptime = self.created_at.elapsed();

        Ok(())
    }
}

#[async_trait]
impl KernelHandleInner for CudaKernel {
    fn id(&self) -> &str {
        self.id.as_str()
    }

    fn status(&self) -> KernelStatus {
        let state = *self.state.read();
        KernelStatus {
            id: self.id.clone(),
            state,
            uptime: self.created_at.elapsed(),
            messages_processed: self.message_counter.load(Ordering::Relaxed),
            messages_pending: 0, // Would need to read from control block
        }
    }

    fn metrics(&self) -> KernelMetrics {
        // Try to update metrics from device
        let _ = self.update_metrics();
        self.metrics.read().clone()
    }

    async fn activate(&self) -> Result<()> {
        let current_state = *self.state.read();
        if current_state != KernelState::Initialized && current_state != KernelState::Suspended {
            return Err(RingKernelError::InvalidStateTransition {
                from: current_state,
                to: KernelState::Active,
            });
        }

        // Set active flag on device
        self.control_block.write().set_active(true)?;

        // If first activation, launch the kernel
        if current_state == KernelState::Initialized {
            self.launch().await?;
        }

        *self.state.write() = KernelState::Active;
        tracing::info!(kernel_id = %self.id, "CUDA kernel activated");

        Ok(())
    }

    async fn suspend(&self) -> Result<()> {
        let current_state = *self.state.read();
        if current_state != KernelState::Active {
            return Err(RingKernelError::InvalidStateTransition {
                from: current_state,
                to: KernelState::Suspended,
            });
        }

        // Clear active flag (kernel will spin-wait)
        self.control_block.write().set_active(false)?;
        *self.state.write() = KernelState::Suspended;

        tracing::info!(kernel_id = %self.id, "CUDA kernel suspended");
        Ok(())
    }

    async fn resume(&self) -> Result<()> {
        let current_state = *self.state.read();
        if current_state != KernelState::Suspended {
            return Err(RingKernelError::InvalidStateTransition {
                from: current_state,
                to: KernelState::Active,
            });
        }

        // Set active flag
        self.control_block.write().set_active(true)?;
        *self.state.write() = KernelState::Active;

        tracing::info!(kernel_id = %self.id, "CUDA kernel resumed");
        Ok(())
    }

    async fn terminate(&self) -> Result<()> {
        // Request termination via control block
        self.control_block.write().request_termination()?;

        // Wait for kernel to terminate (with timeout)
        let start = Instant::now();
        let timeout = Duration::from_secs(5);

        while start.elapsed() < timeout {
            if self.control_block.read().is_terminated()? {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Synchronize device
        self.device.synchronize()?;

        *self.state.write() = KernelState::Terminated;
        self.terminate_notify.notify_waiters();

        tracing::info!(kernel_id = %self.id, "CUDA kernel terminated");
        Ok(())
    }

    async fn send<M: RingMessage + Send + Sync>(&self, message: M) -> Result<()> {
        // Serialize message
        let payload = message.serialize();

        // Create header
        let header = ringkernel_core::message::MessageHeader::new(
            M::message_type(),
            0, // source (host)
            self.id_num,
            payload.len(),
            self.clock.tick(),
        );

        // TODO: Copy to input queue on device
        // For now, this is a placeholder - actual implementation would:
        // 1. Get current input tail from control block
        // 2. Copy header to headers buffer at tail position
        // 3. Copy payload to payloads buffer at tail position
        // 4. Atomically increment tail

        self.message_counter.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    async fn request<M: RingMessage + Send + Sync, R: RingMessage + Send + Sync>(
        &self,
        message: M,
        timeout: Duration,
    ) -> Result<R> {
        let correlation_id = message.correlation_id().0;

        // Create response channel
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.pending.write().insert(correlation_id, tx);

        // Send the message
        self.send(message).await?;

        // Wait for response with timeout
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(envelope)) => R::deserialize(&envelope.payload),
            Ok(Err(_)) => Err(RingKernelError::ChannelError(
                "Response channel closed".to_string(),
            )),
            Err(_) => {
                self.pending.write().remove(&correlation_id);
                Err(RingKernelError::Timeout(timeout))
            }
        }
    }

    async fn recv(&self) -> Result<MessageEnvelope> {
        // TODO: Read from output queue on device
        // This would need to poll the output queue or use events
        Err(RingKernelError::QueueEmpty)
    }

    fn try_recv(&self) -> Option<MessageEnvelope> {
        // TODO: Non-blocking read from output queue
        None
    }
}

impl Drop for CudaKernel {
    fn drop(&mut self) {
        // Ensure kernel is terminated
        if *self.state.read() != KernelState::Terminated {
            let _ = self.control_block.write().request_termination();
            // Best effort synchronize
            let _ = self.device.synchronize();
        }
    }
}
