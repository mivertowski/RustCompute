//! WebGPU kernel management.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
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

use crate::adapter::WgpuAdapter;
use crate::memory::{WgpuControlBlock, WgpuMessageQueue};
use crate::shader::{create_bind_group, ComputePipeline};

/// WebGPU kernel handle.
pub struct WgpuKernel {
    /// Kernel identifier.
    id: KernelId,
    /// Numeric ID.
    id_num: u64,
    /// Current state.
    state: RwLock<KernelState>,
    /// Launch options.
    options: LaunchOptions,
    /// WebGPU adapter.
    adapter: Arc<WgpuAdapter>,
    /// Control block.
    control_block: RwLock<WgpuControlBlock>,
    /// Input queue.
    input_queue: WgpuMessageQueue,
    /// Output queue.
    #[allow(dead_code)]
    output_queue: WgpuMessageQueue,
    /// Compute pipeline.
    pipeline: Option<ComputePipeline>,
    /// Bind group.
    bind_group: Option<wgpu::BindGroup>,
    /// HLC clock.
    clock: HlcClock,
    /// Metrics.
    metrics: RwLock<KernelMetrics>,
    /// Message counter.
    message_counter: AtomicU64,
    /// Created timestamp.
    created_at: Instant,
    /// Termination notifier.
    terminate_notify: Notify,
}

impl WgpuKernel {
    /// Create a new WebGPU kernel.
    pub fn new(
        id: &str,
        id_num: u64,
        adapter: Arc<WgpuAdapter>,
        options: LaunchOptions,
    ) -> Result<Self> {
        let input_capacity = options.input_queue_capacity;
        let output_capacity = options.output_queue_capacity;

        let control_block = WgpuControlBlock::new(&adapter);
        let input_queue = WgpuMessageQueue::new(&adapter, input_capacity, 4096);
        let output_queue = WgpuMessageQueue::new(&adapter, output_capacity, 4096);

        Ok(Self {
            id: KernelId::new(id),
            id_num,
            state: RwLock::new(KernelState::Created),
            options,
            adapter,
            control_block: RwLock::new(control_block),
            input_queue,
            output_queue,
            pipeline: None,
            bind_group: None,
            clock: HlcClock::new(id_num),
            metrics: RwLock::new(KernelMetrics::default()),
            message_counter: AtomicU64::new(0),
            created_at: Instant::now(),
            terminate_notify: Notify::new(),
        })
    }

    /// Get the kernel ID.
    #[allow(dead_code)]
    pub fn kernel_id(&self) -> &KernelId {
        &self.id
    }

    /// Load and compile the shader.
    pub fn load_shader(&mut self, wgsl_source: &str) -> Result<()> {
        let workgroup_size = self.options.block_size;

        let pipeline =
            ComputePipeline::new(&self.adapter, wgsl_source, "main", (workgroup_size, 1, 1))?;

        // Create bind group
        let bind_group = create_bind_group(
            self.adapter.device(),
            pipeline.bind_group_layout(),
            self.control_block.read().as_binding(),
            self.input_queue.headers_binding(),
            self.input_queue.headers_binding(), // Using headers for output too for now
        );

        self.pipeline = Some(pipeline);
        self.bind_group = Some(bind_group);
        *self.state.write() = KernelState::Launched;

        Ok(())
    }

    /// Dispatch the compute shader.
    #[allow(dead_code)]
    pub fn dispatch(&self, workgroups: u32) -> Result<()> {
        let pipeline = self
            .pipeline
            .as_ref()
            .ok_or_else(|| RingKernelError::LaunchFailed("Shader not loaded".to_string()))?;

        let bind_group = self
            .bind_group
            .as_ref()
            .ok_or_else(|| RingKernelError::LaunchFailed("Bind group not created".to_string()))?;

        let mut encoder =
            self.adapter
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("RingKernel Dispatch"),
                });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RingKernel Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline.pipeline());
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.adapter.queue().submit(Some(encoder.finish()));

        Ok(())
    }

    /// Wait for GPU to complete.
    #[allow(dead_code)]
    pub fn gpu_wait(&self) {
        self.adapter.poll(wgpu::Maintain::Wait);
    }
}

#[async_trait]
impl KernelHandleInner for WgpuKernel {
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
            mode: KernelMode::EventDriven, // WebGPU is event-driven
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

        *self.state.write() = KernelState::Active;
        tracing::info!(kernel_id = %self.id, "WebGPU kernel activated");

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
        tracing::info!(kernel_id = %self.id, "WebGPU kernel deactivated");

        Ok(())
    }

    async fn terminate(&self) -> Result<()> {
        // Request termination
        {
            let cb_lock = self.control_block.write();
            let mut cb = cb_lock.read()?;
            cb.should_terminate = 1;
            cb_lock.write(&cb)?;
        }

        *self.state.write() = KernelState::Terminating;

        // Wait for GPU to finish any pending work
        self.adapter.poll(wgpu::Maintain::Wait);

        // Mark as terminated
        {
            let cb_lock = self.control_block.write();
            let mut cb = cb_lock.read()?;
            cb.has_terminated = 1;
            cb_lock.write(&cb)?;
        }

        self.terminate_notify.notify_waiters();

        tracing::info!(kernel_id = %self.id, "WebGPU kernel terminated");
        Ok(())
    }

    async fn send_envelope(&self, envelope: MessageEnvelope) -> Result<()> {
        // TODO: Copy envelope to input queue on GPU
        // For WebGPU, this would involve:
        // 1. Write header to headers buffer
        // 2. Write payload to payloads buffer
        // 3. Update tail index in control block
        // 4. Dispatch compute shader

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
