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

use crate::actor_loop::{ActorLoopConfig, WgpuActorLoop, ACTOR_LOOP_WGSL_TEMPLATE};
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
    /// Actor loop for emulated persistent mode.
    /// When `options.mode == Persistent`, an actor loop is created during
    /// construction and started on `activate()`.
    actor_loop: Option<Arc<WgpuActorLoop>>,
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

        // When persistent mode is requested, create an actor loop that
        // emulates persistence via rapid dispatch + state preservation.
        let actor_loop = if options.mode == KernelMode::Persistent {
            let config = ActorLoopConfig::default();
            let loop_instance = WgpuActorLoop::new(
                Arc::clone(&adapter),
                ACTOR_LOOP_WGSL_TEMPLATE,
                "actor_main",
                config,
            )?;
            Some(Arc::new(loop_instance))
        } else {
            None
        };

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
            actor_loop,
        })
    }

    /// Get the actor loop, if this kernel uses emulated persistent mode.
    pub fn actor_loop(&self) -> Option<&Arc<WgpuActorLoop>> {
        self.actor_loop.as_ref()
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
        // Report Persistent when the actor loop is active (emulated),
        // otherwise EventDriven.
        let mode = if self.actor_loop.is_some() {
            KernelMode::Persistent
        } else {
            KernelMode::EventDriven
        };
        KernelStatus {
            id: self.id.clone(),
            state,
            mode,
            input_queue_depth: cb.input_queue_size() as usize,
            output_queue_depth: cb.output_queue_size() as usize,
            messages_processed: self.message_counter.load(Ordering::Relaxed),
            uptime: self.created_at.elapsed(),
        }
    }

    fn metrics(&self) -> KernelMetrics {
        let mut metrics = self.metrics.read().clone();
        let state = *self.state.read();
        let cb = self.control_block.read().read().unwrap_or_default();
        metrics.kernel_id = self.id.to_string();
        metrics.uptime = self.created_at.elapsed();
        metrics.messages_sent = self.message_counter.load(Ordering::Relaxed);
        metrics.messages_received = cb.messages_processed;
        metrics.input_queue_depth = cb.input_queue_size() as usize;
        metrics.output_queue_depth = cb.output_queue_size() as usize;
        metrics.state = state;
        metrics.gpu_launched = false; // WebGPU is event-driven, not persistently launched
        metrics
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

        // Start the actor loop if we are emulating persistent mode.
        if let Some(actor_loop) = &self.actor_loop {
            actor_loop.start();
            tracing::info!(
                kernel_id = %self.id,
                "WebGPU kernel activated (emulated persistent mode via actor loop)"
            );
        } else {
            tracing::info!(kernel_id = %self.id, "WebGPU kernel activated");
        }

        *self.state.write() = KernelState::Active;

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

        // Stop the actor loop if running.
        if let Some(actor_loop) = &self.actor_loop {
            actor_loop.stop();
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
        // Stop the actor loop if running.
        if let Some(actor_loop) = &self.actor_loop {
            actor_loop.stop();
        }

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
        // Check kernel is active
        let state = *self.state.read();
        if state != KernelState::Active {
            return Err(RingKernelError::KernelNotActive(self.id.to_string()));
        }

        // Enqueue to input queue
        self.input_queue.enqueue(&envelope)?;

        self.message_counter.fetch_add(1, Ordering::Relaxed);

        // For event-driven mode, dispatch compute shader after each message
        if self.options.mode == KernelMode::EventDriven {
            if let (Some(_pipeline), Some(_bind_group)) = (&self.pipeline, &self.bind_group) {
                // Dispatch one workgroup to process the message
                self.dispatch(1)?;
            }
        }

        Ok(())
    }

    async fn receive(&self) -> Result<MessageEnvelope> {
        // Poll for available messages
        loop {
            // Try to dequeue from output queue
            if let Some(envelope) = self.output_queue.try_dequeue() {
                return Ok(envelope);
            }

            // Check if terminated
            if *self.state.read() == KernelState::Terminated {
                return Err(RingKernelError::QueueEmpty);
            }

            // Poll GPU and yield
            self.adapter.poll(wgpu::Maintain::Poll);
            tokio::task::yield_now().await;
        }
    }

    async fn receive_timeout(&self, timeout: Duration) -> Result<MessageEnvelope> {
        // Try to receive with timeout
        match tokio::time::timeout(timeout, self.receive()).await {
            Ok(result) => result,
            Err(_) => Err(RingKernelError::Timeout(timeout)),
        }
    }

    fn try_receive(&self) -> Result<MessageEnvelope> {
        // Non-blocking read from output queue
        self.output_queue
            .try_dequeue()
            .ok_or(RingKernelError::QueueEmpty)
    }

    async fn receive_correlated(
        &self,
        correlation: CorrelationId,
        timeout: Duration,
    ) -> Result<MessageEnvelope> {
        // Simple correlation implementation - receive until we find matching correlation
        let start = Instant::now();
        loop {
            match self.try_receive() {
                Ok(envelope) => {
                    if envelope.header.correlation_id == correlation {
                        return Ok(envelope);
                    }
                    // Not the right message, continue waiting
                }
                Err(RingKernelError::QueueEmpty) => {
                    // Check timeout
                    if start.elapsed() >= timeout {
                        return Err(RingKernelError::Timeout(timeout));
                    }
                    // Poll GPU and yield
                    self.adapter.poll(wgpu::Maintain::Poll);
                    tokio::task::yield_now().await;
                }
                Err(e) => return Err(e),
            }
        }
    }

    async fn wait(&self) -> Result<()> {
        // Wait for termination
        self.terminate_notify.notified().await;
        Ok(())
    }
}
