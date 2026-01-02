//! Metal kernel implementation.

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use metal::{ComputePipelineState, Device, Library, MTLSize};
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

use crate::device::MetalDevice;
use crate::memory::MetalBuffer;

// ============================================================================
// CONTROL BLOCK
// ============================================================================

/// Control block structure for Metal (128 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct MetalControlBlock {
    /// Is kernel active.
    pub is_active: u32,
    /// Should terminate.
    pub should_terminate: u32,
    /// Has terminated.
    pub has_terminated: u32,
    /// Padding.
    pub _pad1: u32,

    /// Messages processed (low 32 bits).
    pub messages_processed_lo: u32,
    /// Messages processed (high 32 bits).
    pub messages_processed_hi: u32,
    /// Messages in flight (low 32 bits).
    pub messages_in_flight_lo: u32,
    /// Messages in flight (high 32 bits).
    pub messages_in_flight_hi: u32,

    /// Input head (low 32 bits).
    pub input_head_lo: u32,
    /// Input head (high 32 bits).
    pub input_head_hi: u32,
    /// Input tail (low 32 bits).
    pub input_tail_lo: u32,
    /// Input tail (high 32 bits).
    pub input_tail_hi: u32,

    /// Output head (low 32 bits).
    pub output_head_lo: u32,
    /// Output head (high 32 bits).
    pub output_head_hi: u32,
    /// Output tail (low 32 bits).
    pub output_tail_lo: u32,
    /// Output tail (high 32 bits).
    pub output_tail_hi: u32,

    /// Input queue capacity.
    pub input_capacity: u32,
    /// Output queue capacity.
    pub output_capacity: u32,
    /// Input mask.
    pub input_mask: u32,
    /// Output mask.
    pub output_mask: u32,

    /// HLC physical (low 32 bits).
    pub hlc_physical_lo: u32,
    /// HLC physical (high 32 bits).
    pub hlc_physical_hi: u32,
    /// HLC logical (low 32 bits).
    pub hlc_logical_lo: u32,
    /// HLC logical (high 32 bits).
    pub hlc_logical_hi: u32,

    /// Last error code.
    pub last_error: u32,
    /// Error count.
    pub error_count: u32,

    /// Reserved for future use.
    pub _reserved: [u8; 16],
}

impl MetalControlBlock {
    /// Get messages processed as u64.
    pub fn messages_processed(&self) -> u64 {
        ((self.messages_processed_hi as u64) << 32) | (self.messages_processed_lo as u64)
    }

    /// Get input queue size.
    pub fn input_queue_size(&self) -> u64 {
        let head = ((self.input_head_hi as u64) << 32) | (self.input_head_lo as u64);
        let tail = ((self.input_tail_hi as u64) << 32) | (self.input_tail_lo as u64);
        tail.saturating_sub(head)
    }

    /// Get output queue size.
    pub fn output_queue_size(&self) -> u64 {
        let head = ((self.output_head_hi as u64) << 32) | (self.output_head_lo as u64);
        let tail = ((self.output_tail_hi as u64) << 32) | (self.output_tail_lo as u64);
        tail.saturating_sub(head)
    }
}

// ============================================================================
// MESSAGE QUEUE
// ============================================================================

/// A message queue backed by Metal buffers.
pub struct MetalMessageQueue {
    /// Header buffer.
    headers: MetalBuffer,
    /// Payload buffer.
    payloads: MetalBuffer,
    /// Queue capacity.
    capacity: u32,
    /// Head index (next to read).
    head: AtomicU64,
    /// Tail index (next to write).
    tail: AtomicU64,
}

impl MetalMessageQueue {
    /// Create a new message queue.
    pub fn new(device: &Device, capacity: u32, max_payload: u32) -> Result<Self> {
        let header_size = capacity as usize * 256; // 256 bytes per header
        let payload_size = capacity as usize * max_payload as usize;

        let headers = MetalBuffer::new(device, header_size)?;
        let payloads = MetalBuffer::new(device, payload_size)?;

        Ok(Self {
            headers,
            payloads,
            capacity,
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
        })
    }

    /// Enqueue a message envelope.
    pub fn enqueue(&self, envelope: &MessageEnvelope) -> Result<()> {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);

        if tail - head >= self.capacity as u64 {
            return Err(RingKernelError::QueueFull);
        }

        let idx = (tail % self.capacity as u64) as usize;

        // Serialize header to buffer
        let header_offset = idx * 256;
        let header_bytes = envelope.header.to_bytes();
        unsafe {
            let ptr = self.headers.buffer().contents() as *mut u8;
            std::ptr::copy_nonoverlapping(
                header_bytes.as_ptr(),
                ptr.add(header_offset),
                header_bytes.len().min(256),
            );
        }

        // Serialize payload
        let payload_offset = idx * 4096; // max_payload assumed 4096
        unsafe {
            let ptr = self.payloads.buffer().contents() as *mut u8;
            std::ptr::copy_nonoverlapping(
                envelope.payload.as_ptr(),
                ptr.add(payload_offset),
                envelope.payload.len().min(4096),
            );
        }

        self.tail.fetch_add(1, Ordering::Release);
        Ok(())
    }

    /// Try to dequeue a message.
    pub fn try_dequeue(&self) -> Option<MessageEnvelope> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);

        if head >= tail {
            return None;
        }

        let idx = (head % self.capacity as u64) as usize;

        // Read header
        let header_offset = idx * 256;
        let header_bytes: Vec<u8> = unsafe {
            let ptr = self.headers.buffer().contents() as *const u8;
            std::slice::from_raw_parts(ptr.add(header_offset), 256).to_vec()
        };

        let header = match ringkernel_core::message::MessageHeader::from_bytes(&header_bytes) {
            Ok(h) => h,
            Err(_) => return None,
        };

        // Read payload
        let payload_offset = idx * 4096;
        let payload_len = header.payload_size as usize;
        let payload: Vec<u8> = unsafe {
            let ptr = self.payloads.buffer().contents() as *const u8;
            std::slice::from_raw_parts(ptr.add(payload_offset), payload_len.min(4096)).to_vec()
        };

        self.head.fetch_add(1, Ordering::Release);

        Some(MessageEnvelope { header, payload })
    }

    /// Get the headers buffer.
    pub fn headers_buffer(&self) -> &MetalBuffer {
        &self.headers
    }

    /// Get the payloads buffer.
    #[allow(dead_code)]
    pub fn payloads_buffer(&self) -> &MetalBuffer {
        &self.payloads
    }
}

// ============================================================================
// METAL KERNEL
// ============================================================================

/// A Metal compute kernel implementing the RingKernel model.
pub struct MetalKernel {
    /// Kernel identifier.
    id: KernelId,
    /// Numeric kernel ID.
    id_num: u64,
    /// Current state.
    state: RwLock<KernelState>,
    /// Launch options.
    options: LaunchOptions,
    /// Metal device.
    device: Arc<MetalDevice>,
    /// Control block buffer.
    control_block: MetalBuffer,
    /// Input queue.
    input_queue: MetalMessageQueue,
    /// Output queue.
    output_queue: MetalMessageQueue,
    /// Compiled library.
    #[allow(dead_code)]
    library: Option<Library>,
    /// Compute pipeline state.
    pipeline: Option<ComputePipelineState>,
    /// Command queue.
    command_queue: metal::CommandQueue,
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

impl MetalKernel {
    /// Create a new Metal kernel.
    pub fn new(
        id: &str,
        id_num: u64,
        device: Arc<MetalDevice>,
        options: LaunchOptions,
    ) -> Result<Self> {
        let input_capacity = options.input_queue_capacity;
        let output_capacity = options.output_queue_capacity;

        // Create control block buffer
        let control_block =
            MetalBuffer::new(device.device(), std::mem::size_of::<MetalControlBlock>())?;

        // Initialize control block
        {
            let cb = MetalControlBlock {
                input_capacity: input_capacity as u32,
                output_capacity: output_capacity as u32,
                input_mask: (input_capacity as u32).saturating_sub(1),
                output_mask: (output_capacity as u32).saturating_sub(1),
                ..Default::default()
            };
            let cb_bytes = unsafe {
                std::slice::from_raw_parts(
                    &cb as *const MetalControlBlock as *const u8,
                    std::mem::size_of::<MetalControlBlock>(),
                )
            };
            let dest = control_block.as_slice();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    cb_bytes.as_ptr(),
                    dest.as_ptr() as *mut u8,
                    cb_bytes.len(),
                );
            }
        }

        // Create queues
        let input_queue = MetalMessageQueue::new(device.device(), input_capacity as u32, 4096)?;
        let output_queue = MetalMessageQueue::new(device.device(), output_capacity as u32, 4096)?;

        // Create command queue
        let command_queue = device.device().new_command_queue();

        Ok(Self {
            id: KernelId::new(id),
            id_num,
            state: RwLock::new(KernelState::Created),
            options,
            device,
            control_block,
            input_queue,
            output_queue,
            library: None,
            pipeline: None,
            command_queue,
            clock: HlcClock::new(id_num),
            metrics: RwLock::new(KernelMetrics::default()),
            message_counter: AtomicU64::new(0),
            created_at: Instant::now(),
            terminate_notify: Notify::new(),
        })
    }

    /// Get the kernel ID.
    pub fn kernel_id(&self) -> &KernelId {
        &self.id
    }

    /// Get the numeric ID.
    pub fn id_num(&self) -> u64 {
        self.id_num
    }

    /// Load and compile MSL shader source.
    pub fn load_shader(&mut self, msl_source: &str) -> Result<()> {
        let compile_options = metal::CompileOptions::new();
        let library = self
            .device
            .device()
            .new_library_with_source(msl_source, &compile_options)
            .map_err(|e| RingKernelError::CompilationFailed(e.to_string()))?;

        let function = library
            .get_function("ring_kernel_main", None)
            .map_err(|e| RingKernelError::CompilationFailed(e.to_string()))?;

        let pipeline = self
            .device
            .device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| RingKernelError::CompilationFailed(e.to_string()))?;

        self.library = Some(library);
        self.pipeline = Some(pipeline);
        *self.state.write() = KernelState::Launched;

        tracing::info!(kernel_id = %self.id, "Metal shader compiled");
        Ok(())
    }

    /// Dispatch the compute shader.
    pub fn dispatch(&self, threads: u64) -> Result<()> {
        let pipeline = self
            .pipeline
            .as_ref()
            .ok_or_else(|| RingKernelError::LaunchFailed("Pipeline not created".to_string()))?;

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(self.control_block.buffer()), 0);
        encoder.set_buffer(1, Some(self.input_queue.headers_buffer().buffer()), 0);
        encoder.set_buffer(2, Some(self.output_queue.headers_buffer().buffer()), 0);

        let threadgroup_size = MTLSize::new(self.options.block_size as u64, 1, 1);
        let grid_size = MTLSize::new(threads, 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }

    /// Read the control block.
    fn read_control_block(&self) -> MetalControlBlock {
        let slice = self.control_block.as_slice();
        unsafe { std::ptr::read(slice.as_ptr() as *const MetalControlBlock) }
    }

    /// Write to the control block.
    fn write_control_block(&self, cb: &MetalControlBlock) {
        let cb_bytes = unsafe {
            std::slice::from_raw_parts(
                cb as *const MetalControlBlock as *const u8,
                std::mem::size_of::<MetalControlBlock>(),
            )
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                cb_bytes.as_ptr(),
                self.control_block.as_slice().as_ptr() as *mut u8,
                cb_bytes.len(),
            );
        }
    }
}

#[async_trait]
impl KernelHandleInner for MetalKernel {
    fn kernel_id_num(&self) -> u64 {
        self.id_num
    }

    fn current_timestamp(&self) -> HlcTimestamp {
        self.clock.tick()
    }

    fn status(&self) -> KernelStatus {
        let state = *self.state.read();
        let cb = self.read_control_block();

        KernelStatus {
            id: self.id.clone(),
            state,
            mode: KernelMode::EventDriven, // Metal is event-driven like WebGPU
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
        let mut cb = self.read_control_block();
        cb.is_active = 1;
        self.write_control_block(&cb);

        *self.state.write() = KernelState::Active;
        tracing::info!(kernel_id = %self.id, "Metal kernel activated");

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
        let mut cb = self.read_control_block();
        cb.is_active = 0;
        self.write_control_block(&cb);

        *self.state.write() = KernelState::Deactivated;
        tracing::info!(kernel_id = %self.id, "Metal kernel deactivated");

        Ok(())
    }

    async fn terminate(&self) -> Result<()> {
        // Request termination
        let mut cb = self.read_control_block();
        cb.should_terminate = 1;
        self.write_control_block(&cb);

        *self.state.write() = KernelState::Terminating;

        // Mark as terminated
        cb.has_terminated = 1;
        self.write_control_block(&cb);

        *self.state.write() = KernelState::Terminated;
        self.terminate_notify.notify_waiters();

        tracing::info!(kernel_id = %self.id, "Metal kernel terminated");
        Ok(())
    }

    async fn send_envelope(&self, envelope: MessageEnvelope) -> Result<()> {
        let state = *self.state.read();
        if state != KernelState::Active {
            return Err(RingKernelError::KernelNotActive(self.id.to_string()));
        }

        // Enqueue to input queue
        self.input_queue.enqueue(&envelope)?;
        self.message_counter.fetch_add(1, Ordering::Relaxed);

        // For event-driven mode, dispatch compute shader after each message
        if self.options.mode == KernelMode::EventDriven {
            if self.pipeline.is_some() {
                self.dispatch(self.options.block_size as u64)?;
            }
        }

        Ok(())
    }

    async fn receive(&self) -> Result<MessageEnvelope> {
        loop {
            if let Some(envelope) = self.output_queue.try_dequeue() {
                return Ok(envelope);
            }

            if *self.state.read() == KernelState::Terminated {
                return Err(RingKernelError::QueueEmpty);
            }

            tokio::task::yield_now().await;
        }
    }

    async fn receive_timeout(&self, timeout: Duration) -> Result<MessageEnvelope> {
        match tokio::time::timeout(timeout, self.receive()).await {
            Ok(result) => result,
            Err(_) => Err(RingKernelError::Timeout(timeout)),
        }
    }

    fn try_receive(&self) -> Result<MessageEnvelope> {
        self.output_queue
            .try_dequeue()
            .ok_or(RingKernelError::QueueEmpty)
    }

    async fn receive_correlated(
        &self,
        correlation: CorrelationId,
        timeout: Duration,
    ) -> Result<MessageEnvelope> {
        let start = Instant::now();
        loop {
            match self.try_receive() {
                Ok(envelope) => {
                    if envelope.header.correlation_id == correlation {
                        return Ok(envelope);
                    }
                }
                Err(RingKernelError::QueueEmpty) => {
                    if start.elapsed() >= timeout {
                        return Err(RingKernelError::Timeout(timeout));
                    }
                    tokio::task::yield_now().await;
                }
                Err(e) => return Err(e),
            }
        }
    }

    async fn wait(&self) -> Result<()> {
        self.terminate_notify.notified().await;
        Ok(())
    }
}
