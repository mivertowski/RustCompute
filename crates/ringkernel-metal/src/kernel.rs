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
// K2K MESSAGING STRUCTURES
// ============================================================================

/// K2K inbox header for Metal (64 bytes).
///
/// Each threadgroup has an inbox for receiving messages from other threadgroups.
/// This structure manages the inbox state.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct MetalK2KInboxHeader {
    /// Message count in inbox.
    pub message_count: u32,
    /// Maximum messages.
    pub max_messages: u32,
    /// Head index (next to read).
    pub head: u32,
    /// Tail index (next to write).
    pub tail: u32,
    /// Source threadgroup ID of last message.
    pub last_source: u32,
    /// Lock for thread-safe access (0 = unlocked, 1 = locked).
    pub lock: u32,
    /// Sequence number for ordering.
    pub sequence: u32,
    /// Reserved for alignment.
    pub _reserved: [u32; 9],
}

impl MetalK2KInboxHeader {
    /// Try to acquire the lock.
    pub fn try_lock(&mut self) -> bool {
        if self.lock == 0 {
            self.lock = 1;
            true
        } else {
            false
        }
    }

    /// Release the lock.
    pub fn unlock(&mut self) {
        self.lock = 0;
    }

    /// Check if inbox has messages.
    pub fn has_messages(&self) -> bool {
        self.message_count > 0
    }

    /// Check if inbox is full.
    pub fn is_full(&self) -> bool {
        self.message_count >= self.max_messages
    }
}

/// K2K route entry for Metal (32 bytes).
///
/// Maps a destination threadgroup to its inbox location.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct MetalK2KRouteEntry {
    /// Destination threadgroup ID.
    pub dest_threadgroup: u32,
    /// Inbox buffer offset.
    pub inbox_offset: u32,
    /// Whether this route is active.
    pub is_active: u32,
    /// Number of hops (for multi-hop routing).
    pub hops: u32,
    /// Bandwidth hint (messages per dispatch).
    pub bandwidth_hint: u32,
    /// Priority level (0 = highest).
    pub priority: u32,
    /// Reserved for alignment.
    pub _reserved: [u32; 2],
}

impl MetalK2KRouteEntry {
    /// Create a new route entry.
    pub fn new(dest: u32, offset: u32) -> Self {
        Self {
            dest_threadgroup: dest,
            inbox_offset: offset,
            is_active: 1,
            hops: 1,
            bandwidth_hint: 16,
            priority: 0,
            _reserved: [0; 2],
        }
    }
}

/// K2K routing table for Metal.
///
/// Contains routes to neighboring threadgroups for halo exchange patterns.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MetalK2KRoutingTable {
    /// This threadgroup's ID.
    pub self_id: u32,
    /// Number of active routes.
    pub route_count: u32,
    /// Grid dimensions (for neighbor calculation).
    pub grid_dim_x: u32,
    /// Grid dimensions.
    pub grid_dim_y: u32,
    /// Grid dimensions.
    pub grid_dim_z: u32,
    /// Reserved for alignment.
    pub _reserved: [u32; 3],
    /// Routes to neighbors (max 26 for 3D Moore neighborhood).
    pub routes: [MetalK2KRouteEntry; 26],
}

impl Default for MetalK2KRoutingTable {
    fn default() -> Self {
        Self {
            self_id: 0,
            route_count: 0,
            grid_dim_x: 1,
            grid_dim_y: 1,
            grid_dim_z: 1,
            _reserved: [0; 3],
            routes: [MetalK2KRouteEntry::default(); 26],
        }
    }
}

impl MetalK2KRoutingTable {
    /// Create a 2D 4-neighbor routing table (von Neumann neighborhood).
    pub fn new_2d_4neighbor(self_id: u32, grid_x: u32, grid_y: u32, inbox_size: u32) -> Self {
        let mut table = Self {
            self_id,
            grid_dim_x: grid_x,
            grid_dim_y: grid_y,
            grid_dim_z: 1,
            ..Default::default()
        };

        let x = self_id % grid_x;
        let y = self_id / grid_x;
        let mut count = 0;

        // North
        if y > 0 {
            let neighbor = (y - 1) * grid_x + x;
            table.routes[count] = MetalK2KRouteEntry::new(neighbor, neighbor * inbox_size);
            count += 1;
        }
        // South
        if y < grid_y - 1 {
            let neighbor = (y + 1) * grid_x + x;
            table.routes[count] = MetalK2KRouteEntry::new(neighbor, neighbor * inbox_size);
            count += 1;
        }
        // West
        if x > 0 {
            let neighbor = y * grid_x + (x - 1);
            table.routes[count] = MetalK2KRouteEntry::new(neighbor, neighbor * inbox_size);
            count += 1;
        }
        // East
        if x < grid_x - 1 {
            let neighbor = y * grid_x + (x + 1);
            table.routes[count] = MetalK2KRouteEntry::new(neighbor, neighbor * inbox_size);
            count += 1;
        }

        table.route_count = count as u32;
        table
    }

    /// Create a 3D 6-neighbor routing table (von Neumann neighborhood).
    pub fn new_3d_6neighbor(
        self_id: u32,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        inbox_size: u32,
    ) -> Self {
        let mut table = Self {
            self_id,
            grid_dim_x: grid_x,
            grid_dim_y: grid_y,
            grid_dim_z: grid_z,
            ..Default::default()
        };

        let z = self_id / (grid_x * grid_y);
        let rem = self_id % (grid_x * grid_y);
        let y = rem / grid_x;
        let x = rem % grid_x;
        let mut count = 0;

        // -X
        if x > 0 {
            let neighbor = z * (grid_x * grid_y) + y * grid_x + (x - 1);
            table.routes[count] = MetalK2KRouteEntry::new(neighbor, neighbor * inbox_size);
            count += 1;
        }
        // +X
        if x < grid_x - 1 {
            let neighbor = z * (grid_x * grid_y) + y * grid_x + (x + 1);
            table.routes[count] = MetalK2KRouteEntry::new(neighbor, neighbor * inbox_size);
            count += 1;
        }
        // -Y
        if y > 0 {
            let neighbor = z * (grid_x * grid_y) + (y - 1) * grid_x + x;
            table.routes[count] = MetalK2KRouteEntry::new(neighbor, neighbor * inbox_size);
            count += 1;
        }
        // +Y
        if y < grid_y - 1 {
            let neighbor = z * (grid_x * grid_y) + (y + 1) * grid_x + x;
            table.routes[count] = MetalK2KRouteEntry::new(neighbor, neighbor * inbox_size);
            count += 1;
        }
        // -Z
        if z > 0 {
            let neighbor = (z - 1) * (grid_x * grid_y) + y * grid_x + x;
            table.routes[count] = MetalK2KRouteEntry::new(neighbor, neighbor * inbox_size);
            count += 1;
        }
        // +Z
        if z < grid_z - 1 {
            let neighbor = (z + 1) * (grid_x * grid_y) + y * grid_x + x;
            table.routes[count] = MetalK2KRouteEntry::new(neighbor, neighbor * inbox_size);
            count += 1;
        }

        table.route_count = count as u32;
        table
    }

    /// Get route to a specific neighbor.
    pub fn get_route(&self, dest: u32) -> Option<&MetalK2KRouteEntry> {
        self.routes[..self.route_count as usize]
            .iter()
            .find(|r| r.dest_threadgroup == dest && r.is_active != 0)
    }
}

/// Halo exchange message for stencil computations.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct MetalHaloMessage {
    /// Source threadgroup.
    pub source: u32,
    /// Direction index (0-5 for 3D, 0-3 for 2D).
    pub direction: u32,
    /// Halo width.
    pub width: u32,
    /// Halo height.
    pub height: u32,
    /// Halo depth (1 for 2D).
    pub depth: u32,
    /// Data type size in bytes.
    pub element_size: u32,
    /// Sequence number.
    pub sequence: u32,
    /// Flags.
    pub flags: u32,
}

impl MetalHaloMessage {
    /// Create a new halo message for 2D.
    pub fn new_2d(source: u32, direction: u32, width: u32, height: u32) -> Self {
        Self {
            source,
            direction,
            width,
            height,
            depth: 1,
            element_size: 4, // f32
            sequence: 0,
            flags: 0,
        }
    }

    /// Create a new halo message for 3D.
    pub fn new_3d(source: u32, direction: u32, width: u32, height: u32, depth: u32) -> Self {
        Self {
            source,
            direction,
            width,
            height,
            depth,
            element_size: 4, // f32
            sequence: 0,
            flags: 0,
        }
    }

    /// Calculate payload size in bytes.
    pub fn payload_size(&self) -> usize {
        (self.width as usize)
            * (self.height as usize)
            * (self.depth as usize)
            * (self.element_size as usize)
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

// ============================================================================
// HALO EXCHANGE MANAGER
// ============================================================================

/// Configuration for halo exchange.
#[derive(Debug, Clone)]
pub struct HaloExchangeConfig {
    /// Grid dimensions (number of tiles).
    pub grid_dims: (u32, u32, u32),
    /// Tile dimensions.
    pub tile_dims: (u32, u32, u32),
    /// Halo width.
    pub halo_size: u32,
    /// Maximum messages per inbox.
    pub max_messages: u32,
}

impl Default for HaloExchangeConfig {
    fn default() -> Self {
        Self {
            grid_dims: (4, 4, 1),
            tile_dims: (64, 64, 1),
            halo_size: 1,
            max_messages: 16,
        }
    }
}

impl HaloExchangeConfig {
    /// Create a 2D configuration.
    pub fn new_2d(grid_x: u32, grid_y: u32, tile_w: u32, tile_h: u32, halo: u32) -> Self {
        Self {
            grid_dims: (grid_x, grid_y, 1),
            tile_dims: (tile_w, tile_h, 1),
            halo_size: halo,
            max_messages: 16,
        }
    }

    /// Create a 3D configuration.
    pub fn new_3d(
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        tile_w: u32,
        tile_h: u32,
        tile_d: u32,
        halo: u32,
    ) -> Self {
        Self {
            grid_dims: (grid_x, grid_y, grid_z),
            tile_dims: (tile_w, tile_h, tile_d),
            halo_size: halo,
            max_messages: 16,
        }
    }

    /// Calculate total number of tiles.
    pub fn total_tiles(&self) -> u32 {
        self.grid_dims.0 * self.grid_dims.1 * self.grid_dims.2
    }

    /// Calculate inbox size per tile.
    pub fn inbox_size(&self) -> usize {
        // Header (64) + max_messages * (header(32) + max_payload)
        let max_payload = self.tile_dims.0 * self.halo_size * 4; // One row
        64 + self.max_messages as usize * (32 + max_payload as usize)
    }
}

/// Manager for K2K halo exchange operations.
///
/// This struct manages the buffers and dispatch for halo exchange
/// between neighboring tiles in a grid decomposition.
pub struct MetalHaloExchange {
    /// Configuration.
    config: HaloExchangeConfig,
    /// Routing tables for each tile.
    routing_tables: Vec<MetalK2KRoutingTable>,
    /// Inbox buffer (shared between all tiles).
    inbox_buffer: Option<MetalBuffer>,
    /// Exchange pipeline.
    exchange_pipeline: Option<metal::ComputePipelineState>,
    /// Apply pipeline.
    apply_pipeline: Option<metal::ComputePipelineState>,
    /// Command queue.
    command_queue: Option<metal::CommandQueue>,
    /// Statistics: total exchanges performed.
    total_exchanges: AtomicU64,
    /// Statistics: total messages sent.
    total_messages_sent: AtomicU64,
}

impl MetalHaloExchange {
    /// Create a new halo exchange manager.
    pub fn new(config: HaloExchangeConfig) -> Self {
        let total_tiles = config.total_tiles();
        let inbox_size = config.inbox_size() as u32;

        // Build routing tables for each tile
        let mut routing_tables = Vec::with_capacity(total_tiles as usize);

        if config.grid_dims.2 == 1 {
            // 2D grid
            for tile_id in 0..total_tiles {
                let table = MetalK2KRoutingTable::new_2d_4neighbor(
                    tile_id,
                    config.grid_dims.0,
                    config.grid_dims.1,
                    inbox_size,
                );
                routing_tables.push(table);
            }
        } else {
            // 3D grid
            for tile_id in 0..total_tiles {
                let table = MetalK2KRoutingTable::new_3d_6neighbor(
                    tile_id,
                    config.grid_dims.0,
                    config.grid_dims.1,
                    config.grid_dims.2,
                    inbox_size,
                );
                routing_tables.push(table);
            }
        }

        Self {
            config,
            routing_tables,
            inbox_buffer: None,
            exchange_pipeline: None,
            apply_pipeline: None,
            command_queue: None,
            total_exchanges: AtomicU64::new(0),
            total_messages_sent: AtomicU64::new(0),
        }
    }

    /// Initialize buffers and compile shaders.
    pub fn initialize(&mut self, device: &Device) -> Result<()> {
        // Allocate inbox buffer
        let total_inbox_size = self.config.total_tiles() as usize * self.config.inbox_size();
        self.inbox_buffer = Some(MetalBuffer::new(device, total_inbox_size)?);

        // Initialize inbox headers
        self.initialize_inboxes()?;

        // Create command queue
        self.command_queue = Some(device.new_command_queue());

        // Compile halo exchange shaders
        let compile_options = metal::CompileOptions::new();
        let msl_source = crate::K2K_HALO_EXCHANGE_MSL_TEMPLATE;

        let library = device
            .new_library_with_source(msl_source, &compile_options)
            .map_err(|e| RingKernelError::CompilationFailed(format!("K2K MSL compile: {}", e)))?;

        // Create exchange pipeline
        if let Ok(func) = library.get_function("k2k_halo_exchange", None) {
            self.exchange_pipeline = Some(
                device
                    .new_compute_pipeline_state_with_function(&func)
                    .map_err(|e| RingKernelError::CompilationFailed(e.to_string()))?,
            );
        }

        // Create apply pipeline
        if let Ok(func) = library.get_function("k2k_halo_apply", None) {
            self.apply_pipeline = Some(
                device
                    .new_compute_pipeline_state_with_function(&func)
                    .map_err(|e| RingKernelError::CompilationFailed(e.to_string()))?,
            );
        }

        tracing::info!(
            "Metal K2K halo exchange initialized: {}x{}x{} grid, {} tiles",
            self.config.grid_dims.0,
            self.config.grid_dims.1,
            self.config.grid_dims.2,
            self.config.total_tiles()
        );

        Ok(())
    }

    /// Initialize all inbox headers.
    fn initialize_inboxes(&self) -> Result<()> {
        let inbox_buffer = self
            .inbox_buffer
            .as_ref()
            .ok_or_else(|| RingKernelError::BufferAllocationFailed(0))?;

        let ptr = inbox_buffer.buffer().contents() as *mut u8;

        for tile_id in 0..self.config.total_tiles() {
            let offset = tile_id as usize * self.config.inbox_size();

            // Write inbox header
            let header = MetalK2KInboxHeader {
                message_count: 0,
                max_messages: self.config.max_messages,
                head: 0,
                tail: 0,
                last_source: 0,
                lock: 0,
                sequence: 0,
                _reserved: [0; 9],
            };

            unsafe {
                std::ptr::write(ptr.add(offset) as *mut MetalK2KInboxHeader, header);
            }
        }

        Ok(())
    }

    /// Get the routing table for a tile.
    pub fn routing_table(&self, tile_id: u32) -> Option<&MetalK2KRoutingTable> {
        self.routing_tables.get(tile_id as usize)
    }

    /// Get configuration.
    pub fn config(&self) -> &HaloExchangeConfig {
        &self.config
    }

    /// Get statistics.
    pub fn stats(&self) -> HaloExchangeStats {
        HaloExchangeStats {
            total_exchanges: self.total_exchanges.load(Ordering::Relaxed),
            total_messages_sent: self.total_messages_sent.load(Ordering::Relaxed),
            tiles: self.config.total_tiles(),
            grid_dims: self.config.grid_dims,
        }
    }

    /// Perform a full halo exchange cycle.
    ///
    /// This dispatches the exchange kernel followed by the apply kernel.
    pub fn exchange(&self, tile_data_buffers: &[&MetalBuffer]) -> Result<()> {
        let command_queue = self
            .command_queue
            .as_ref()
            .ok_or_else(|| RingKernelError::LaunchFailed("Not initialized".to_string()))?;

        let exchange_pipeline = self.exchange_pipeline.as_ref().ok_or_else(|| {
            RingKernelError::LaunchFailed("Exchange pipeline not compiled".to_string())
        })?;

        let apply_pipeline = self.apply_pipeline.as_ref().ok_or_else(|| {
            RingKernelError::LaunchFailed("Apply pipeline not compiled".to_string())
        })?;

        let inbox_buffer = self
            .inbox_buffer
            .as_ref()
            .ok_or_else(|| RingKernelError::BufferAllocationFailed(0))?;

        // Phase 1: Exchange - all tiles send their halos
        let command_buffer = command_queue.new_command_buffer();

        for (tile_id, data_buffer) in tile_data_buffers.iter().enumerate() {
            if tile_id >= self.routing_tables.len() {
                break;
            }

            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(exchange_pipeline);

            // Would need to create routing table buffer for this tile
            // For now, just set the data buffer
            encoder.set_buffer(1, Some(inbox_buffer.buffer()), 0);
            encoder.set_buffer(2, Some(data_buffer.buffer()), 0);

            let threads = MTLSize::new(self.config.tile_dims.0 as u64, 1, 1);
            let threadgroup_size = MTLSize::new(64, 1, 1);
            encoder.dispatch_threads(threads, threadgroup_size);
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Phase 2: Apply - all tiles receive and apply halos
        let command_buffer = command_queue.new_command_buffer();

        for (tile_id, data_buffer) in tile_data_buffers.iter().enumerate() {
            if tile_id >= self.routing_tables.len() {
                break;
            }

            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(apply_pipeline);

            encoder.set_buffer(1, Some(inbox_buffer.buffer()), 0);
            encoder.set_buffer(2, Some(data_buffer.buffer()), 0);

            let threads = MTLSize::new(self.config.tile_dims.0 as u64, 1, 1);
            let threadgroup_size = MTLSize::new(64, 1, 1);
            encoder.dispatch_threads(threads, threadgroup_size);
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        self.total_exchanges.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Reset all inboxes (clear messages).
    pub fn reset(&self) -> Result<()> {
        self.initialize_inboxes()
    }
}

/// Statistics for halo exchange operations.
#[derive(Debug, Clone)]
pub struct HaloExchangeStats {
    /// Total exchange cycles performed.
    pub total_exchanges: u64,
    /// Total messages sent.
    pub total_messages_sent: u64,
    /// Number of tiles.
    pub tiles: u32,
    /// Grid dimensions.
    pub grid_dims: (u32, u32, u32),
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_block_size() {
        assert_eq!(std::mem::size_of::<MetalControlBlock>(), 128);
    }

    #[test]
    fn test_k2k_inbox_header_size() {
        assert_eq!(std::mem::size_of::<MetalK2KInboxHeader>(), 64);
    }

    #[test]
    fn test_k2k_route_entry_size() {
        assert_eq!(std::mem::size_of::<MetalK2KRouteEntry>(), 32);
    }

    #[test]
    fn test_halo_message_size() {
        assert_eq!(std::mem::size_of::<MetalHaloMessage>(), 32);
    }

    #[test]
    fn test_k2k_inbox_operations() {
        let mut header = MetalK2KInboxHeader {
            max_messages: 16,
            ..Default::default()
        };

        assert!(!header.has_messages());
        assert!(!header.is_full());

        header.message_count = 5;
        assert!(header.has_messages());
        assert!(!header.is_full());

        header.message_count = 16;
        assert!(header.is_full());
    }

    #[test]
    fn test_k2k_inbox_lock() {
        let mut header = MetalK2KInboxHeader::default();

        assert!(header.try_lock());
        assert!(!header.try_lock()); // Already locked

        header.unlock();
        assert!(header.try_lock()); // Can lock again
    }

    #[test]
    fn test_k2k_route_entry_new() {
        let entry = MetalK2KRouteEntry::new(5, 1024);

        assert_eq!(entry.dest_threadgroup, 5);
        assert_eq!(entry.inbox_offset, 1024);
        assert_eq!(entry.is_active, 1);
        assert_eq!(entry.hops, 1);
    }

    #[test]
    fn test_routing_table_2d_4neighbor_center() {
        // 3x3 grid, center cell (id=4)
        let table = MetalK2KRoutingTable::new_2d_4neighbor(4, 3, 3, 64);

        assert_eq!(table.self_id, 4);
        assert_eq!(table.route_count, 4); // All 4 neighbors exist

        // Check neighbors: north=1, south=7, west=3, east=5
        assert!(table.get_route(1).is_some()); // North
        assert!(table.get_route(7).is_some()); // South
        assert!(table.get_route(3).is_some()); // West
        assert!(table.get_route(5).is_some()); // East
    }

    #[test]
    fn test_routing_table_2d_4neighbor_corner() {
        // 3x3 grid, top-left corner (id=0)
        let table = MetalK2KRoutingTable::new_2d_4neighbor(0, 3, 3, 64);

        assert_eq!(table.route_count, 2); // Only south and east
        assert!(table.get_route(3).is_some()); // South
        assert!(table.get_route(1).is_some()); // East
    }

    #[test]
    fn test_routing_table_3d_6neighbor_center() {
        // 3x3x3 grid, center cell (id=13)
        let table = MetalK2KRoutingTable::new_3d_6neighbor(13, 3, 3, 3, 64);

        assert_eq!(table.self_id, 13);
        assert_eq!(table.route_count, 6); // All 6 neighbors exist
    }

    #[test]
    fn test_routing_table_3d_6neighbor_corner() {
        // 3x3x3 grid, origin corner (id=0)
        let table = MetalK2KRoutingTable::new_3d_6neighbor(0, 3, 3, 3, 64);

        assert_eq!(table.route_count, 3); // Only +X, +Y, +Z
    }

    #[test]
    fn test_halo_message_2d() {
        let msg = MetalHaloMessage::new_2d(5, 0, 16, 1);

        assert_eq!(msg.source, 5);
        assert_eq!(msg.direction, 0);
        assert_eq!(msg.width, 16);
        assert_eq!(msg.height, 1);
        assert_eq!(msg.depth, 1);
        assert_eq!(msg.payload_size(), 64); // 16 * 1 * 1 * 4
    }

    #[test]
    fn test_halo_message_3d() {
        let msg = MetalHaloMessage::new_3d(10, 2, 8, 8, 1);

        assert_eq!(msg.source, 10);
        assert_eq!(msg.direction, 2);
        assert_eq!(msg.payload_size(), 256); // 8 * 8 * 1 * 4
    }

    #[test]
    fn test_control_block_queue_size() {
        let mut cb = MetalControlBlock::default();

        // Empty queue
        assert_eq!(cb.input_queue_size(), 0);

        // Add some messages
        cb.input_tail_lo = 10;
        assert_eq!(cb.input_queue_size(), 10);

        // Read some
        cb.input_head_lo = 5;
        assert_eq!(cb.input_queue_size(), 5);
    }

    #[test]
    fn test_control_block_messages_processed() {
        let mut cb = MetalControlBlock::default();

        cb.messages_processed_lo = 0xFFFFFFFF;
        cb.messages_processed_hi = 0x1;

        // Should be 0x1_FFFFFFFF
        assert_eq!(cb.messages_processed(), 0x1_FFFFFFFF);
    }

    // ========================================================================
    // HALO EXCHANGE TESTS
    // ========================================================================

    #[test]
    fn test_halo_exchange_config_default() {
        let config = HaloExchangeConfig::default();

        assert_eq!(config.grid_dims, (4, 4, 1));
        assert_eq!(config.tile_dims, (64, 64, 1));
        assert_eq!(config.halo_size, 1);
        assert_eq!(config.total_tiles(), 16);
    }

    #[test]
    fn test_halo_exchange_config_2d() {
        let config = HaloExchangeConfig::new_2d(8, 8, 32, 32, 2);

        assert_eq!(config.grid_dims, (8, 8, 1));
        assert_eq!(config.tile_dims, (32, 32, 1));
        assert_eq!(config.halo_size, 2);
        assert_eq!(config.total_tiles(), 64);
    }

    #[test]
    fn test_halo_exchange_config_3d() {
        let config = HaloExchangeConfig::new_3d(4, 4, 4, 16, 16, 16, 1);

        assert_eq!(config.grid_dims, (4, 4, 4));
        assert_eq!(config.tile_dims, (16, 16, 16));
        assert_eq!(config.halo_size, 1);
        assert_eq!(config.total_tiles(), 64);
    }

    #[test]
    fn test_halo_exchange_config_inbox_size() {
        let config = HaloExchangeConfig::new_2d(4, 4, 64, 64, 1);

        // Inbox size: header(64) + max_messages(16) * (header(32) + payload(64*1*4=256))
        // = 64 + 16 * 288 = 64 + 4608 = 4672
        let expected = 64 + 16 * (32 + 256);
        assert_eq!(config.inbox_size(), expected);
    }

    #[test]
    fn test_halo_exchange_manager_creation_2d() {
        let config = HaloExchangeConfig::new_2d(4, 4, 32, 32, 1);
        let manager = MetalHaloExchange::new(config);

        assert_eq!(manager.routing_tables.len(), 16);

        // Check center tile (id=5) has 4 neighbors
        let table = manager.routing_table(5).unwrap();
        assert_eq!(table.route_count, 4);

        // Check corner tile (id=0) has 2 neighbors
        let table = manager.routing_table(0).unwrap();
        assert_eq!(table.route_count, 2);

        // Check edge tile (id=1) has 3 neighbors
        let table = manager.routing_table(1).unwrap();
        assert_eq!(table.route_count, 3);
    }

    #[test]
    fn test_halo_exchange_manager_creation_3d() {
        let config = HaloExchangeConfig::new_3d(3, 3, 3, 16, 16, 16, 1);
        let manager = MetalHaloExchange::new(config);

        assert_eq!(manager.routing_tables.len(), 27);

        // Check center tile (id=13) has 6 neighbors
        let table = manager.routing_table(13).unwrap();
        assert_eq!(table.route_count, 6);

        // Check corner tile (id=0) has 3 neighbors
        let table = manager.routing_table(0).unwrap();
        assert_eq!(table.route_count, 3);
    }

    #[test]
    fn test_halo_exchange_stats() {
        let config = HaloExchangeConfig::new_2d(4, 4, 32, 32, 1);
        let manager = MetalHaloExchange::new(config);

        let stats = manager.stats();
        assert_eq!(stats.tiles, 16);
        assert_eq!(stats.grid_dims, (4, 4, 1));
        assert_eq!(stats.total_exchanges, 0);
        assert_eq!(stats.total_messages_sent, 0);
    }

    #[test]
    fn test_halo_exchange_config_getter() {
        let config = HaloExchangeConfig::new_2d(8, 8, 64, 64, 2);
        let manager = MetalHaloExchange::new(config.clone());

        assert_eq!(manager.config().grid_dims, (8, 8, 1));
        assert_eq!(manager.config().halo_size, 2);
    }
}
