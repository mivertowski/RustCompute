//! Persistent GPU Actor Infrastructure
//!
//! This module provides truly persistent GPU actors that run for the entire
//! simulation lifetime, enabling:
//!
//! - Single kernel launch for entire simulation
//! - H2K (Host-to-Kernel) command messaging
//! - K2H (Kernel-to-Host) response messaging
//! - K2K (Kernel-to-Kernel) halo exchange on device memory
//! - Grid-wide synchronization via cooperative groups
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                         HOST (CPU)                              │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
//! │  │ Controller  │  │ Visualizer  │  │   Audio     │              │
//! │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
//! │         └────────────────┴────────────────┘                     │
//! │                          │                                      │
//! │  ┌───────────────────────▼───────────────────────────────────┐  │
//! │  │          MAPPED MEMORY (CPU + GPU visible)                │  │
//! │  │  [ControlBlock] [H2K Queue] [K2H Queue] [Progress]        │  │
//! │  └───────────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────┘
//!                              │ PCIe (commands only)
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    GPU (PERSISTENT KERNEL)                      │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │                 COORDINATOR (Block 0)                    │   │
//! │  │  • Process H2K commands • Send K2H responses            │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! │                          │ grid.sync()                          │
//! │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
//! │  │Block 1 │ │Block 2 │ │Block 3 │ │  ...   │ │Block N │       │
//! │  │ Actor  │ │ Actor  │ │ Actor  │ │        │ │ Actor  │       │
//! │  └───┬────┘ └───┬────┘ └───┬────┘ └────────┘ └───┬────┘       │
//! │      └──────────┴──────────┼────────────────────┘              │
//! │                    K2K HALO EXCHANGE                            │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │  [Halo Buffers] - 6 faces × N blocks × 2 (ping-pong)    │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │  [Pressure A] [Pressure B] - Double-buffered simulation │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use cudarc::driver::sys as cuda_sys;

use ringkernel_core::error::{Result, RingKernelError};
use ringkernel_core::memory::GpuBuffer;

use crate::device::CudaDevice;
use crate::driver_api::{CooperativeLaunchConfig, DirectCooperativeKernel, KernelParams};
use crate::memory::CudaBuffer;

// ============================================================================
// MAPPED MEMORY (CPU + GPU visible)
// ============================================================================

/// CUDA mapped memory buffer visible to both CPU and GPU.
///
/// Uses `cudaHostAllocMapped` to create memory that:
/// - Can be read/written by CPU without explicit transfers
/// - Can be read/written by GPU via device pointer
/// - Provides cache-coherent access for lock-free communication
///
/// # Memory Model
///
/// - CPU writes are visible to GPU after `__threadfence_system()` on GPU
/// - GPU writes are visible to CPU after `cudaDeviceSynchronize()` or polling
/// - For lock-free queues, use atomic operations and memory fences
pub struct CudaMappedBuffer<T: Copy + Send + Sync> {
    /// Host pointer (CPU-accessible).
    host_ptr: *mut T,
    /// Device pointer (GPU-accessible).
    device_ptr: u64,
    /// Number of elements.
    len: usize,
    /// Parent device.
    device: CudaDevice,
    /// Marker for T.
    _marker: PhantomData<T>,
}

// Safety: CudaMappedBuffer uses pinned memory that is accessible from both
// CPU and GPU. Access is synchronized via atomic operations and fences.
unsafe impl<T: Copy + Send + Sync> Send for CudaMappedBuffer<T> {}
unsafe impl<T: Copy + Send + Sync> Sync for CudaMappedBuffer<T> {}

impl<T: Copy + Send + Sync> CudaMappedBuffer<T> {
    /// Allocate mapped memory visible to both CPU and GPU.
    pub fn new(device: &CudaDevice, len: usize) -> Result<Self> {
        if len == 0 {
            return Err(RingKernelError::AllocationFailed {
                size: 0,
                reason: "Cannot allocate zero-length mapped buffer".to_string(),
            });
        }

        // Ensure device context is active
        let _ = device.inner();

        let size_bytes = len * std::mem::size_of::<T>();
        let mut host_ptr: *mut c_void = ptr::null_mut();

        // Allocate pinned mapped memory
        unsafe {
            let result = cuda_sys::lib().cuMemHostAlloc(
                &mut host_ptr,
                size_bytes,
                cuda_sys::CU_MEMHOSTALLOC_DEVICEMAP | cuda_sys::CU_MEMHOSTALLOC_PORTABLE,
            );

            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                return Err(RingKernelError::AllocationFailed {
                    size: size_bytes,
                    reason: format!("cuMemHostAlloc failed: {:?}", result),
                });
            }
        }

        // Get device pointer for this mapped memory
        let mut device_ptr: u64 = 0;
        unsafe {
            let result = cuda_sys::lib().cuMemHostGetDevicePointer_v2(
                &mut device_ptr,
                host_ptr,
                0, // flags (must be 0)
            );

            if result != cuda_sys::CUresult::CUDA_SUCCESS {
                // Clean up host allocation
                let _ = cuda_sys::lib().cuMemFreeHost(host_ptr);
                return Err(RingKernelError::AllocationFailed {
                    size: size_bytes,
                    reason: format!("cuMemHostGetDevicePointer failed: {:?}", result),
                });
            }
        }

        // Zero-initialize
        unsafe {
            ptr::write_bytes(host_ptr as *mut u8, 0, size_bytes);
        }

        Ok(Self {
            host_ptr: host_ptr as *mut T,
            device_ptr,
            len,
            device: device.clone(),
            _marker: PhantomData,
        })
    }

    /// Get host pointer for CPU access.
    #[inline]
    pub fn host_ptr(&self) -> *mut T {
        self.host_ptr
    }

    /// Get device pointer for GPU access.
    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    /// Get number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get reference to element at index (CPU access).
    ///
    /// # Safety
    ///
    /// Caller must ensure no concurrent GPU writes to this element,
    /// or use atomic operations for synchronization.
    #[inline]
    pub unsafe fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            Some(&*self.host_ptr.add(index))
        } else {
            None
        }
    }

    /// Get mutable reference to element at index (CPU access).
    ///
    /// # Safety
    ///
    /// Caller must ensure no concurrent GPU access to this element,
    /// or use atomic operations for synchronization.
    #[inline]
    #[allow(clippy::mut_from_ref)] // Intentional: raw pointer to GPU-mapped memory
    pub unsafe fn get_mut(&self, index: usize) -> Option<&mut T> {
        if index < self.len {
            Some(&mut *self.host_ptr.add(index))
        } else {
            None
        }
    }

    /// Read element at index (CPU access).
    ///
    /// Uses volatile read for visibility of GPU writes.
    #[inline]
    pub fn read(&self, index: usize) -> Option<T> {
        if index < self.len {
            unsafe { Some(ptr::read_volatile(self.host_ptr.add(index))) }
        } else {
            None
        }
    }

    /// Write element at index (CPU access).
    ///
    /// Uses volatile write for visibility to GPU.
    #[inline]
    pub fn write(&self, index: usize, value: T) -> bool {
        if index < self.len {
            unsafe {
                ptr::write_volatile(self.host_ptr.add(index), value);
            }
            true
        } else {
            false
        }
    }

    /// Get slice of all elements (CPU access).
    ///
    /// # Safety
    ///
    /// Caller must ensure no concurrent GPU writes.
    #[inline]
    pub unsafe fn as_slice(&self) -> &[T] {
        std::slice::from_raw_parts(self.host_ptr, self.len)
    }

    /// Get mutable slice of all elements (CPU access).
    ///
    /// # Safety
    ///
    /// Caller must ensure no concurrent GPU access.
    #[inline]
    #[allow(clippy::mut_from_ref)] // Intentional: raw pointer to GPU-mapped memory
    pub unsafe fn as_mut_slice(&self) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.host_ptr, self.len)
    }
}

impl<T: Copy + Send + Sync> Drop for CudaMappedBuffer<T> {
    fn drop(&mut self) {
        // Ensure device context is active before freeing
        let _ = self.device.inner();

        unsafe {
            let _ = cuda_sys::lib().cuMemFreeHost(self.host_ptr as *mut c_void);
        }
    }
}

// ============================================================================
// PERSISTENT CONTROL BLOCK
// ============================================================================

/// Control block for persistent GPU actor kernels.
///
/// This structure lives in mapped memory and controls the persistent kernel's
/// lifecycle. It uses cache-line alignment for efficient atomic access.
#[repr(C, align(256))]
#[derive(Debug, Clone, Copy)]
pub struct PersistentControlBlock {
    // === Lifecycle Control (written by host, read by GPU) ===
    /// Request kernel to terminate (host → GPU).
    pub should_terminate: u32,
    /// Padding for alignment.
    pub _pad0: u32,
    /// Number of steps remaining to process.
    pub steps_remaining: u64,

    // === State (written by GPU, read by host) ===
    /// Current simulation step.
    pub current_step: u64,
    /// Which pressure buffer is current (0 or 1).
    pub current_buffer: u32,
    /// Kernel has terminated.
    pub has_terminated: u32,
    /// Total energy in simulation (for monitoring).
    pub total_energy: f32,
    /// Padding for alignment.
    pub _pad1: u32,

    // === Configuration (set once at init) ===
    /// Grid dimensions (number of blocks).
    pub grid_dim: [u32; 3],
    /// Block dimensions (threads per block).
    pub block_dim: [u32; 3],
    /// Simulation grid size.
    pub sim_size: [u32; 3],
    /// Block tile size (cells per block dimension).
    pub tile_size: [u32; 3],

    // === Acoustic Parameters ===
    /// Speed of sound squared times dt squared (c² * dt²).
    pub c2_dt2: f32,
    /// Damping factor (0-1).
    pub damping: f32,
    /// Cell size in meters.
    pub cell_size: f32,
    /// Time step in seconds.
    pub dt: f32,

    // === Synchronization ===
    /// Barrier counter for software grid sync.
    pub barrier_counter: u32,
    /// Barrier generation for software grid sync.
    pub barrier_generation: u32,

    // === Statistics ===
    /// Messages processed count.
    pub messages_processed: u64,
    /// K2K messages sent.
    pub k2k_messages_sent: u64,
    /// K2K messages received.
    pub k2k_messages_received: u64,

    /// Reserved for future use (15 u64s = 120 bytes to reach 256 total).
    pub _reserved: [u64; 15],
}

impl Default for PersistentControlBlock {
    fn default() -> Self {
        Self {
            should_terminate: 0,
            _pad0: 0,
            steps_remaining: 0,
            current_step: 0,
            current_buffer: 0,
            has_terminated: 0,
            total_energy: 0.0,
            _pad1: 0,
            grid_dim: [1, 1, 1],
            block_dim: [256, 1, 1],
            sim_size: [64, 64, 64],
            tile_size: [8, 8, 8],
            c2_dt2: 0.25,
            damping: 0.999,
            cell_size: 0.01,
            dt: 0.00001,
            barrier_counter: 0,
            barrier_generation: 0,
            messages_processed: 0,
            k2k_messages_sent: 0,
            k2k_messages_received: 0,
            _reserved: [0; 15],
        }
    }
}

// ============================================================================
// H2K / K2H MESSAGE QUEUES
// ============================================================================

/// Command from host to kernel.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimCommand {
    /// No operation.
    Nop = 0,
    /// Run N simulation steps.
    RunSteps = 1,
    /// Pause simulation (stop processing steps).
    Pause = 2,
    /// Resume simulation.
    Resume = 3,
    /// Terminate kernel (graceful shutdown).
    Terminate = 4,
    /// Inject impulse at position.
    InjectImpulse = 5,
    /// Set continuous source.
    SetSource = 6,
    /// Query current progress.
    GetProgress = 7,
}

/// H2K message structure (host → kernel command).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct H2KMessage {
    /// Command type.
    pub cmd: u32,
    /// Command flags.
    pub flags: u32,
    /// Command ID for tracking.
    pub cmd_id: u64,
    /// Parameter 1 (usage depends on cmd).
    pub param1: u64,
    /// Parameter 2.
    pub param2: u32,
    /// Parameter 3.
    pub param3: u32,
    /// Parameter 4 (float).
    pub param4: f32,
    /// Parameter 5 (float).
    pub param5: f32,
    /// Reserved padding (3 u64s = 24 bytes to reach 64 total).
    pub _reserved: [u64; 3],
}

impl Default for H2KMessage {
    fn default() -> Self {
        Self {
            cmd: SimCommand::Nop as u32,
            flags: 0,
            cmd_id: 0,
            param1: 0,
            param2: 0,
            param3: 0,
            param4: 0.0,
            param5: 0.0,
            _reserved: [0; 3],
        }
    }
}

impl H2KMessage {
    /// Create a RunSteps command.
    pub fn run_steps(cmd_id: u64, count: u64) -> Self {
        Self {
            cmd: SimCommand::RunSteps as u32,
            cmd_id,
            param1: count,
            ..Default::default()
        }
    }

    /// Create a Terminate command.
    pub fn terminate(cmd_id: u64) -> Self {
        Self {
            cmd: SimCommand::Terminate as u32,
            cmd_id,
            ..Default::default()
        }
    }

    /// Create an InjectImpulse command.
    pub fn inject_impulse(cmd_id: u64, x: u32, y: u32, z: u32, amplitude: f32) -> Self {
        Self {
            cmd: SimCommand::InjectImpulse as u32,
            cmd_id,
            param1: ((x as u64) << 32) | (y as u64),
            param2: z,
            param4: amplitude,
            ..Default::default()
        }
    }

    /// Create a SetSource command.
    pub fn set_source(cmd_id: u64, x: u32, y: u32, z: u32, frequency: f32, amplitude: f32) -> Self {
        Self {
            cmd: SimCommand::SetSource as u32,
            cmd_id,
            param1: ((x as u64) << 32) | (y as u64),
            param2: z,
            param4: frequency,
            param5: amplitude,
            ..Default::default()
        }
    }
}

/// Response type from kernel to host.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResponseType {
    /// Acknowledgement of command.
    Ack = 0,
    /// Progress update.
    Progress = 1,
    /// Error occurred.
    Error = 2,
    /// Kernel terminated.
    Terminated = 3,
    /// Energy measurement.
    Energy = 4,
}

/// K2H message structure (kernel → host response).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct K2HMessage {
    /// Response type.
    pub resp_type: u32,
    /// Response flags.
    pub flags: u32,
    /// Command ID this responds to.
    pub cmd_id: u64,
    /// Current step number.
    pub step: u64,
    /// Steps remaining.
    pub steps_remaining: u64,
    /// Energy value.
    pub energy: f32,
    /// Error code (if error).
    pub error_code: u32,
    /// Reserved padding (3 u64s = 24 bytes to reach 64 total).
    pub _reserved: [u64; 3],
}

impl Default for K2HMessage {
    fn default() -> Self {
        Self {
            resp_type: ResponseType::Ack as u32,
            flags: 0,
            cmd_id: 0,
            step: 0,
            steps_remaining: 0,
            energy: 0.0,
            error_code: 0,
            _reserved: [0; 3],
        }
    }
}

/// Lock-free SPSC queue header for H2K/K2H communication (raw layout).
///
/// Uses raw u64 fields accessed atomically through volatile/fence operations.
/// Queue lives in mapped memory for CPU+GPU access.
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy, Default)]
pub struct SpscQueueHeader {
    /// Write position (incremented by producer).
    pub head: u64,
    /// Read position (incremented by consumer).
    pub tail: u64,
    /// Queue capacity (must be power of 2).
    pub capacity: u32,
    /// Capacity mask (capacity - 1).
    pub mask: u32,
    /// Padding to cache line.
    pub _padding: [u64; 12],
}

/// H2K message queue (host produces, kernel consumes).
pub struct H2KQueue {
    /// Queue header in mapped memory.
    header: CudaMappedBuffer<SpscQueueHeader>,
    /// Message slots in mapped memory.
    slots: CudaMappedBuffer<H2KMessage>,
    /// Next command ID.
    next_cmd_id: AtomicU64,
}

impl H2KQueue {
    /// Create a new H2K queue with given capacity.
    ///
    /// Capacity must be a power of 2.
    pub fn new(device: &CudaDevice, capacity: usize) -> Result<Self> {
        if !capacity.is_power_of_two() {
            return Err(RingKernelError::BackendError(
                "H2K queue capacity must be power of 2".to_string(),
            ));
        }

        let header = CudaMappedBuffer::new(device, 1)?;
        let slots = CudaMappedBuffer::new(device, capacity)?;

        // Initialize header
        let init_header = SpscQueueHeader {
            head: 0,
            tail: 0,
            capacity: capacity as u32,
            mask: (capacity - 1) as u32,
            _padding: [0; 12],
        };
        header.write(0, init_header);

        Ok(Self {
            header,
            slots,
            next_cmd_id: AtomicU64::new(1),
        })
    }

    /// Get header device pointer for kernel.
    pub fn header_device_ptr(&self) -> u64 {
        self.header.device_ptr()
    }

    /// Get slots device pointer for kernel.
    pub fn slots_device_ptr(&self) -> u64 {
        self.slots.device_ptr()
    }

    /// Send a command to the kernel (non-blocking).
    ///
    /// Returns the command ID on success, or error if queue is full.
    pub fn send(&self, mut msg: H2KMessage) -> Result<u64> {
        let cmd_id = self.next_cmd_id.fetch_add(1, Ordering::Relaxed);
        msg.cmd_id = cmd_id;

        // Read header with volatile for visibility
        let h = self.header.read(0).ok_or_else(|| {
            RingKernelError::BackendError("Failed to read H2K header".to_string())
        })?;

        let head = h.head;
        let tail = h.tail;
        let capacity = h.capacity as u64;

        // Check if full
        if head.wrapping_sub(tail) >= capacity {
            return Err(RingKernelError::QueueFull {
                capacity: capacity as usize,
            });
        }

        // Write message to slot
        let slot = (head & h.mask as u64) as usize;
        self.slots.write(slot, msg);

        // Memory barrier before publishing
        std::sync::atomic::fence(Ordering::Release);

        // Publish by incrementing head (volatile write)
        let new_header = SpscQueueHeader {
            head: head.wrapping_add(1),
            ..h
        };
        self.header.write(0, new_header);

        Ok(cmd_id)
    }

    /// Send a RunSteps command.
    pub fn run_steps(&self, count: u64) -> Result<u64> {
        self.send(H2KMessage::run_steps(0, count))
    }

    /// Send a Terminate command.
    pub fn terminate(&self) -> Result<u64> {
        self.send(H2KMessage::terminate(0))
    }

    /// Send an InjectImpulse command.
    pub fn inject_impulse(&self, x: u32, y: u32, z: u32, amplitude: f32) -> Result<u64> {
        self.send(H2KMessage::inject_impulse(0, x, y, z, amplitude))
    }
}

/// K2H message queue (kernel produces, host consumes).
pub struct K2HQueue {
    /// Queue header in mapped memory.
    header: CudaMappedBuffer<SpscQueueHeader>,
    /// Message slots in mapped memory.
    slots: CudaMappedBuffer<K2HMessage>,
}

impl K2HQueue {
    /// Create a new K2H queue with given capacity.
    pub fn new(device: &CudaDevice, capacity: usize) -> Result<Self> {
        if !capacity.is_power_of_two() {
            return Err(RingKernelError::BackendError(
                "K2H queue capacity must be power of 2".to_string(),
            ));
        }

        let header = CudaMappedBuffer::new(device, 1)?;
        let slots = CudaMappedBuffer::new(device, capacity)?;

        // Initialize header
        let init_header = SpscQueueHeader {
            head: 0,
            tail: 0,
            capacity: capacity as u32,
            mask: (capacity - 1) as u32,
            _padding: [0; 12],
        };
        header.write(0, init_header);

        Ok(Self { header, slots })
    }

    /// Get header device pointer for kernel.
    pub fn header_device_ptr(&self) -> u64 {
        self.header.device_ptr()
    }

    /// Get slots device pointer for kernel.
    pub fn slots_device_ptr(&self) -> u64 {
        self.slots.device_ptr()
    }

    /// Try to receive a message (non-blocking).
    pub fn try_recv(&self) -> Option<K2HMessage> {
        // Read header with volatile for visibility
        let h = self.header.read(0)?;

        let head = h.head;
        let tail = h.tail;

        // Check if empty
        if head == tail {
            return None;
        }

        // Read message from slot
        let slot = (tail & h.mask as u64) as usize;
        let msg = self.slots.read(slot)?;

        // Consume by incrementing tail (volatile write)
        let new_header = SpscQueueHeader {
            tail: tail.wrapping_add(1),
            ..h
        };
        self.header.write(0, new_header);

        Some(msg)
    }

    /// Receive all available messages.
    pub fn recv_all(&self) -> Vec<K2HMessage> {
        let mut messages = Vec::new();
        while let Some(msg) = self.try_recv() {
            messages.push(msg);
        }
        messages
    }

    /// Check if queue has messages.
    pub fn has_messages(&self) -> bool {
        if let Some(h) = self.header.read(0) {
            h.head != h.tail
        } else {
            false
        }
    }
}

// ============================================================================
// K2K HALO EXCHANGE STRUCTURES
// ============================================================================

/// Face indices for 3D halo exchange.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Face {
    /// +X face (east).
    PosX = 0,
    /// -X face (west).
    NegX = 1,
    /// +Y face (north).
    PosY = 2,
    /// -Y face (south).
    NegY = 3,
    /// +Z face (up).
    PosZ = 4,
    /// -Z face (down).
    NegZ = 5,
}

/// Neighbor block IDs for K2K routing.
///
/// -1 indicates no neighbor (boundary).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockNeighbors {
    /// +X neighbor block ID (-1 if boundary).
    pub pos_x: i32,
    /// -X neighbor block ID.
    pub neg_x: i32,
    /// +Y neighbor block ID.
    pub pos_y: i32,
    /// -Y neighbor block ID.
    pub neg_y: i32,
    /// +Z neighbor block ID.
    pub pos_z: i32,
    /// -Z neighbor block ID.
    pub neg_z: i32,
    /// Padding.
    pub _padding: [i32; 2],
}

/// K2K routing table entry.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct K2KRouteEntry {
    /// Block's neighbors.
    pub neighbors: BlockNeighbors,
    /// Block's position in grid (bx, by, bz).
    pub block_pos: [u32; 3],
    /// Padding.
    pub _padding: u32,
    /// Global cell offset for this block.
    pub cell_offset: [u32; 3],
    /// Padding.
    pub _padding2: u32,
}

/// Halo buffer configuration.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HaloConfig {
    /// Tile size per dimension (e.g., 8 for 8×8×8 blocks).
    pub tile_size: u32,
    /// Face size (tile_size × tile_size).
    pub face_size: u32,
    /// Number of blocks.
    pub num_blocks: u32,
    /// Stride between blocks in halo buffer.
    pub block_stride: u32,
    /// Stride between faces within a block.
    pub face_stride: u32,
    /// Stride for ping-pong buffer.
    pub pingpong_stride: u32,
}

impl HaloConfig {
    /// Create halo config for given tile size and block count.
    pub fn new(tile_size: u32, num_blocks: u32) -> Self {
        let face_size = tile_size * tile_size;
        let face_stride = face_size;
        let block_stride = 6 * face_stride * 2; // 6 faces × 2 ping-pong
        let pingpong_stride = 6 * face_stride;

        Self {
            tile_size,
            face_size,
            num_blocks,
            block_stride,
            face_stride,
            pingpong_stride,
        }
    }

    /// Total size in floats.
    pub fn total_floats(&self) -> usize {
        (self.num_blocks * self.block_stride) as usize
    }
}

// ============================================================================
// PERSISTENT SIMULATION
// ============================================================================

/// Configuration for persistent simulation.
#[derive(Debug, Clone)]
pub struct PersistentSimulationConfig {
    /// Simulation grid size (cells).
    pub grid_size: (usize, usize, usize),
    /// Block tile size (cells per block).
    pub tile_size: (usize, usize, usize),
    /// Speed of sound (m/s).
    pub speed_of_sound: f32,
    /// Cell size (meters).
    pub cell_size: f32,
    /// Damping factor (0-1, 1 = no damping).
    pub damping: f32,
    /// H2K queue capacity.
    pub h2k_capacity: usize,
    /// K2H queue capacity.
    pub k2h_capacity: usize,
    /// Use cooperative launch (requires grid to fit).
    pub use_cooperative: bool,
}

impl Default for PersistentSimulationConfig {
    fn default() -> Self {
        Self {
            grid_size: (64, 64, 64),
            tile_size: (8, 8, 8),
            speed_of_sound: 343.0,
            cell_size: 0.01,
            damping: 0.999,
            h2k_capacity: 256,
            k2h_capacity: 256,
            use_cooperative: true,
        }
    }
}

impl PersistentSimulationConfig {
    /// Create config for given grid size.
    pub fn new(width: usize, height: usize, depth: usize) -> Self {
        Self {
            grid_size: (width, height, depth),
            ..Default::default()
        }
    }

    /// Set tile size.
    pub fn with_tile_size(mut self, tx: usize, ty: usize, tz: usize) -> Self {
        self.tile_size = (tx, ty, tz);
        self
    }

    /// Set acoustic parameters.
    pub fn with_acoustics(mut self, speed: f32, cell_size: f32, damping: f32) -> Self {
        self.speed_of_sound = speed;
        self.cell_size = cell_size;
        self.damping = damping;
        self
    }

    /// Calculate number of blocks needed.
    pub fn num_blocks(&self) -> (usize, usize, usize) {
        let bx = self.grid_size.0.div_ceil(self.tile_size.0);
        let by = self.grid_size.1.div_ceil(self.tile_size.1);
        let bz = self.grid_size.2.div_ceil(self.tile_size.2);
        (bx, by, bz)
    }

    /// Calculate total block count.
    pub fn total_blocks(&self) -> usize {
        let (bx, by, bz) = self.num_blocks();
        bx * by * bz
    }

    /// Calculate time step for stability (CFL condition).
    pub fn calculate_dt(&self) -> f32 {
        // CFL: dt < dx / (c * sqrt(3)) for 3D
        let cfl_factor = 0.5; // Safety margin
        self.cell_size / (self.speed_of_sound * 1.732) * cfl_factor
    }

    /// Calculate c² × dt².
    pub fn calculate_c2_dt2(&self) -> f32 {
        let dt = self.calculate_dt();
        let c = self.speed_of_sound;
        (c * dt / self.cell_size).powi(2)
    }
}

/// Statistics from a running persistent simulation.
#[derive(Debug, Clone, Default)]
pub struct PersistentSimulationStats {
    /// Current simulation step.
    pub current_step: u64,
    /// Steps remaining to process.
    pub steps_remaining: u64,
    /// Total energy in simulation.
    pub total_energy: f32,
    /// Messages processed by kernel.
    pub messages_processed: u64,
    /// K2K messages sent.
    pub k2k_sent: u64,
    /// K2K messages received.
    pub k2k_received: u64,
    /// Kernel has terminated.
    pub has_terminated: bool,
}

/// Persistent GPU simulation with actor-based blocks.
///
/// This is the main entry point for the persistent GPU actor system.
pub struct PersistentSimulation {
    /// Configuration.
    config: PersistentSimulationConfig,
    /// CUDA device.
    device: CudaDevice,
    /// Control block in mapped memory.
    control: CudaMappedBuffer<PersistentControlBlock>,
    /// H2K command queue.
    h2k_queue: H2KQueue,
    /// K2H response queue.
    k2h_queue: K2HQueue,
    /// Pressure buffer A (device memory).
    pressure_a: CudaBuffer,
    /// Pressure buffer B (device memory).
    pressure_b: CudaBuffer,
    /// K2K routing table (device memory).
    k2k_routes: CudaBuffer,
    /// K2K halo buffers (device memory).
    halo_buffers: CudaBuffer,
    /// Halo configuration.
    halo_config: HaloConfig,
    /// Cooperative kernel (loaded when started).
    kernel: Option<DirectCooperativeKernel>,
    /// Kernel is running.
    is_running: AtomicBool,
}

impl PersistentSimulation {
    /// Create a new persistent simulation.
    pub fn new(device: &CudaDevice, config: PersistentSimulationConfig) -> Result<Self> {
        let (gx, gy, gz) = config.grid_size;
        let total_cells = gx * gy * gz;
        let cell_bytes = total_cells * std::mem::size_of::<f32>();

        let (bx, by, bz) = config.num_blocks();
        let total_blocks = bx * by * bz;

        // Allocate mapped memory for control block
        let control = CudaMappedBuffer::new(device, 1)?;

        // Initialize control block
        let dt = config.calculate_dt();
        let c2_dt2 = config.calculate_c2_dt2();
        let (tx, ty, tz) = config.tile_size;

        let cb = PersistentControlBlock {
            grid_dim: [bx as u32, by as u32, bz as u32],
            block_dim: [(tx * ty * tz) as u32, 1, 1],
            sim_size: [gx as u32, gy as u32, gz as u32],
            tile_size: [tx as u32, ty as u32, tz as u32],
            c2_dt2,
            damping: config.damping,
            cell_size: config.cell_size,
            dt,
            ..Default::default()
        };
        control.write(0, cb);

        // Allocate H2K/K2H queues
        let h2k_queue = H2KQueue::new(device, config.h2k_capacity)?;
        let k2h_queue = K2HQueue::new(device, config.k2h_capacity)?;

        // Allocate pressure buffers (device memory)
        let pressure_a = CudaBuffer::new(device, cell_bytes)?;
        let pressure_b = CudaBuffer::new(device, cell_bytes)?;

        // Zero-initialize pressure buffers
        let zeros = vec![0u8; cell_bytes];
        pressure_a.copy_from_host(&zeros)?;
        pressure_b.copy_from_host(&zeros)?;

        // Allocate K2K routing table
        let routes_size = total_blocks * std::mem::size_of::<K2KRouteEntry>();
        let k2k_routes = CudaBuffer::new(device, routes_size)?;

        // Initialize routing table
        Self::init_routing_table(device, &k2k_routes, bx, by, bz, tx, ty, tz)?;

        // Allocate halo buffers
        let halo_config = HaloConfig::new(tx as u32, total_blocks as u32);
        let halo_bytes = halo_config.total_floats() * std::mem::size_of::<f32>();
        let halo_buffers = CudaBuffer::new(device, halo_bytes)?;

        // Zero-initialize halo buffers
        let halo_zeros = vec![0u8; halo_bytes];
        halo_buffers.copy_from_host(&halo_zeros)?;

        Ok(Self {
            config,
            device: device.clone(),
            control,
            h2k_queue,
            k2h_queue,
            pressure_a,
            pressure_b,
            k2k_routes,
            halo_buffers,
            halo_config,
            kernel: None,
            is_running: AtomicBool::new(false),
        })
    }

    /// Initialize the K2K routing table.
    #[allow(clippy::too_many_arguments)]
    fn init_routing_table(
        _device: &CudaDevice,
        routes_buffer: &CudaBuffer,
        bx: usize,
        by: usize,
        bz: usize,
        tx: usize,
        ty: usize,
        tz: usize,
    ) -> Result<()> {
        let total_blocks = bx * by * bz;
        let mut routes = vec![K2KRouteEntry::default(); total_blocks];

        for block_z in 0..bz {
            for block_y in 0..by {
                for block_x in 0..bx {
                    let block_id = block_x + block_y * bx + block_z * bx * by;

                    let entry = &mut routes[block_id];
                    entry.block_pos = [block_x as u32, block_y as u32, block_z as u32];
                    entry.cell_offset = [
                        (block_x * tx) as u32,
                        (block_y * ty) as u32,
                        (block_z * tz) as u32,
                    ];

                    // Set neighbors (-1 for boundary)
                    entry.neighbors.pos_x = if block_x + 1 < bx {
                        (block_id + 1) as i32
                    } else {
                        -1
                    };
                    entry.neighbors.neg_x = if block_x > 0 {
                        (block_id - 1) as i32
                    } else {
                        -1
                    };
                    entry.neighbors.pos_y = if block_y + 1 < by {
                        (block_id + bx) as i32
                    } else {
                        -1
                    };
                    entry.neighbors.neg_y = if block_y > 0 {
                        (block_id - bx) as i32
                    } else {
                        -1
                    };
                    entry.neighbors.pos_z = if block_z + 1 < bz {
                        (block_id + bx * by) as i32
                    } else {
                        -1
                    };
                    entry.neighbors.neg_z = if block_z > 0 {
                        (block_id - bx * by) as i32
                    } else {
                        -1
                    };
                }
            }
        }

        // Upload to device
        let routes_bytes = unsafe {
            std::slice::from_raw_parts(
                routes.as_ptr() as *const u8,
                total_blocks * std::mem::size_of::<K2KRouteEntry>(),
            )
        };
        routes_buffer.copy_from_host(routes_bytes)?;

        Ok(())
    }

    /// Load and start the persistent kernel.
    ///
    /// The kernel will run until terminated via `shutdown()`.
    pub fn start(&mut self, ptx: &str, func_name: &str) -> Result<()> {
        if self.is_running.load(Ordering::Acquire) {
            return Err(RingKernelError::InvalidState {
                expected: "not running".to_string(),
                actual: "running".to_string(),
            });
        }

        let (bx, by, bz) = self.config.num_blocks();
        let total_blocks = bx * by * bz;
        let threads_per_block =
            self.config.tile_size.0 * self.config.tile_size.1 * self.config.tile_size.2;

        // Load cooperative kernel
        let kernel = DirectCooperativeKernel::from_ptx(
            &self.device,
            ptx,
            func_name,
            threads_per_block as u32,
        )?;

        // Check if grid fits cooperative limits
        let max_blocks = kernel.max_concurrent_blocks();
        if total_blocks as u32 > max_blocks && self.config.use_cooperative {
            return Err(RingKernelError::LaunchFailed(format!(
                "Grid size {} exceeds cooperative limit {}. Use software sync or smaller grid.",
                total_blocks, max_blocks
            )));
            // TODO: Fall back to software sync when not using cooperative mode
        }

        // Configure launch
        let config = CooperativeLaunchConfig {
            grid_dim: (bx as u32, by as u32, bz as u32),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0, // Shared memory allocated in kernel
        };

        // Build kernel parameters
        let mut params = PersistentKernelParams {
            control_ptr: self.control.device_ptr(),
            pressure_a_ptr: self.pressure_a.device_ptr(),
            pressure_b_ptr: self.pressure_b.device_ptr(),
            h2k_header_ptr: self.h2k_queue.header_device_ptr(),
            h2k_slots_ptr: self.h2k_queue.slots_device_ptr(),
            k2h_header_ptr: self.k2h_queue.header_device_ptr(),
            k2h_slots_ptr: self.k2h_queue.slots_device_ptr(),
            routes_ptr: self.k2k_routes.device_ptr(),
            halo_ptr: self.halo_buffers.device_ptr(),
        };

        // Launch kernel cooperatively
        unsafe {
            kernel.launch_cooperative_params(&config, &mut params)?;
        }

        self.kernel = Some(kernel);
        self.is_running.store(true, Ordering::Release);

        Ok(())
    }

    /// Check if the simulation is running.
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }

    /// Run N simulation steps.
    ///
    /// Writes directly to the control block's steps_remaining field
    /// for reliable communication (bypasses H2K queue due to visibility issues).
    pub fn run_steps(&self, count: u64) -> Result<u64> {
        // Write directly to control block for reliable communication
        if let Some(mut cb) = self.control.read(0) {
            cb.steps_remaining = cb.steps_remaining.saturating_add(count);
            self.control.write(0, cb);
            Ok(0) // Return dummy command ID since we're not using queue
        } else {
            Err(RingKernelError::BackendError(
                "Failed to read control block".to_string(),
            ))
        }
    }

    /// Inject an impulse at the given position.
    pub fn inject_impulse(&self, x: u32, y: u32, z: u32, amplitude: f32) -> Result<u64> {
        self.h2k_queue.inject_impulse(x, y, z, amplitude)
    }

    /// Poll for responses from the kernel.
    pub fn poll_responses(&self) -> Vec<K2HMessage> {
        self.k2h_queue.recv_all()
    }

    /// Get current statistics.
    pub fn stats(&self) -> PersistentSimulationStats {
        if let Some(cb) = self.control.read(0) {
            PersistentSimulationStats {
                current_step: cb.current_step,
                steps_remaining: cb.steps_remaining,
                total_energy: cb.total_energy,
                messages_processed: cb.messages_processed,
                k2k_sent: cb.k2k_messages_sent,
                k2k_received: cb.k2k_messages_received,
                has_terminated: cb.has_terminated != 0,
            }
        } else {
            PersistentSimulationStats::default()
        }
    }

    /// Read pressure field to host buffer.
    pub fn read_pressure(&self, output: &mut [f32]) -> Result<()> {
        let cb = self
            .control
            .read(0)
            .ok_or_else(|| RingKernelError::InvalidState {
                expected: "valid control block".to_string(),
                actual: "read failed".to_string(),
            })?;

        let buffer = if cb.current_buffer == 0 {
            &self.pressure_a
        } else {
            &self.pressure_b
        };

        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                output.as_mut_ptr() as *mut u8,
                std::mem::size_of_val(output),
            )
        };
        buffer.copy_to_host(bytes)
    }

    /// Gracefully shut down the simulation.
    pub fn shutdown(&mut self) -> Result<()> {
        if !self.is_running.load(Ordering::Acquire) {
            return Ok(());
        }

        // Set should_terminate directly in control block (dual-path shutdown)
        // This ensures the kernel sees the termination even if H2K queue has issues
        if let Some(mut cb) = self.control.read(0) {
            cb.should_terminate = 1;
            self.control.write(0, cb);
        }

        // Also send terminate command via H2K queue for completeness
        let _ = self.h2k_queue.terminate();

        // Wait for kernel to terminate
        let timeout = Duration::from_secs(5);
        let start = std::time::Instant::now();

        loop {
            let stats = self.stats();
            if stats.has_terminated {
                break;
            }

            if start.elapsed() > timeout {
                return Err(RingKernelError::Timeout(timeout));
            }

            std::thread::sleep(Duration::from_millis(1));
        }

        // Synchronize device
        if let Some(ref kernel) = self.kernel {
            kernel.synchronize()?;
        }

        self.is_running.store(false, Ordering::Release);
        Ok(())
    }

    /// Get the configuration.
    pub fn config(&self) -> &PersistentSimulationConfig {
        &self.config
    }

    /// Get halo configuration.
    pub fn halo_config(&self) -> &HaloConfig {
        &self.halo_config
    }

    /// Get device reference.
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }
}

impl Drop for PersistentSimulation {
    fn drop(&mut self) {
        // Try to shut down gracefully
        let _ = self.shutdown();
    }
}

/// Kernel parameters for persistent FDTD simulation.
#[repr(C)]
pub struct PersistentKernelParams {
    /// Control block pointer (mapped memory).
    pub control_ptr: u64,
    /// Pressure buffer A pointer.
    pub pressure_a_ptr: u64,
    /// Pressure buffer B pointer.
    pub pressure_b_ptr: u64,
    /// H2K queue header pointer.
    pub h2k_header_ptr: u64,
    /// H2K queue slots pointer.
    pub h2k_slots_ptr: u64,
    /// K2H queue header pointer.
    pub k2h_header_ptr: u64,
    /// K2H queue slots pointer.
    pub k2h_slots_ptr: u64,
    /// K2K routing table pointer.
    pub routes_ptr: u64,
    /// Halo buffers pointer.
    pub halo_ptr: u64,
}

impl KernelParams for PersistentKernelParams {
    fn as_param_ptrs(&mut self) -> Vec<*mut c_void> {
        vec![
            &mut self.control_ptr as *mut u64 as *mut c_void,
            &mut self.pressure_a_ptr as *mut u64 as *mut c_void,
            &mut self.pressure_b_ptr as *mut u64 as *mut c_void,
            &mut self.h2k_header_ptr as *mut u64 as *mut c_void,
            &mut self.h2k_slots_ptr as *mut u64 as *mut c_void,
            &mut self.k2h_header_ptr as *mut u64 as *mut c_void,
            &mut self.k2h_slots_ptr as *mut u64 as *mut c_void,
            &mut self.routes_ptr as *mut u64 as *mut c_void,
            &mut self.halo_ptr as *mut u64 as *mut c_void,
        ]
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_persistent_control_block_size() {
        assert_eq!(
            std::mem::size_of::<PersistentControlBlock>(),
            256,
            "PersistentControlBlock must be 256 bytes"
        );
    }

    #[test]
    fn test_h2k_message_size() {
        assert_eq!(
            std::mem::size_of::<H2KMessage>(),
            64,
            "H2KMessage must be 64 bytes"
        );
    }

    #[test]
    fn test_k2h_message_size() {
        assert_eq!(
            std::mem::size_of::<K2HMessage>(),
            64,
            "K2HMessage must be 64 bytes"
        );
    }

    #[test]
    fn test_spsc_queue_header_size() {
        assert_eq!(
            std::mem::size_of::<SpscQueueHeader>(),
            128,
            "SpscQueueHeader must be 128 bytes (cache line aligned)"
        );
    }

    #[test]
    fn test_halo_config() {
        let config = HaloConfig::new(8, 125); // 8×8×8 tiles, 5×5×5 blocks
        assert_eq!(config.tile_size, 8);
        assert_eq!(config.face_size, 64); // 8×8
        assert_eq!(config.num_blocks, 125);

        // Each block: 6 faces × 64 floats × 2 ping-pong = 768 floats
        assert_eq!(config.block_stride, 768);

        // Total: 125 blocks × 768 = 96000 floats
        assert_eq!(config.total_floats(), 96000);
    }

    #[test]
    fn test_simulation_config() {
        let config = PersistentSimulationConfig::new(64, 64, 64).with_tile_size(8, 8, 8);

        assert_eq!(config.num_blocks(), (8, 8, 8));
        assert_eq!(config.total_blocks(), 512);

        // CFL check
        let dt = config.calculate_dt();
        assert!(dt > 0.0);
        assert!(dt < config.cell_size / config.speed_of_sound);
    }

    #[test]
    fn test_h2k_message_constructors() {
        let run = H2KMessage::run_steps(42, 1000);
        assert_eq!(run.cmd, SimCommand::RunSteps as u32);
        assert_eq!(run.cmd_id, 42);
        assert_eq!(run.param1, 1000);

        let term = H2KMessage::terminate(99);
        assert_eq!(term.cmd, SimCommand::Terminate as u32);
        assert_eq!(term.cmd_id, 99);

        let impulse = H2KMessage::inject_impulse(1, 32, 32, 32, 1.5);
        assert_eq!(impulse.cmd, SimCommand::InjectImpulse as u32);
        assert_eq!(impulse.param2, 32); // z
        assert_eq!(impulse.param4, 1.5); // amplitude
    }

    #[test]
    fn test_kernel_params_layout() {
        let mut params = PersistentKernelParams {
            control_ptr: 0x1000,
            pressure_a_ptr: 0x2000,
            pressure_b_ptr: 0x3000,
            h2k_header_ptr: 0x4000,
            h2k_slots_ptr: 0x5000,
            k2h_header_ptr: 0x6000,
            k2h_slots_ptr: 0x7000,
            routes_ptr: 0x8000,
            halo_ptr: 0x9000,
        };

        let ptrs = params.as_param_ptrs();
        assert_eq!(ptrs.len(), 9);

        unsafe {
            assert_eq!(*(ptrs[0] as *const u64), 0x1000);
            assert_eq!(*(ptrs[8] as *const u64), 0x9000);
        }
    }

    // Hardware tests
    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_mapped_buffer_allocation() {
        let device = CudaDevice::new(0).expect("Failed to create device");
        let buffer: CudaMappedBuffer<u64> =
            CudaMappedBuffer::new(&device, 1024).expect("Failed to allocate mapped buffer");

        assert_eq!(buffer.len(), 1024);
        assert!(buffer.device_ptr() > 0);
        assert!(!buffer.host_ptr().is_null());

        // Test read/write
        buffer.write(0, 0xDEADBEEF);
        assert_eq!(buffer.read(0), Some(0xDEADBEEF));

        buffer.write(1023, 0xCAFEBABE);
        assert_eq!(buffer.read(1023), Some(0xCAFEBABE));
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_h2k_k2h_queues() {
        let device = CudaDevice::new(0).expect("Failed to create device");

        let h2k = H2KQueue::new(&device, 64).expect("Failed to create H2K queue");
        let k2h = K2HQueue::new(&device, 64).expect("Failed to create K2H queue");

        // Test H2K send
        let cmd_id = h2k.run_steps(100).expect("Failed to send");
        assert!(cmd_id > 0);

        // K2H should be empty (no kernel running)
        assert!(!k2h.has_messages());
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_persistent_simulation_creation() {
        let device = CudaDevice::new(0).expect("Failed to create device");

        let config = PersistentSimulationConfig::new(40, 40, 40) // Small grid for cooperative
            .with_tile_size(8, 8, 8);

        let sim = PersistentSimulation::new(&device, config).expect("Failed to create simulation");

        assert!(!sim.is_running());
        let stats = sim.stats();
        assert_eq!(stats.current_step, 0);
        assert!(!stats.has_terminated);
    }
}
