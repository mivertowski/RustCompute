//! GPU-side K2K (Kernel-to-Kernel) messaging structures.
//!
//! This module defines the GPU buffer structures for direct kernel-to-kernel
//! communication. K2K messaging allows kernels to communicate directly on the
//! GPU without host intervention, enabling efficient multi-kernel pipelines.
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │                         GPU Memory                              │
//! │                                                                  │
//! │  ┌─────────────────┐     ┌─────────────────┐                    │
//! │  │ Kernel A        │     │ Kernel B        │                    │
//! │  │                 │     │                 │                    │
//! │  │ ┌─────────────┐ │     │ ┌─────────────┐ │                    │
//! │  │ │ K2K Outbox  │─┼─────┼─│ K2K Inbox   │ │                    │
//! │  │ └─────────────┘ │     │ └─────────────┘ │                    │
//! │  │                 │     │                 │                    │
//! │  │ ┌─────────────┐ │     │ ┌─────────────┐ │                    │
//! │  │ │Route Table  │ │     │ │Route Table  │ │                    │
//! │  │ │ B→inbox_ptr │ │     │ │ A→inbox_ptr │ │                    │
//! │  │ └─────────────┘ │     │ └─────────────┘ │                    │
//! │  └─────────────────┘     └─────────────────┘                    │
//! └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # K2K Messaging Flow
//!
//! 1. Kernel A calls `k2k_send(target_id, msg)` via device intrinsics
//! 2. The intrinsic looks up target_id in the routing table
//! 3. The message is written directly to Kernel B's inbox
//! 4. Kernel B sees the message via `k2k_try_recv()` or `k2k_pending_count()`

use std::mem;

use ringkernel_core::error::{Result, RingKernelError};
use ringkernel_core::memory::GpuBuffer;

use crate::device::CudaDevice;
use crate::memory::CudaBuffer;

/// K2K inbox header stored on GPU.
///
/// This structure is placed at the beginning of the inbox buffer
/// and is accessed atomically by both sender and receiver kernels.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, Default)]
pub struct K2KInboxHeader {
    /// Head pointer (sender increments).
    pub head: u64,
    /// Tail pointer (receiver increments).
    pub tail: u64,
    /// Queue capacity (must be power of 2).
    pub capacity: u32,
    /// Mask for index wrapping (capacity - 1).
    pub mask: u32,
    /// Size of each message in bytes.
    pub msg_size: u32,
    /// Reserved for alignment.
    pub _padding: u32,
}

const _: () = assert!(mem::size_of::<K2KInboxHeader>() == 64);

impl K2KInboxHeader {
    /// Create a new inbox header.
    pub fn new(capacity: u32, msg_size: u32) -> Self {
        Self {
            head: 0,
            tail: 0,
            capacity,
            mask: capacity.saturating_sub(1),
            msg_size,
            _padding: 0,
        }
    }

    /// Check if queue is full.
    pub fn is_full(&self) -> bool {
        self.head.wrapping_sub(self.tail) >= self.capacity as u64
    }

    /// Check if queue is empty.
    pub fn is_empty(&self) -> bool {
        self.head == self.tail
    }

    /// Get current queue size.
    pub fn size(&self) -> u64 {
        self.head.wrapping_sub(self.tail)
    }
}

/// K2K routing table entry stored on GPU.
///
/// Each kernel has a routing table mapping destination kernel IDs
/// to their inbox device pointers.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct K2KRouteEntry {
    /// Target kernel ID (0 = unused slot).
    pub target_kernel_id: u64,
    /// Device pointer to target's inbox buffer.
    pub target_inbox: u64,
    /// Device pointer to target's inbox head (for atomic writes).
    pub target_head: u64,
    /// Device pointer to target's inbox tail (for checking full).
    pub target_tail: u64,
    /// Target inbox capacity.
    pub capacity: u32,
    /// Target inbox mask.
    pub mask: u32,
    /// Message size for this route.
    pub msg_size: u32,
    /// Reserved for alignment.
    pub _padding: u32,
}

const _: () = assert!(mem::size_of::<K2KRouteEntry>() == 48);

impl K2KRouteEntry {
    /// Check if this entry is unused.
    pub fn is_empty(&self) -> bool {
        self.target_kernel_id == 0
    }
}

/// K2K routing table on GPU.
///
/// Fixed-size array of route entries. Kernel code looks up
/// destination by linear scan (small tables) or hash (large tables).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct K2KRoutingTable<const N: usize> {
    /// Number of active routes.
    pub route_count: u32,
    /// Reserved for alignment.
    pub _padding: u32,
    /// Route entries.
    pub routes: [K2KRouteEntry; N],
}

impl<const N: usize> Default for K2KRoutingTable<N> {
    fn default() -> Self {
        Self {
            route_count: 0,
            _padding: 0,
            routes: [K2KRouteEntry::default(); N],
        }
    }
}

/// Maximum number of K2K routes per kernel.
pub const MAX_K2K_ROUTES: usize = 8;

/// Standard routing table type.
pub type StandardRoutingTable = K2KRoutingTable<MAX_K2K_ROUTES>;

/// CUDA K2K buffers for a single kernel.
///
/// Each kernel that participates in K2K messaging needs:
/// - An inbox to receive messages from other kernels
/// - A routing table to know where to send messages
/// - An optional outbox for staging (currently not used)
pub struct CudaK2KBuffers {
    /// Routing table buffer.
    routing_table: CudaBuffer,
    /// Inbox buffer (header + message slots).
    inbox: CudaBuffer,
    /// Inbox capacity.
    inbox_capacity: u32,
    /// Message size.
    msg_size: u32,
}

impl CudaK2KBuffers {
    /// Create new K2K buffers.
    ///
    /// # Arguments
    ///
    /// * `device` - CUDA device to allocate on
    /// * `inbox_capacity` - Number of message slots (must be power of 2)
    /// * `msg_size` - Size of each message in bytes
    /// * `max_routes` - Maximum number of K2K routes (default: 8)
    pub fn new(
        device: &CudaDevice,
        inbox_capacity: u32,
        msg_size: u32,
        max_routes: usize,
    ) -> Result<Self> {
        // Validate capacity is power of 2
        if !inbox_capacity.is_power_of_two() {
            return Err(RingKernelError::InvalidConfig(format!(
                "K2K inbox capacity must be power of 2, got {}",
                inbox_capacity
            )));
        }

        // Routing table size
        let routing_size = mem::size_of::<u32>() * 2 // route_count + padding
            + mem::size_of::<K2KRouteEntry>() * max_routes;

        // Inbox size: header + message slots
        let inbox_size =
            mem::size_of::<K2KInboxHeader>() + (msg_size as usize) * (inbox_capacity as usize);

        let routing_table = CudaBuffer::new(device, routing_size)?;
        let inbox = CudaBuffer::new(device, inbox_size)?;

        // Initialize routing table to zeros
        let zeros_routing = vec![0u8; routing_size];
        routing_table.copy_from_host_at(&zeros_routing, 0)?;

        // Initialize inbox header
        let header = K2KInboxHeader::new(inbox_capacity, msg_size);
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const K2KInboxHeader as *const u8,
                mem::size_of::<K2KInboxHeader>(),
            )
        };
        inbox.copy_from_host_at(header_bytes, 0)?;

        Ok(Self {
            routing_table,
            inbox,
            inbox_capacity,
            msg_size,
        })
    }

    /// Get routing table device pointer.
    pub fn routing_table_ptr(&self) -> u64 {
        self.routing_table.device_ptr()
    }

    /// Get inbox device pointer.
    pub fn inbox_ptr(&self) -> u64 {
        self.inbox.device_ptr()
    }

    /// Get inbox head pointer (for routing entries).
    pub fn inbox_head_ptr(&self) -> u64 {
        // head is at offset 0 in K2KInboxHeader
        self.inbox.device_ptr()
    }

    /// Get inbox tail pointer (for routing entries).
    pub fn inbox_tail_ptr(&self) -> u64 {
        // tail is at offset 8 in K2KInboxHeader
        self.inbox.device_ptr() + 8
    }

    /// Get inbox capacity.
    pub fn inbox_capacity(&self) -> u32 {
        self.inbox_capacity
    }

    /// Get message size.
    pub fn msg_size(&self) -> u32 {
        self.msg_size
    }

    /// Add a route to another kernel.
    ///
    /// # Arguments
    ///
    /// * `slot` - Route slot index (0..max_routes)
    /// * `target` - Target kernel's K2K buffers
    /// * `target_kernel_id` - Target kernel's numeric ID
    pub fn add_route(
        &self,
        slot: usize,
        target: &CudaK2KBuffers,
        target_kernel_id: u64,
    ) -> Result<()> {
        if slot >= MAX_K2K_ROUTES {
            return Err(RingKernelError::InvalidIndex(slot));
        }

        let entry = K2KRouteEntry {
            target_kernel_id,
            target_inbox: target.inbox_ptr(),
            target_head: target.inbox_head_ptr(),
            target_tail: target.inbox_tail_ptr(),
            capacity: target.inbox_capacity,
            mask: target.inbox_capacity.saturating_sub(1),
            msg_size: target.msg_size,
            _padding: 0,
        };

        // Calculate offset: skip route_count (u32) + padding (u32) + previous entries
        let entry_offset = 8 + slot * mem::size_of::<K2KRouteEntry>();

        let entry_bytes = unsafe {
            std::slice::from_raw_parts(
                &entry as *const K2KRouteEntry as *const u8,
                mem::size_of::<K2KRouteEntry>(),
            )
        };

        self.routing_table.copy_from_host_at(entry_bytes, entry_offset)?;

        // Update route count
        let new_count = (slot + 1) as u32;
        let count_bytes = new_count.to_le_bytes();
        self.routing_table.copy_from_host_at(&count_bytes, 0)?;

        Ok(())
    }

    /// Read inbox header from GPU.
    pub fn read_inbox_header(&self) -> Result<K2KInboxHeader> {
        let mut header_bytes = [0u8; mem::size_of::<K2KInboxHeader>()];
        self.inbox.copy_to_host_at(&mut header_bytes, 0)?;

        Ok(unsafe { std::ptr::read(header_bytes.as_ptr() as *const K2KInboxHeader) })
    }

    /// Get total memory usage.
    pub fn total_size(&self) -> usize {
        self.routing_table.size() + self.inbox.size()
    }
}

/// K2K connection manager for a set of kernels.
///
/// Tracks K2K buffers and connections between kernels.
pub struct K2KConnectionManager {
    /// K2K buffers indexed by kernel numeric ID.
    buffers: std::collections::HashMap<u64, CudaK2KBuffers>,
    /// Connection list: (source_id, target_id, slot).
    connections: Vec<(u64, u64, usize)>,
}

impl K2KConnectionManager {
    /// Create a new connection manager.
    pub fn new() -> Self {
        Self {
            buffers: std::collections::HashMap::new(),
            connections: Vec::new(),
        }
    }

    /// Register K2K buffers for a kernel.
    pub fn register(&mut self, kernel_id: u64, buffers: CudaK2KBuffers) {
        self.buffers.insert(kernel_id, buffers);
    }

    /// Get K2K buffers for a kernel.
    pub fn get(&self, kernel_id: u64) -> Option<&CudaK2KBuffers> {
        self.buffers.get(&kernel_id)
    }

    /// Connect two kernels for K2K messaging.
    ///
    /// Creates a unidirectional route from source to target.
    pub fn connect(&mut self, source_id: u64, target_id: u64) -> Result<()> {
        // Find next available slot for source
        let slot = self
            .connections
            .iter()
            .filter(|(src, _, _)| *src == source_id)
            .count();

        if slot >= MAX_K2K_ROUTES {
            return Err(RingKernelError::K2KError(format!(
                "Kernel {} has reached max K2K routes ({})",
                source_id, MAX_K2K_ROUTES
            )));
        }

        let source_buffers = self.buffers.get(&source_id).ok_or_else(|| {
            RingKernelError::K2KDestinationNotFound(format!("Source kernel {} not registered", source_id))
        })?;

        let target_buffers = self.buffers.get(&target_id).ok_or_else(|| {
            RingKernelError::K2KDestinationNotFound(format!("Target kernel {} not registered", target_id))
        })?;

        // Add route in source's routing table
        source_buffers.add_route(slot, target_buffers, target_id)?;

        self.connections.push((source_id, target_id, slot));

        Ok(())
    }

    /// Get all connections.
    pub fn connections(&self) -> &[(u64, u64, usize)] {
        &self.connections
    }
}

impl Default for K2KConnectionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inbox_header_size() {
        assert_eq!(mem::size_of::<K2KInboxHeader>(), 64);
    }

    #[test]
    fn test_route_entry_size() {
        assert_eq!(mem::size_of::<K2KRouteEntry>(), 48);
    }

    #[test]
    fn test_inbox_header_ops() {
        let mut header = K2KInboxHeader::new(64, 256);

        assert!(header.is_empty());
        assert!(!header.is_full());
        assert_eq!(header.size(), 0);
        assert_eq!(header.mask, 63);

        // Simulate adding messages
        header.head = 10;
        assert!(!header.is_empty());
        assert_eq!(header.size(), 10);

        // Simulate consuming messages
        header.tail = 5;
        assert_eq!(header.size(), 5);

        // Test full condition
        header.head = 64;
        header.tail = 0;
        assert!(header.is_full());
    }

    #[test]
    fn test_routing_table_default() {
        let table = StandardRoutingTable::default();
        assert_eq!(table.route_count, 0);
        for entry in table.routes.iter() {
            assert!(entry.is_empty());
        }
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_k2k_buffers_creation() {
        let device = CudaDevice::new(0).expect("Failed to create device");
        let buffers =
            CudaK2KBuffers::new(&device, 64, 256, MAX_K2K_ROUTES).expect("Failed to create buffers");

        assert_eq!(buffers.inbox_capacity(), 64);
        assert_eq!(buffers.msg_size(), 256);
        assert!(buffers.inbox_ptr() > 0);
        assert!(buffers.routing_table_ptr() > 0);
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_k2k_connection() {
        let device = CudaDevice::new(0).expect("Failed to create device");

        let mut manager = K2KConnectionManager::new();

        // Create buffers for two kernels
        let buffers_a =
            CudaK2KBuffers::new(&device, 64, 256, MAX_K2K_ROUTES).expect("Failed to create A");
        let buffers_b =
            CudaK2KBuffers::new(&device, 64, 256, MAX_K2K_ROUTES).expect("Failed to create B");

        manager.register(1, buffers_a);
        manager.register(2, buffers_b);

        // Connect A -> B
        manager.connect(1, 2).expect("Failed to connect");

        assert_eq!(manager.connections().len(), 1);
        assert_eq!(manager.connections()[0], (1, 2, 0));
    }
}
