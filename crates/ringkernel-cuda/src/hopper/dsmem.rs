//! Distributed Shared Memory (DSMEM) for Hopper Thread Block Clusters.
//!
//! DSMEM allows thread blocks within a cluster to access each other's shared memory,
//! enabling intra-cluster K2K messaging without going through global memory.
//!
//! # Architecture
//!
//! ```text
//! ┌─── Cluster (4 blocks) ────────────────────────────────────────┐
//! │ ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐               │
//! │ │Block 0 │  │Block 1 │  │Block 2 │  │Block 3 │               │
//! │ │ SMEM   │◄─►│ SMEM   │◄─►│ SMEM   │◄─►│ SMEM   │               │
//! │ │ 228KB  │  │ 228KB  │  │ 228KB  │  │ 228KB  │               │
//! │ └────────┘  └────────┘  └────────┘  └────────┘               │
//! │         ↕ DSMEM (no global memory) ↕                          │
//! │ Total cluster DSMEM pool: 4 × 228KB = 912KB                  │
//! └───────────────────────────────────────────────────────────────┘
//! ```
//!
//! On Hopper, each SM has 228KB of shared memory. A cluster of 8 blocks
//! provides up to 1.78MB of DSMEM — 7x the bandwidth of global memory
//! for intra-cluster messaging.
//!
//! # Device-Side Usage (in CUDA C++)
//!
//! ```c
//! // In cluster kernel (PTX):
//! // Use cluster.map_shared_rank(ptr, target_rank) to access remote shared memory
//! // Use cluster.sync() for cluster-level synchronization
//! ```

use ringkernel_core::error::{Result, RingKernelError};

use crate::device::CudaDevice;

/// Configuration for DSMEM-backed K2K messaging.
#[derive(Debug, Clone)]
pub struct DsmemConfig {
    /// Number of message slots in the DSMEM queue per block pair.
    pub queue_slots: u32,
    /// Size of each message slot in bytes.
    pub slot_size: u32,
    /// Shared memory reserved for DSMEM queues per block (bytes).
    /// Remaining shared memory is available for computation.
    pub dsmem_reservation: u32,
}

impl Default for DsmemConfig {
    fn default() -> Self {
        Self {
            queue_slots: 16,
            slot_size: 256,
            dsmem_reservation: 16384, // 16KB for DSMEM queues per block
        }
    }
}

impl DsmemConfig {
    /// Total DSMEM queue size per block in bytes.
    pub fn queue_size_bytes(&self) -> u32 {
        self.queue_slots * self.slot_size
    }

    /// Compute per-block shared memory needed for DSMEM K2K queues.
    ///
    /// Each block needs inbox + outbox for each neighbor in the cluster.
    /// For a cluster of N blocks, each block has (N-1) neighbors.
    pub fn shared_mem_per_block(&self, cluster_size: u32) -> u32 {
        if cluster_size <= 1 {
            return 0;
        }

        let neighbors = cluster_size - 1;
        // Each neighbor: one inbox queue + header
        let header_size = 64; // queue head/tail/capacity/padding
        let per_neighbor = header_size + self.queue_size_bytes();
        neighbors * per_neighbor
    }

    /// Validate configuration against available shared memory.
    pub fn validate(&self, device: &CudaDevice, cluster_size: u32) -> Result<()> {
        let needed = self.shared_mem_per_block(cluster_size);
        let max_smem = device
            .inner()
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
            .unwrap_or(228 * 1024) as u32; // Hopper default 228KB

        if needed > max_smem {
            return Err(RingKernelError::InvalidConfig(format!(
                "DSMEM K2K needs {} bytes/block but device has {} bytes shared memory. \
                 Reduce cluster_size ({}) or slot_size ({})",
                needed, max_smem, cluster_size, self.slot_size
            )));
        }

        if !self.queue_slots.is_power_of_two() {
            return Err(RingKernelError::InvalidConfig(format!(
                "DSMEM queue_slots must be power of 2, got {}",
                self.queue_slots
            )));
        }

        Ok(())
    }
}

/// DSMEM K2K queue header laid out in shared memory.
///
/// This structure is placed at the beginning of each DSMEM queue region.
/// The actual message slots follow immediately after the header.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct DsmemQueueHeader {
    /// Write index (producer advances).
    pub head: u32,
    /// Read index (consumer advances).
    pub tail: u32,
    /// Queue capacity (power of 2).
    pub capacity: u32,
    /// Capacity mask for modular indexing (capacity - 1).
    pub mask: u32,
    /// Source block rank within cluster.
    pub src_rank: u32,
    /// Destination block rank within cluster.
    pub dst_rank: u32,
    /// Slot size in bytes.
    pub slot_size: u32,
    /// Padding to 64-byte alignment.
    pub _pad: [u32; 9],
}

/// Compute the DSMEM layout for a cluster.
///
/// Returns the shared memory offset where each block's DSMEM queues begin,
/// and the total shared memory required per block.
pub fn compute_dsmem_layout(config: &DsmemConfig, cluster_size: u32) -> DsmemLayout {
    if cluster_size <= 1 {
        return DsmemLayout {
            queues_per_block: 0,
            shared_mem_per_block: 0,
            queue_offsets: Vec::new(),
        };
    }

    let neighbors = cluster_size - 1;
    let header_size = std::mem::size_of::<DsmemQueueHeader>() as u32;
    let queue_data_size = config.queue_slots * config.slot_size;
    let per_queue = header_size + queue_data_size;

    let mut queue_offsets = Vec::with_capacity(neighbors as usize);
    for i in 0..neighbors {
        queue_offsets.push(i * per_queue);
    }

    DsmemLayout {
        queues_per_block: neighbors,
        shared_mem_per_block: neighbors * per_queue,
        queue_offsets,
    }
}

/// Layout of DSMEM queues within each block's shared memory.
#[derive(Debug, Clone)]
pub struct DsmemLayout {
    /// Number of queues per block (one per neighbor in cluster).
    pub queues_per_block: u32,
    /// Total shared memory needed per block for DSMEM queues.
    pub shared_mem_per_block: u32,
    /// Byte offset of each queue within shared memory.
    pub queue_offsets: Vec<u32>,
}

/// Check if DSMEM is available for the given configuration.
pub fn is_dsmem_available(device: &CudaDevice, cluster_size: u32) -> bool {
    if !super::supports_cluster_launch(device) {
        return false;
    }
    cluster_size > 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dsmem_config_defaults() {
        let config = DsmemConfig::default();
        assert_eq!(config.queue_slots, 16);
        assert_eq!(config.slot_size, 256);
        assert_eq!(config.queue_size_bytes(), 4096);
    }

    #[test]
    fn test_shared_mem_per_block() {
        let config = DsmemConfig::default();

        // No cluster = no DSMEM
        assert_eq!(config.shared_mem_per_block(1), 0);

        // 2-block cluster: 1 neighbor
        let smem_2 = config.shared_mem_per_block(2);
        assert!(smem_2 > 0);

        // 4-block cluster: 3 neighbors, should be ~3x
        let smem_4 = config.shared_mem_per_block(4);
        assert_eq!(smem_4, smem_2 * 3);
    }

    #[test]
    fn test_dsmem_layout() {
        let config = DsmemConfig::default();
        let layout = compute_dsmem_layout(&config, 4);

        assert_eq!(layout.queues_per_block, 3); // 4 - 1 neighbors
        assert_eq!(layout.queue_offsets.len(), 3);
        assert!(layout.shared_mem_per_block > 0);
    }

    #[test]
    fn test_dsmem_layout_no_cluster() {
        let config = DsmemConfig::default();
        let layout = compute_dsmem_layout(&config, 1);

        assert_eq!(layout.queues_per_block, 0);
        assert_eq!(layout.shared_mem_per_block, 0);
    }

    #[test]
    fn test_queue_slots_power_of_two() {
        let mut config = DsmemConfig::default();
        config.queue_slots = 7; // Not power of 2

        // We can't test validate() without a device, but we can verify the check logic
        assert!(!config.queue_slots.is_power_of_two());
    }
}
