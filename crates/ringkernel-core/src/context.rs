//! Ring context providing GPU intrinsics facade for kernel handlers.
//!
//! The RingContext provides a unified interface for GPU operations that
//! abstracts over different backends (CUDA, Metal, WebGPU, CPU).

use crate::hlc::{HlcClock, HlcTimestamp};
use crate::message::MessageEnvelope;
use crate::types::{BlockId, Dim3, FenceScope, GlobalThreadId, MemoryOrder, ThreadId, WarpId};

/// GPU intrinsics facade for kernel handlers.
///
/// This struct provides access to GPU-specific operations like thread
/// identification, synchronization, and atomic operations. The actual
/// implementation varies by backend.
///
/// # Lifetime
///
/// The context is borrowed for the duration of the kernel handler execution.
pub struct RingContext<'a> {
    /// Thread identity within block.
    pub thread_id: ThreadId,
    /// Block identity within grid.
    pub block_id: BlockId,
    /// Block dimensions.
    pub block_dim: Dim3,
    /// Grid dimensions.
    pub grid_dim: Dim3,
    /// HLC clock instance.
    clock: &'a HlcClock,
    /// Kernel ID.
    kernel_id: u64,
    /// Backend implementation.
    backend: ContextBackend,
}

/// Backend-specific context implementation.
#[derive(Debug, Clone)]
pub enum ContextBackend {
    /// CPU backend (for testing).
    Cpu,
    /// CUDA backend.
    Cuda,
    /// Metal backend.
    Metal,
    /// WebGPU backend.
    Wgpu,
}

impl<'a> RingContext<'a> {
    /// Create a new context.
    pub fn new(
        thread_id: ThreadId,
        block_id: BlockId,
        block_dim: Dim3,
        grid_dim: Dim3,
        clock: &'a HlcClock,
        kernel_id: u64,
        backend: ContextBackend,
    ) -> Self {
        Self {
            thread_id,
            block_id,
            block_dim,
            grid_dim,
            clock,
            kernel_id,
            backend,
        }
    }

    // === Thread Identity ===

    /// Get thread ID within block.
    #[inline]
    pub fn thread_id(&self) -> ThreadId {
        self.thread_id
    }

    /// Get block ID within grid.
    #[inline]
    pub fn block_id(&self) -> BlockId {
        self.block_id
    }

    /// Get global thread ID across all blocks.
    #[inline]
    pub fn global_thread_id(&self) -> GlobalThreadId {
        GlobalThreadId::from_block_thread(self.block_id, self.thread_id, self.block_dim)
    }

    /// Get warp ID within block.
    #[inline]
    pub fn warp_id(&self) -> WarpId {
        let linear = self
            .thread_id
            .linear_for_dim(self.block_dim.x, self.block_dim.y);
        WarpId::from_thread_linear(linear)
    }

    /// Get lane ID within warp (0-31).
    #[inline]
    pub fn lane_id(&self) -> u32 {
        let linear = self
            .thread_id
            .linear_for_dim(self.block_dim.x, self.block_dim.y);
        WarpId::lane_id(linear)
    }

    /// Get block dimensions.
    #[inline]
    pub fn block_dim(&self) -> Dim3 {
        self.block_dim
    }

    /// Get grid dimensions.
    #[inline]
    pub fn grid_dim(&self) -> Dim3 {
        self.grid_dim
    }

    /// Get kernel ID.
    #[inline]
    pub fn kernel_id(&self) -> u64 {
        self.kernel_id
    }

    // === Synchronization ===

    /// Synchronize all threads in the block.
    ///
    /// All threads in the block must reach this barrier before any
    /// thread can proceed past it.
    #[inline]
    pub fn sync_threads(&self) {
        match self.backend {
            ContextBackend::Cpu => {
                // CPU: no-op (single-threaded simulation)
            }
            _ => {
                // GPU backends would call __syncthreads() or equivalent
                // Placeholder for actual implementation
            }
        }
    }

    /// Synchronize all threads in the grid (cooperative groups).
    ///
    /// Requires cooperative kernel launch support.
    #[inline]
    pub fn sync_grid(&self) {
        match self.backend {
            ContextBackend::Cpu => {
                // CPU: no-op
            }
            _ => {
                // GPU backends would call cooperative grid sync
            }
        }
    }

    /// Synchronize threads within a warp.
    #[inline]
    pub fn sync_warp(&self) {
        match self.backend {
            ContextBackend::Cpu => {
                // CPU: no-op
            }
            _ => {
                // GPU backends would call __syncwarp()
            }
        }
    }

    // === Memory Fencing ===

    /// Memory fence at the specified scope.
    #[inline]
    pub fn thread_fence(&self, scope: FenceScope) {
        match (self.backend.clone(), scope) {
            (ContextBackend::Cpu, _) => {
                std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            }
            _ => {
                // GPU backends would call appropriate fence intrinsic
            }
        }
    }

    /// Thread-scope fence (compiler barrier).
    #[inline]
    pub fn fence_thread(&self) {
        self.thread_fence(FenceScope::Thread);
    }

    /// Block-scope fence.
    #[inline]
    pub fn fence_block(&self) {
        self.thread_fence(FenceScope::Block);
    }

    /// Device-scope fence.
    #[inline]
    pub fn fence_device(&self) {
        self.thread_fence(FenceScope::Device);
    }

    /// System-scope fence (CPU+GPU visible).
    #[inline]
    pub fn fence_system(&self) {
        self.thread_fence(FenceScope::System);
    }

    // === HLC Operations ===

    /// Get current HLC timestamp.
    #[inline]
    pub fn now(&self) -> HlcTimestamp {
        self.clock.now()
    }

    /// Generate a new HLC timestamp (advances clock).
    #[inline]
    pub fn tick(&self) -> HlcTimestamp {
        self.clock.tick()
    }

    /// Update clock with received timestamp.
    #[inline]
    pub fn update_clock(&self, received: &HlcTimestamp) -> crate::error::Result<HlcTimestamp> {
        self.clock.update(received)
    }

    // === Atomic Operations ===

    /// Atomic add and return old value.
    #[inline]
    pub fn atomic_add(
        &self,
        ptr: &std::sync::atomic::AtomicU64,
        val: u64,
        order: MemoryOrder,
    ) -> u64 {
        let ordering = match order {
            MemoryOrder::Relaxed => std::sync::atomic::Ordering::Relaxed,
            MemoryOrder::Acquire => std::sync::atomic::Ordering::Acquire,
            MemoryOrder::Release => std::sync::atomic::Ordering::Release,
            MemoryOrder::AcquireRelease => std::sync::atomic::Ordering::AcqRel,
            MemoryOrder::SeqCst => std::sync::atomic::Ordering::SeqCst,
        };
        ptr.fetch_add(val, ordering)
    }

    /// Atomic compare-and-swap.
    #[inline]
    pub fn atomic_cas(
        &self,
        ptr: &std::sync::atomic::AtomicU64,
        expected: u64,
        desired: u64,
        success: MemoryOrder,
        failure: MemoryOrder,
    ) -> Result<u64, u64> {
        let success_ord = match success {
            MemoryOrder::Relaxed => std::sync::atomic::Ordering::Relaxed,
            MemoryOrder::Acquire => std::sync::atomic::Ordering::Acquire,
            MemoryOrder::Release => std::sync::atomic::Ordering::Release,
            MemoryOrder::AcquireRelease => std::sync::atomic::Ordering::AcqRel,
            MemoryOrder::SeqCst => std::sync::atomic::Ordering::SeqCst,
        };
        let failure_ord = match failure {
            MemoryOrder::Relaxed => std::sync::atomic::Ordering::Relaxed,
            MemoryOrder::Acquire => std::sync::atomic::Ordering::Acquire,
            MemoryOrder::Release => std::sync::atomic::Ordering::Release,
            MemoryOrder::AcquireRelease => std::sync::atomic::Ordering::AcqRel,
            MemoryOrder::SeqCst => std::sync::atomic::Ordering::SeqCst,
        };
        ptr.compare_exchange(expected, desired, success_ord, failure_ord)
    }

    /// Atomic exchange.
    #[inline]
    pub fn atomic_exchange(
        &self,
        ptr: &std::sync::atomic::AtomicU64,
        val: u64,
        order: MemoryOrder,
    ) -> u64 {
        let ordering = match order {
            MemoryOrder::Relaxed => std::sync::atomic::Ordering::Relaxed,
            MemoryOrder::Acquire => std::sync::atomic::Ordering::Acquire,
            MemoryOrder::Release => std::sync::atomic::Ordering::Release,
            MemoryOrder::AcquireRelease => std::sync::atomic::Ordering::AcqRel,
            MemoryOrder::SeqCst => std::sync::atomic::Ordering::SeqCst,
        };
        ptr.swap(val, ordering)
    }

    // === Warp Primitives ===

    /// Warp shuffle - get value from another lane.
    ///
    /// Returns the value from the specified source lane.
    #[inline]
    pub fn warp_shuffle<T: Copy>(&self, value: T, src_lane: u32) -> T {
        match self.backend {
            ContextBackend::Cpu => {
                // CPU: just return own value (no other lanes)
                let _ = src_lane;
                value
            }
            _ => {
                // GPU would use __shfl_sync()
                let _ = src_lane;
                value
            }
        }
    }

    /// Warp shuffle down - get value from lane + delta.
    #[inline]
    pub fn warp_shuffle_down<T: Copy>(&self, value: T, delta: u32) -> T {
        self.warp_shuffle(value, self.lane_id().saturating_add(delta))
    }

    /// Warp shuffle up - get value from lane - delta.
    #[inline]
    pub fn warp_shuffle_up<T: Copy>(&self, value: T, delta: u32) -> T {
        self.warp_shuffle(value, self.lane_id().saturating_sub(delta))
    }

    /// Warp shuffle XOR - get value from lane XOR mask.
    #[inline]
    pub fn warp_shuffle_xor<T: Copy>(&self, value: T, mask: u32) -> T {
        self.warp_shuffle(value, self.lane_id() ^ mask)
    }

    /// Warp ballot - get bitmask of lanes where predicate is true.
    #[inline]
    pub fn warp_ballot(&self, predicate: bool) -> u32 {
        match self.backend {
            ContextBackend::Cpu => {
                // CPU: single thread, return 1 or 0
                if predicate {
                    1
                } else {
                    0
                }
            }
            _ => {
                // GPU would use __ballot_sync()
                if predicate {
                    1 << self.lane_id()
                } else {
                    0
                }
            }
        }
    }

    /// Warp all - check if predicate is true for all lanes.
    #[inline]
    pub fn warp_all(&self, predicate: bool) -> bool {
        match self.backend {
            ContextBackend::Cpu => predicate,
            _ => {
                // GPU would use __all_sync()
                predicate
            }
        }
    }

    /// Warp any - check if predicate is true for any lane.
    #[inline]
    pub fn warp_any(&self, predicate: bool) -> bool {
        match self.backend {
            ContextBackend::Cpu => predicate,
            _ => {
                // GPU would use __any_sync()
                predicate
            }
        }
    }

    // === K2K Messaging ===

    /// Send message to another kernel (K2K).
    ///
    /// This is a placeholder; actual implementation requires runtime support.
    #[inline]
    pub fn k2k_send(
        &self,
        _target_kernel: u64,
        _envelope: &MessageEnvelope,
    ) -> crate::error::Result<()> {
        // K2K messaging requires runtime bridge support
        Err(crate::error::RingKernelError::NotSupported(
            "K2K messaging requires runtime".to_string(),
        ))
    }

    /// Try to receive message from K2K queue.
    #[inline]
    pub fn k2k_try_recv(&self) -> crate::error::Result<MessageEnvelope> {
        // K2K messaging requires runtime bridge support
        Err(crate::error::RingKernelError::NotSupported(
            "K2K messaging requires runtime".to_string(),
        ))
    }
}

impl<'a> std::fmt::Debug for RingContext<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RingContext")
            .field("thread_id", &self.thread_id)
            .field("block_id", &self.block_id)
            .field("block_dim", &self.block_dim)
            .field("grid_dim", &self.grid_dim)
            .field("kernel_id", &self.kernel_id)
            .field("backend", &self.backend)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_context(clock: &HlcClock) -> RingContext<'_> {
        RingContext::new(
            ThreadId::new_1d(0),
            BlockId::new_1d(0),
            Dim3::new_1d(256),
            Dim3::new_1d(1),
            clock,
            1,
            ContextBackend::Cpu,
        )
    }

    #[test]
    fn test_thread_identity() {
        let clock = HlcClock::new(1);
        let ctx = make_test_context(&clock);

        assert_eq!(ctx.thread_id().x, 0);
        assert_eq!(ctx.block_id().x, 0);
        assert_eq!(ctx.global_thread_id().x, 0);
    }

    #[test]
    fn test_warp_id() {
        let clock = HlcClock::new(1);
        let ctx = RingContext::new(
            ThreadId::new_1d(35), // Thread 35 is in warp 1, lane 3
            BlockId::new_1d(0),
            Dim3::new_1d(256),
            Dim3::new_1d(1),
            &clock,
            1,
            ContextBackend::Cpu,
        );

        assert_eq!(ctx.warp_id().0, 1);
        assert_eq!(ctx.lane_id(), 3);
    }

    #[test]
    fn test_hlc_operations() {
        let clock = HlcClock::new(1);
        let ctx = make_test_context(&clock);

        let ts1 = ctx.now();
        let ts2 = ctx.tick();
        assert!(ts2 >= ts1);
    }

    #[test]
    fn test_warp_ballot_cpu() {
        let clock = HlcClock::new(1);
        let ctx = make_test_context(&clock);

        assert_eq!(ctx.warp_ballot(true), 1);
        assert_eq!(ctx.warp_ballot(false), 0);
    }
}
