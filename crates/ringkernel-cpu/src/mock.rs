//! GPU Mock Testing Utilities
//!
//! This module provides utilities for mocking GPU behavior in CPU tests.
//! It simulates GPU intrinsics, thread organization, and memory patterns.
//!
//! # Example
//!
//! ```rust
//! use ringkernel_cpu::mock::{MockGpu, MockThread, MockKernelConfig};
//!
//! // Configure a mock kernel launch
//! let config = MockKernelConfig::new()
//!     .with_grid_size(4, 4, 1)
//!     .with_block_size(32, 8, 1);
//!
//! // Create mock GPU context
//! let gpu = MockGpu::new(config);
//!
//! // Execute kernel with mock threads
//! gpu.dispatch(|thread| {
//!     let gid = thread.global_id();
//!     // Kernel code here
//! });
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Barrier, RwLock};

// ============================================================================
// MOCK KERNEL CONFIGURATION
// ============================================================================

/// Configuration for mock kernel execution.
#[derive(Debug, Clone)]
pub struct MockKernelConfig {
    /// Grid dimensions (number of blocks).
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions (threads per block).
    pub block_dim: (u32, u32, u32),
    /// Shared memory size in bytes.
    pub shared_memory_size: usize,
    /// Whether to simulate warp execution.
    pub simulate_warps: bool,
    /// Warp size (typically 32 for NVIDIA, 64 for AMD).
    pub warp_size: u32,
}

impl Default for MockKernelConfig {
    fn default() -> Self {
        Self {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_memory_size: 49152, // 48KB default
            simulate_warps: false,
            warp_size: 32,
        }
    }
}

impl MockKernelConfig {
    /// Create a new mock kernel configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set grid dimensions.
    pub fn with_grid_size(mut self, x: u32, y: u32, z: u32) -> Self {
        self.grid_dim = (x, y, z);
        self
    }

    /// Set block dimensions.
    pub fn with_block_size(mut self, x: u32, y: u32, z: u32) -> Self {
        self.block_dim = (x, y, z);
        self
    }

    /// Set shared memory size.
    pub fn with_shared_memory(mut self, bytes: usize) -> Self {
        self.shared_memory_size = bytes;
        self
    }

    /// Enable warp simulation.
    pub fn with_warp_simulation(mut self, warp_size: u32) -> Self {
        self.simulate_warps = true;
        self.warp_size = warp_size;
        self
    }

    /// Calculate total number of threads.
    pub fn total_threads(&self) -> u64 {
        let blocks = self.grid_dim.0 as u64 * self.grid_dim.1 as u64 * self.grid_dim.2 as u64;
        let threads_per_block =
            self.block_dim.0 as u64 * self.block_dim.1 as u64 * self.block_dim.2 as u64;
        blocks * threads_per_block
    }

    /// Calculate threads per block.
    pub fn threads_per_block(&self) -> u32 {
        self.block_dim.0 * self.block_dim.1 * self.block_dim.2
    }

    /// Calculate total blocks.
    pub fn total_blocks(&self) -> u32 {
        self.grid_dim.0 * self.grid_dim.1 * self.grid_dim.2
    }
}

// ============================================================================
// MOCK THREAD CONTEXT
// ============================================================================

/// Mock thread context providing GPU intrinsics.
#[derive(Debug, Clone)]
pub struct MockThread {
    /// Thread index within block (x, y, z).
    pub thread_idx: (u32, u32, u32),
    /// Block index within grid (x, y, z).
    pub block_idx: (u32, u32, u32),
    /// Block dimensions.
    pub block_dim: (u32, u32, u32),
    /// Grid dimensions.
    pub grid_dim: (u32, u32, u32),
    /// Warp ID (within block).
    pub warp_id: u32,
    /// Lane ID (within warp).
    pub lane_id: u32,
    /// Warp size.
    pub warp_size: u32,
}

impl MockThread {
    /// Create a new mock thread.
    pub fn new(
        thread_idx: (u32, u32, u32),
        block_idx: (u32, u32, u32),
        config: &MockKernelConfig,
    ) -> Self {
        let linear_tid = thread_idx.0
            + thread_idx.1 * config.block_dim.0
            + thread_idx.2 * config.block_dim.0 * config.block_dim.1;

        Self {
            thread_idx,
            block_idx,
            block_dim: config.block_dim,
            grid_dim: config.grid_dim,
            warp_id: linear_tid / config.warp_size,
            lane_id: linear_tid % config.warp_size,
            warp_size: config.warp_size,
        }
    }

    // ========================================================================
    // GPU Intrinsics
    // ========================================================================

    /// Get thread index X.
    #[inline]
    pub fn thread_idx_x(&self) -> u32 {
        self.thread_idx.0
    }

    /// Get thread index Y.
    #[inline]
    pub fn thread_idx_y(&self) -> u32 {
        self.thread_idx.1
    }

    /// Get thread index Z.
    #[inline]
    pub fn thread_idx_z(&self) -> u32 {
        self.thread_idx.2
    }

    /// Get block index X.
    #[inline]
    pub fn block_idx_x(&self) -> u32 {
        self.block_idx.0
    }

    /// Get block index Y.
    #[inline]
    pub fn block_idx_y(&self) -> u32 {
        self.block_idx.1
    }

    /// Get block index Z.
    #[inline]
    pub fn block_idx_z(&self) -> u32 {
        self.block_idx.2
    }

    /// Get block dimension X.
    #[inline]
    pub fn block_dim_x(&self) -> u32 {
        self.block_dim.0
    }

    /// Get block dimension Y.
    #[inline]
    pub fn block_dim_y(&self) -> u32 {
        self.block_dim.1
    }

    /// Get block dimension Z.
    #[inline]
    pub fn block_dim_z(&self) -> u32 {
        self.block_dim.2
    }

    /// Get grid dimension X.
    #[inline]
    pub fn grid_dim_x(&self) -> u32 {
        self.grid_dim.0
    }

    /// Get grid dimension Y.
    #[inline]
    pub fn grid_dim_y(&self) -> u32 {
        self.grid_dim.1
    }

    /// Get grid dimension Z.
    #[inline]
    pub fn grid_dim_z(&self) -> u32 {
        self.grid_dim.2
    }

    /// Get global thread ID (1D linearized).
    #[inline]
    pub fn global_id(&self) -> u64 {
        let block_linear = self.block_idx.0 as u64
            + self.block_idx.1 as u64 * self.grid_dim.0 as u64
            + self.block_idx.2 as u64 * self.grid_dim.0 as u64 * self.grid_dim.1 as u64;

        let threads_per_block =
            self.block_dim.0 as u64 * self.block_dim.1 as u64 * self.block_dim.2 as u64;
        let thread_linear = self.thread_idx.0 as u64
            + self.thread_idx.1 as u64 * self.block_dim.0 as u64
            + self.thread_idx.2 as u64 * self.block_dim.0 as u64 * self.block_dim.1 as u64;

        block_linear * threads_per_block + thread_linear
    }

    /// Get global X coordinate.
    #[inline]
    pub fn global_x(&self) -> u32 {
        self.block_idx.0 * self.block_dim.0 + self.thread_idx.0
    }

    /// Get global Y coordinate.
    #[inline]
    pub fn global_y(&self) -> u32 {
        self.block_idx.1 * self.block_dim.1 + self.thread_idx.1
    }

    /// Get global Z coordinate.
    #[inline]
    pub fn global_z(&self) -> u32 {
        self.block_idx.2 * self.block_dim.2 + self.thread_idx.2
    }

    /// Check if this is the first thread in the block.
    #[inline]
    pub fn is_block_leader(&self) -> bool {
        self.thread_idx == (0, 0, 0)
    }

    /// Check if this is the first thread in the warp.
    #[inline]
    pub fn is_warp_leader(&self) -> bool {
        self.lane_id == 0
    }
}

// ============================================================================
// MOCK SHARED MEMORY
// ============================================================================

/// Mock shared memory for a block.
pub struct MockSharedMemory {
    data: RefCell<Vec<u8>>,
    size: usize,
}

impl MockSharedMemory {
    /// Create new shared memory.
    pub fn new(size: usize) -> Self {
        Self {
            data: RefCell::new(vec![0u8; size]),
            size,
        }
    }

    /// Get size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Read a value at offset.
    pub fn read<T: Copy>(&self, offset: usize) -> T {
        let data = self.data.borrow();
        assert!(offset + std::mem::size_of::<T>() <= self.size);
        unsafe { std::ptr::read(data.as_ptr().add(offset) as *const T) }
    }

    /// Write a value at offset.
    pub fn write<T: Copy>(&self, offset: usize, value: T) {
        let mut data = self.data.borrow_mut();
        assert!(offset + std::mem::size_of::<T>() <= self.size);
        unsafe { std::ptr::write(data.as_mut_ptr().add(offset) as *mut T, value) };
    }

    /// Get a slice view.
    pub fn as_slice<T: Copy>(&self, offset: usize, count: usize) -> Vec<T> {
        let data = self.data.borrow();
        let byte_size = count * std::mem::size_of::<T>();
        assert!(offset + byte_size <= self.size);

        let mut result = Vec::with_capacity(count);
        unsafe {
            let ptr = data.as_ptr().add(offset) as *const T;
            for i in 0..count {
                result.push(*ptr.add(i));
            }
        }
        result
    }

    /// Write a slice.
    pub fn write_slice<T: Copy>(&self, offset: usize, values: &[T]) {
        let mut data = self.data.borrow_mut();
        let byte_size = std::mem::size_of_val(values);
        assert!(offset + byte_size <= self.size);

        unsafe {
            let ptr = data.as_mut_ptr().add(offset) as *mut T;
            for (i, v) in values.iter().enumerate() {
                *ptr.add(i) = *v;
            }
        }
    }
}

// ============================================================================
// MOCK ATOMICS
// ============================================================================

/// Mock atomic operations.
pub struct MockAtomics {
    u32_values: RwLock<HashMap<usize, AtomicU32>>,
    u64_values: RwLock<HashMap<usize, AtomicU64>>,
}

impl Default for MockAtomics {
    fn default() -> Self {
        Self::new()
    }
}

impl MockAtomics {
    /// Create new atomics storage.
    pub fn new() -> Self {
        Self {
            u32_values: RwLock::new(HashMap::new()),
            u64_values: RwLock::new(HashMap::new()),
        }
    }

    /// Atomic add (u32).
    pub fn atomic_add_u32(&self, addr: usize, val: u32) -> u32 {
        let mut map = self.u32_values.write().unwrap();
        let atomic = map.entry(addr).or_insert_with(|| AtomicU32::new(0));
        atomic.fetch_add(val, Ordering::SeqCst)
    }

    /// Atomic add (u64).
    pub fn atomic_add_u64(&self, addr: usize, val: u64) -> u64 {
        let mut map = self.u64_values.write().unwrap();
        let atomic = map.entry(addr).or_insert_with(|| AtomicU64::new(0));
        atomic.fetch_add(val, Ordering::SeqCst)
    }

    /// Atomic CAS (u32).
    pub fn atomic_cas_u32(&self, addr: usize, expected: u32, new: u32) -> u32 {
        let mut map = self.u32_values.write().unwrap();
        let atomic = map.entry(addr).or_insert_with(|| AtomicU32::new(0));
        match atomic.compare_exchange(expected, new, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(v) | Err(v) => v,
        }
    }

    /// Atomic max (u32).
    pub fn atomic_max_u32(&self, addr: usize, val: u32) -> u32 {
        let mut map = self.u32_values.write().unwrap();
        let atomic = map.entry(addr).or_insert_with(|| AtomicU32::new(0));
        atomic.fetch_max(val, Ordering::SeqCst)
    }

    /// Atomic min (u32).
    pub fn atomic_min_u32(&self, addr: usize, val: u32) -> u32 {
        let mut map = self.u32_values.write().unwrap();
        let atomic = map.entry(addr).or_insert_with(|| AtomicU32::new(0));
        atomic.fetch_min(val, Ordering::SeqCst)
    }

    /// Load value (u32).
    pub fn load_u32(&self, addr: usize) -> u32 {
        let map = self.u32_values.read().unwrap();
        map.get(&addr)
            .map(|a| a.load(Ordering::SeqCst))
            .unwrap_or(0)
    }

    /// Store value (u32).
    pub fn store_u32(&self, addr: usize, val: u32) {
        let mut map = self.u32_values.write().unwrap();
        let atomic = map.entry(addr).or_insert_with(|| AtomicU32::new(0));
        atomic.store(val, Ordering::SeqCst);
    }
}

// ============================================================================
// MOCK GPU
// ============================================================================

/// Mock GPU for testing kernel execution.
pub struct MockGpu {
    config: MockKernelConfig,
    atomics: Arc<MockAtomics>,
}

impl MockGpu {
    /// Create a new mock GPU.
    pub fn new(config: MockKernelConfig) -> Self {
        Self {
            config,
            atomics: Arc::new(MockAtomics::new()),
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &MockKernelConfig {
        &self.config
    }

    /// Get atomics.
    pub fn atomics(&self) -> &MockAtomics {
        &self.atomics
    }

    /// Dispatch kernel execution sequentially.
    ///
    /// Executes the kernel function for each thread in order.
    /// Useful for deterministic testing.
    pub fn dispatch<F>(&self, kernel: F)
    where
        F: Fn(&MockThread),
    {
        for bz in 0..self.config.grid_dim.2 {
            for by in 0..self.config.grid_dim.1 {
                for bx in 0..self.config.grid_dim.0 {
                    for tz in 0..self.config.block_dim.2 {
                        for ty in 0..self.config.block_dim.1 {
                            for tx in 0..self.config.block_dim.0 {
                                let thread =
                                    MockThread::new((tx, ty, tz), (bx, by, bz), &self.config);
                                kernel(&thread);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Dispatch with block synchronization.
    ///
    /// Provides a barrier for `sync_threads()` simulation within blocks.
    pub fn dispatch_with_sync<F>(&self, kernel: F)
    where
        F: Fn(&MockThread, &Barrier) + Send + Sync,
    {
        let threads_per_block = self.config.threads_per_block() as usize;

        for bz in 0..self.config.grid_dim.2 {
            for by in 0..self.config.grid_dim.1 {
                for bx in 0..self.config.grid_dim.0 {
                    // Each block runs in parallel threads
                    let barrier = Arc::new(Barrier::new(threads_per_block));
                    std::thread::scope(|s| {
                        for tz in 0..self.config.block_dim.2 {
                            for ty in 0..self.config.block_dim.1 {
                                for tx in 0..self.config.block_dim.0 {
                                    let barrier = Arc::clone(&barrier);
                                    let config = &self.config;
                                    let kernel_ref = &kernel;
                                    s.spawn(move || {
                                        let thread =
                                            MockThread::new((tx, ty, tz), (bx, by, bz), config);
                                        kernel_ref(&thread, &barrier);
                                    });
                                }
                            }
                        }
                    });
                }
            }
        }
    }
}

// ============================================================================
// MOCK WARP OPERATIONS
// ============================================================================

/// Mock warp operations for testing warp-level primitives.
pub struct MockWarp {
    /// Lane values (up to 64 lanes for AMD).
    lane_values: Vec<u32>,
    /// Warp size.
    warp_size: u32,
}

impl MockWarp {
    /// Create a new mock warp.
    pub fn new(warp_size: u32) -> Self {
        Self {
            lane_values: vec![0; warp_size as usize],
            warp_size,
        }
    }

    /// Set lane value.
    pub fn set_lane(&mut self, lane: u32, value: u32) {
        if (lane as usize) < self.lane_values.len() {
            self.lane_values[lane as usize] = value;
        }
    }

    /// Simulate warp shuffle.
    pub fn shuffle(&self, src_lane: u32) -> u32 {
        self.lane_values
            .get(src_lane as usize)
            .copied()
            .unwrap_or(0)
    }

    /// Simulate warp shuffle XOR.
    pub fn shuffle_xor(&self, lane_id: u32, mask: u32) -> u32 {
        let src = lane_id ^ mask;
        self.shuffle(src)
    }

    /// Simulate warp shuffle up.
    pub fn shuffle_up(&self, lane_id: u32, delta: u32) -> u32 {
        if lane_id >= delta {
            self.shuffle(lane_id - delta)
        } else {
            self.lane_values[lane_id as usize]
        }
    }

    /// Simulate warp shuffle down.
    pub fn shuffle_down(&self, lane_id: u32, delta: u32) -> u32 {
        if lane_id + delta < self.warp_size {
            self.shuffle(lane_id + delta)
        } else {
            self.lane_values[lane_id as usize]
        }
    }

    /// Simulate warp ballot.
    pub fn ballot(&self, predicate: impl Fn(u32) -> bool) -> u64 {
        let mut result = 0u64;
        for lane in 0..self.warp_size {
            if predicate(lane) {
                result |= 1 << lane;
            }
        }
        result
    }

    /// Simulate warp any.
    pub fn any(&self, predicate: impl Fn(u32) -> bool) -> bool {
        (0..self.warp_size).any(predicate)
    }

    /// Simulate warp all.
    pub fn all(&self, predicate: impl Fn(u32) -> bool) -> bool {
        (0..self.warp_size).all(predicate)
    }

    /// Simulate warp reduction (sum).
    pub fn reduce_sum(&self) -> u32 {
        self.lane_values.iter().sum()
    }

    /// Simulate warp prefix sum (exclusive).
    pub fn prefix_sum_exclusive(&self) -> Vec<u32> {
        let mut result = Vec::with_capacity(self.warp_size as usize);
        let mut sum = 0;
        for &v in &self.lane_values {
            result.push(sum);
            sum += v;
        }
        result
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_config() {
        let config = MockKernelConfig::new()
            .with_grid_size(4, 4, 1)
            .with_block_size(32, 8, 1);

        assert_eq!(config.total_blocks(), 16);
        assert_eq!(config.threads_per_block(), 256);
        assert_eq!(config.total_threads(), 4096);
    }

    #[test]
    fn test_mock_thread_intrinsics() {
        let config = MockKernelConfig::new()
            .with_grid_size(2, 2, 1)
            .with_block_size(16, 16, 1);

        let thread = MockThread::new((5, 3, 0), (1, 0, 0), &config);

        assert_eq!(thread.thread_idx_x(), 5);
        assert_eq!(thread.thread_idx_y(), 3);
        assert_eq!(thread.block_idx_x(), 1);
        assert_eq!(thread.block_dim_x(), 16);
        assert_eq!(thread.global_x(), 21); // 1*16 + 5
        assert_eq!(thread.global_y(), 3); // 0*16 + 3
    }

    #[test]
    fn test_mock_shared_memory() {
        let shmem = MockSharedMemory::new(1024);

        shmem.write::<f32>(0, 3.14);
        shmem.write::<f32>(4, 2.71);

        assert!((shmem.read::<f32>(0) - 3.14).abs() < 0.001);
        assert!((shmem.read::<f32>(4) - 2.71).abs() < 0.001);

        shmem.write_slice::<u32>(100, &[1, 2, 3, 4]);
        let slice = shmem.as_slice::<u32>(100, 4);
        assert_eq!(slice, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_mock_atomics() {
        let atomics = MockAtomics::new();

        let old = atomics.atomic_add_u32(0, 5);
        assert_eq!(old, 0);

        let old = atomics.atomic_add_u32(0, 3);
        assert_eq!(old, 5);

        assert_eq!(atomics.load_u32(0), 8);
    }

    #[test]
    fn test_mock_gpu_dispatch() {
        let config = MockKernelConfig::new()
            .with_grid_size(2, 1, 1)
            .with_block_size(4, 1, 1);

        let gpu = MockGpu::new(config);
        let counter = Arc::new(AtomicU32::new(0));

        let c = Arc::clone(&counter);
        gpu.dispatch(move |_thread| {
            c.fetch_add(1, Ordering::SeqCst);
        });

        assert_eq!(counter.load(Ordering::SeqCst), 8); // 2 blocks * 4 threads
    }

    #[test]
    fn test_mock_warp_shuffle() {
        let mut warp = MockWarp::new(32);

        // Set lane values
        for i in 0..32 {
            warp.set_lane(i, i * 2);
        }

        // Test shuffle
        assert_eq!(warp.shuffle(5), 10);
        assert_eq!(warp.shuffle(15), 30);

        // Test shuffle XOR
        assert_eq!(warp.shuffle_xor(0, 1), 2); // lane 0 XOR 1 = lane 1 value
        assert_eq!(warp.shuffle_xor(2, 1), 6); // lane 2 XOR 1 = lane 3 value
    }

    #[test]
    fn test_mock_warp_ballot() {
        let warp = MockWarp::new(32);

        // Ballot: all even lanes
        let ballot = warp.ballot(|lane| lane % 2 == 0);
        assert_eq!(ballot, 0x55555555); // Even bits set
    }

    #[test]
    fn test_mock_warp_reduce() {
        let mut warp = MockWarp::new(4);

        warp.set_lane(0, 1);
        warp.set_lane(1, 2);
        warp.set_lane(2, 3);
        warp.set_lane(3, 4);

        assert_eq!(warp.reduce_sum(), 10);

        let prefix = warp.prefix_sum_exclusive();
        assert_eq!(prefix, vec![0, 1, 3, 6]);
    }

    #[test]
    fn test_thread_global_id() {
        let config = MockKernelConfig::new()
            .with_grid_size(2, 2, 1)
            .with_block_size(4, 4, 1);

        // Thread (0,0) in block (0,0) -> global ID 0
        let t1 = MockThread::new((0, 0, 0), (0, 0, 0), &config);
        assert_eq!(t1.global_id(), 0);

        // Thread (0,0) in block (1,0) -> global ID 16 (one block worth)
        let t2 = MockThread::new((0, 0, 0), (1, 0, 0), &config);
        assert_eq!(t2.global_id(), 16);

        // Thread (3,3) in block (0,0) -> linear ID 15
        let t3 = MockThread::new((3, 3, 0), (0, 0, 0), &config);
        assert_eq!(t3.global_id(), 15);
    }
}
