---
layout: default
title: Memory Management
nav_order: 5
---

# Memory Management

## GPU Memory Challenges

Ring Kernels require careful memory management across CPU↔GPU boundaries:

1. **GPU Buffer Allocation/Deallocation**
2. **Pinned Host Memory** for DMA transfers
3. **Unified Memory** (where available)
4. **Memory Pools** for frequent allocations
5. **Cross-Device P2P** for multi-GPU

---

## Rust Ownership Model for GPU Memory

### GpuBuffer Wrapper

```rust
// crates/ringkernel-core/src/buffer.rs

use std::marker::PhantomData;
use std::ptr::NonNull;

/// Owned GPU memory buffer with RAII cleanup.
pub struct GpuBuffer<T: Copy + Send> {
    device_ptr: NonNull<T>,
    len: usize,
    capacity: usize,
    allocator: Box<dyn GpuAllocator>,
    _marker: PhantomData<T>,
}

// Safety: GPU memory is thread-safe when properly synchronized
unsafe impl<T: Copy + Send> Send for GpuBuffer<T> {}
unsafe impl<T: Copy + Send> Sync for GpuBuffer<T> {}

impl<T: Copy + Send> GpuBuffer<T> {
    /// Allocate new GPU buffer.
    pub fn new(len: usize, allocator: Box<dyn GpuAllocator>) -> Result<Self> {
        let size = len * std::mem::size_of::<T>();
        let device_ptr = allocator.allocate(size)?;

        Ok(Self {
            device_ptr: NonNull::new(device_ptr as *mut T)
                .ok_or(Error::AllocationFailed)?,
            len,
            capacity: len,
            allocator,
            _marker: PhantomData,
        })
    }

    /// Get raw device pointer.
    pub fn as_ptr(&self) -> *const T {
        self.device_ptr.as_ptr()
    }

    /// Get mutable device pointer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.device_ptr.as_ptr()
    }

    /// Length in elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

impl<T: Copy + Send> Drop for GpuBuffer<T> {
    fn drop(&mut self) {
        // Safety: we own this memory
        unsafe {
            let _ = self.allocator.free(self.device_ptr.as_ptr() as *mut u8);
        }
    }
}
```

---

## Pinned Host Memory

Required for efficient DMA transfers:

```rust
/// Pinned (page-locked) host memory for DMA transfers.
pub struct PinnedBuffer<T: Copy + Send> {
    host_ptr: NonNull<T>,
    len: usize,
    allocator: Box<dyn HostAllocator>,
    _marker: PhantomData<T>,
}

impl<T: Copy + Send> PinnedBuffer<T> {
    /// Allocate pinned memory.
    pub fn new(len: usize, allocator: Box<dyn HostAllocator>) -> Result<Self> {
        let size = len * std::mem::size_of::<T>();
        let host_ptr = allocator.allocate_pinned(size)?;

        Ok(Self {
            host_ptr: NonNull::new(host_ptr as *mut T)
                .ok_or(Error::AllocationFailed)?,
            len,
            allocator,
            _marker: PhantomData,
        })
    }

    /// Get slice for reading.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.host_ptr.as_ptr(), self.len) }
    }

    /// Get mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.host_ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy + Send> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            let _ = self.allocator.free_pinned(self.host_ptr.as_ptr() as *mut u8);
        }
    }
}
```

---

## Memory Pool for Ring Buffers

Reduces allocation overhead for message queues:

```rust
/// Thread-safe memory pool for GPU allocations.
pub struct GpuMemoryPool {
    allocator: Arc<dyn GpuAllocator>,
    pools: Mutex<HashMap<usize, Vec<NonNull<u8>>>>,
    stats: AtomicPoolStats,
}

impl GpuMemoryPool {
    /// Get a buffer from the pool or allocate new.
    pub fn acquire(&self, size: usize) -> Result<PooledBuffer> {
        let bucket_size = size.next_power_of_two();

        let mut pools = self.pools.lock().unwrap();
        if let Some(pool) = pools.get_mut(&bucket_size) {
            if let Some(ptr) = pool.pop() {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(PooledBuffer::from_pool(ptr, bucket_size, self));
            }
        }

        // Allocate new
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        let ptr = self.allocator.allocate(bucket_size)?;
        Ok(PooledBuffer::new(
            NonNull::new(ptr).ok_or(Error::AllocationFailed)?,
            bucket_size,
            self,
        ))
    }

    /// Return buffer to pool.
    pub fn release(&self, ptr: NonNull<u8>, size: usize) {
        let mut pools = self.pools.lock().unwrap();
        pools.entry(size).or_default().push(ptr);
        self.stats.returns.fetch_add(1, Ordering::Relaxed);
    }
}

/// RAII handle that returns to pool on drop.
pub struct PooledBuffer<'a> {
    ptr: NonNull<u8>,
    size: usize,
    pool: &'a GpuMemoryPool,
}

impl Drop for PooledBuffer<'_> {
    fn drop(&mut self) {
        self.pool.release(self.ptr, self.size);
    }
}
```

---

## Size-Stratified Memory Pool (v0.3.0)

For analytics workloads with varying buffer sizes, the stratified pool provides efficient multi-size buffer reuse:

```rust
use ringkernel_core::memory::{StratifiedMemoryPool, SizeBucket, StratifiedPoolStats};

// Create a stratified pool with automatic bucket selection
let pool = StratifiedMemoryPool::new("analytics");

// Allocate buffers - automatically routed to appropriate bucket
let tiny_buf = pool.allocate(100);     // → Tiny bucket (256B)
let small_buf = pool.allocate(800);    // → Small bucket (1KB)
let medium_buf = pool.allocate(2000);  // → Medium bucket (4KB)
let large_buf = pool.allocate(10000);  // → Large bucket (16KB)
let huge_buf = pool.allocate(50000);   // → Huge bucket (64KB)

// Buffers automatically return to correct bucket on drop
drop(medium_buf);

// Check pool statistics
let stats = pool.stats();
println!("Total allocations: {}", stats.total_allocations);
println!("Cache hit rate: {:.1}%", stats.hit_rate() * 100.0);
println!("Hits per bucket: {:?}", stats.hits_per_bucket);
```

### Size Buckets

| Bucket | Size | Use Case |
|--------|------|----------|
| `Tiny` | 256 B | Metadata, small messages, control structures |
| `Small` | 1 KB | Typical message payloads, small vectors |
| `Medium` | 4 KB | Page-sized allocations, batch metadata |
| `Large` | 16 KB | Batch operations, intermediate results |
| `Huge` | 64 KB | Large transfers, big data chunks |

```rust
use ringkernel_core::memory::SizeBucket;

// Find appropriate bucket for a size
let bucket = SizeBucket::for_size(2500);  // Returns Medium (4KB)

// Bucket operations
let upgraded = bucket.upgrade();    // Large (16KB)
let downgraded = bucket.downgrade(); // Small (1KB)

// Get bucket size
println!("Medium bucket size: {} bytes", SizeBucket::Medium.size()); // 4096
```

### StratifiedBuffer RAII Wrapper

```rust
use ringkernel_core::memory::{StratifiedMemoryPool, StratifiedBuffer};

let pool = StratifiedMemoryPool::new("compute");

// StratifiedBuffer automatically returns to the correct bucket on drop
{
    let buffer: StratifiedBuffer = pool.allocate(3000);
    // Use buffer...
    println!("Allocated {} bytes in {:?} bucket", buffer.size(), buffer.bucket());
} // Buffer returned to Medium bucket here

// Next allocation of similar size reuses the buffer
let reused = pool.allocate(3500);  // Cache hit!
```

---

## Analytics Context (v0.3.0)

For operations that allocate multiple related buffers (BFS traversal, DFG mining, pattern detection), `AnalyticsContext` provides grouped lifecycle management:

```rust
use ringkernel_core::analytics_context::{AnalyticsContext, AnalyticsContextBuilder};

// Create context for a graph algorithm
let mut ctx = AnalyticsContext::new("bfs_traversal");

// Allocate buffers - all tracked together
let frontier = ctx.allocate(1024);
let visited = ctx.allocate(num_nodes / 8);  // Bit vector
let distances = ctx.allocate_typed::<u32>(num_nodes);

// Check memory usage
let stats = ctx.stats();
println!("Current: {} bytes", stats.current_bytes);
println!("Peak: {} bytes", stats.peak_bytes);
println!("Allocations: {}", stats.allocation_count);

// All buffers released when context drops
drop(ctx);
```

### Typed Allocations

```rust
use ringkernel_core::analytics_context::AnalyticsContext;

let mut ctx = AnalyticsContext::new("matrix_ops");

// Type-safe allocation with automatic sizing
let matrix = ctx.allocate_typed::<f32>(1024 * 1024);  // 4MB for 1M floats
let indices = ctx.allocate_typed::<u64>(65536);       // 512KB for 64K indices

// Track typed allocation statistics
println!("Typed allocations: {}", ctx.stats().typed_allocations);
```

### Builder Pattern with Preallocation

```rust
use ringkernel_core::analytics_context::AnalyticsContextBuilder;

// Pre-allocate for known workload
let ctx = AnalyticsContextBuilder::new("pagerank")
    .with_expected_allocations(5)
    .with_preallocation(num_nodes * 4)      // Ranks buffer
    .with_preallocation(num_nodes * 4)      // Previous ranks
    .with_preallocation(num_edges * 8)      // Edge weights
    .build();
```

---

## Memory Pressure Handling (v0.3.0)

Monitor and react to memory pressure with configurable strategies:

```rust
use ringkernel_core::memory::{
    PressureHandler, PressureReaction, PressureLevel, StratifiedMemoryPool
};

// Create pool with pressure monitoring
let pool = StratifiedMemoryPool::new("monitored");

// Configure pressure handler
let handler = PressureHandler::new()
    .on_elevated(PressureReaction::None)           // Ignore mild pressure
    .on_warning(PressureReaction::Shrink {
        target_utilization: 0.7
    })                                              // Shrink to 70% at warning
    .on_critical(PressureReaction::Callback(Box::new(|level| {
        eprintln!("Critical memory pressure: {:?}", level);
        // Trigger emergency cleanup
    })));

// Check and handle pressure
let level = handler.check_pressure(&pool);
match level {
    PressureLevel::Normal => { /* Continue normally */ }
    PressureLevel::Elevated => { /* Maybe defer new allocations */ }
    PressureLevel::Warning => { /* Release caches */ }
    PressureLevel::Critical => { /* Emergency measures */ }
    PressureLevel::OutOfMemory => { /* Fail gracefully */ }
}
```

### Pressure Reactions

| Reaction | Description |
|----------|-------------|
| `None` | Ignore pressure at this level |
| `Shrink { target_utilization }` | Release buffers until utilization drops to target |
| `Callback(fn)` | Execute custom callback for application-specific handling |

---

## CUDA Reduction Buffer Cache (v0.3.0)

For algorithms requiring repeated reductions (PageRank iterations, convergence checks):

```rust
use ringkernel_cuda::reduction::{ReductionBufferCache, ReductionOp};

// Create cache with max 4 buffers per key
let cache = ReductionBufferCache::new(&device, 4);

// Acquire reduction buffer (allocates on first call)
let buffer = cache.acquire::<f32>(4, ReductionOp::Sum)?;

// Use for reduction...
// buffer.device_ptr() for kernel
// buffer.read_result(slot) for host read

// Buffer automatically returned to cache on drop
drop(buffer);

// Next acquire with same key reuses the buffer
let reused = cache.acquire::<f32>(4, ReductionOp::Sum)?;  // Cache hit!

// Check cache statistics
let stats = cache.stats();
println!("Hit rate: {:.1}%", stats.hit_rate() * 100.0);
```

---

## WebGPU Staging Buffer Pool (v0.3.0)

Efficient staging buffer reuse for GPU-to-host transfers:

```rust
use ringkernel_wgpu::memory::{StagingBufferPool, StagingPoolStats};

// Create staging pool
let pool = StagingBufferPool::new(&device, 16);  // Max 16 buffers

// Acquire staging buffer for readback
let staging = pool.acquire(data_size)?;

// Use for GPU → host copy
encoder.copy_buffer_to_buffer(&gpu_buffer, 0, staging.buffer(), 0, data_size);

// Map and read
staging.map_async(wgpu::MapMode::Read);
device.poll(wgpu::Maintain::Wait);
let data = staging.get_mapped_range();

// Buffer returned to pool on drop
drop(staging);

// Check statistics
let stats = pool.stats();
println!("Staging buffer reuse rate: {:.1}%", stats.hit_rate() * 100.0);
```

---

## Unified Memory (CUDA Managed Memory)

```rust
/// Unified memory accessible from CPU and GPU.
pub struct UnifiedBuffer<T: Copy + Send> {
    ptr: NonNull<T>,
    len: usize,
    allocator: Box<dyn UnifiedAllocator>,
    _marker: PhantomData<T>,
}

impl<T: Copy + Send> UnifiedBuffer<T> {
    pub fn new(len: usize, allocator: Box<dyn UnifiedAllocator>) -> Result<Self> {
        let ptr = allocator.allocate_managed(len * std::mem::size_of::<T>())?;
        Ok(Self {
            ptr: NonNull::new(ptr as *mut T).ok_or(Error::AllocationFailed)?,
            len,
            allocator,
            _marker: PhantomData,
        })
    }

    /// Access from CPU (may trigger page migration).
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Prefetch to GPU.
    pub fn prefetch_to_device(&self, device_id: i32) -> Result<()> {
        self.allocator.prefetch(self.ptr.as_ptr() as *const u8, self.len, device_id)
    }
}
```

---

## WSL2 Memory Visibility Issues

**Critical**: WSL2 has limited GPU memory coherence (see CLAUDE.md).

```rust
/// Memory visibility strategy for different platforms.
#[derive(Debug, Clone, Copy)]
pub enum MemoryVisibility {
    /// Full system-scope atomics (native Linux)
    SystemScope,

    /// Device-scope only, requires explicit sync (WSL2)
    DeviceScope,

    /// Explicit DMA transfers only
    ExplicitDma,
}

impl MemoryVisibility {
    /// Detect platform capabilities.
    pub fn detect() -> Self {
        #[cfg(target_os = "linux")]
        {
            if is_wsl2() {
                return Self::DeviceScope; // WSL2 limitation
            }
            Self::SystemScope
        }

        #[cfg(target_os = "windows")]
        {
            Self::SystemScope // Native Windows CUDA works
        }

        #[cfg(target_os = "macos")]
        {
            Self::DeviceScope // Metal doesn't have system atomics
        }
    }
}

fn is_wsl2() -> bool {
    std::fs::read_to_string("/proc/version")
        .map(|v| v.contains("microsoft") || v.contains("WSL"))
        .unwrap_or(false)
}
```

---

## Memory Transfer Strategies

```rust
/// Strategy for host↔device data transfer.
pub enum TransferStrategy {
    /// Synchronous copy (blocking)
    Sync,

    /// Async copy with stream
    Async { stream: StreamHandle },

    /// Zero-copy via unified memory
    Unified,

    /// Explicit DMA with polling
    DmaPolling { poll_interval_us: u32 },
}

/// Bridge for host↔GPU ring buffer synchronization.
pub struct RingBufferBridge<T: RingMessage> {
    host_buffer: PinnedBuffer<T>,
    gpu_buffer: GpuBuffer<T>,
    strategy: TransferStrategy,

    // Transfer tracking
    host_to_gpu_count: AtomicU64,
    gpu_to_host_count: AtomicU64,
}

impl<T: RingMessage> RingBufferBridge<T> {
    /// Transfer pending messages from host to GPU.
    pub async fn flush_to_device(&mut self) -> Result<usize> {
        match &self.strategy {
            TransferStrategy::Sync => {
                self.sync_copy_to_device()
            }
            TransferStrategy::Async { stream } => {
                self.async_copy_to_device(stream).await
            }
            TransferStrategy::Unified => {
                // No-op, unified memory handles it
                Ok(0)
            }
            TransferStrategy::DmaPolling { poll_interval_us } => {
                self.dma_poll_to_device(*poll_interval_us).await
            }
        }
    }
}
```

---

## Cache-Line Alignment

Critical for GPU performance:

```rust
/// Ensure cache-line alignment for GPU structures.
#[repr(C, align(128))]
pub struct CacheLineAligned<T> {
    value: T,
}

/// Control block must be 128-byte aligned (dual cache line).
#[repr(C, align(128))]
pub struct ControlBlock {
    // ... fields (see 01-architecture-overview.md)
}

/// Telemetry buffer is 64-byte aligned (single cache line).
#[repr(C, align(64))]
pub struct TelemetryBuffer {
    // ... fields
}

// Compile-time size assertions
const _: () = assert!(std::mem::size_of::<ControlBlock>() == 128);
const _: () = assert!(std::mem::size_of::<TelemetryBuffer>() == 64);
```

---

## Next: [GPU Backends](./05-gpu-backends.md)
