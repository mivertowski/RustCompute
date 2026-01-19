---
layout: default
title: Core Abstractions
nav_order: 4
---

# Core Abstractions

## Mapping DotCompute Types to Rust

This document details the core traits and types that form the Ring Kernel abstraction layer.

---

## 1. RingMessage Trait

**DotCompute**: `IRingKernelMessage`

```rust
// crates/ringkernel-core/src/message.rs

use std::any::TypeId;

/// Marker trait for messages that can be sent between ring kernels.
///
/// Implementations are typically derived using `#[derive(RingMessage)]`.
pub trait RingMessage: Send + Sync + Sized + 'static {
    /// Unique message type identifier for routing/deserialization.
    const MESSAGE_TYPE: &'static str;

    /// Returns the message ID (for tracking/deduplication).
    fn message_id(&self) -> Uuid;

    /// Sets the message ID.
    fn set_message_id(&mut self, id: Uuid);

    /// Priority level (0-255, higher = more urgent).
    fn priority(&self) -> u8;

    /// Optional correlation ID for request/response patterns.
    fn correlation_id(&self) -> Option<Uuid>;

    /// Serialized size in bytes.
    fn payload_size(&self) -> usize;
}

/// Priority levels following DotCompute conventions.
pub mod priority {
    pub const LOW: u8 = 32;        // 0-63: Batch processing
    pub const NORMAL: u8 = 128;    // 64-127: Default
    pub const HIGH: u8 = 160;      // 128-191: Interactive
    pub const CRITICAL: u8 = 224;  // 192-255: System messages
}
```

### Derive Macro Usage

```rust
use ringkernel::prelude::*;
use rkyv::{Archive, Serialize, Deserialize};

#[derive(RingMessage, Archive, Serialize, Deserialize)]
#[rkyv(compare(PartialEq))]
pub struct VectorAddRequest {
    #[message(id)]
    pub id: Uuid,

    #[message(priority = "priority::NORMAL")]
    pub priority: u8,

    #[message(correlation)]
    pub correlation_id: Option<Uuid>,

    // Payload fields
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}
```

---

## 2. MessageQueue Trait

**DotCompute**: `IMessageQueue<T>`

```rust
// crates/ringkernel-core/src/queue.rs

use std::future::Future;

/// Lock-free ring buffer for message passing.
pub trait MessageQueue<T: RingMessage>: Send + Sync {
    /// Maximum capacity (must be power of 2).
    fn capacity(&self) -> usize;

    /// Current message count.
    fn len(&self) -> usize;

    /// Check if queue is empty.
    fn is_empty(&self) -> bool { self.len() == 0 }

    /// Check if queue is full.
    fn is_full(&self) -> bool { self.len() >= self.capacity() - 1 }

    /// Non-blocking enqueue attempt.
    fn try_enqueue(&self, message: T) -> Result<(), QueueError<T>>;

    /// Non-blocking dequeue attempt.
    fn try_dequeue(&self) -> Option<T>;

    /// Blocking enqueue with timeout.
    fn enqueue(&self, message: T, timeout: Duration) -> impl Future<Output = Result<(), QueueError<T>>> + Send;

    /// Blocking dequeue with timeout.
    fn dequeue(&self, timeout: Duration) -> impl Future<Output = Result<T, QueueError<T>>> + Send;

    /// Get queue statistics.
    fn statistics(&self) -> QueueStatistics;
}

/// Queue operation errors.
#[derive(Debug, thiserror::Error)]
pub enum QueueError<T> {
    #[error("Queue is full")]
    Full(T),

    #[error("Queue is empty")]
    Empty,

    #[error("Operation timed out")]
    Timeout,

    #[error("Queue has been closed")]
    Closed,
}

/// Queue performance statistics.
#[derive(Debug, Clone, Default)]
pub struct QueueStatistics {
    pub total_enqueued: u64,
    pub total_dequeued: u64,
    pub total_dropped: u64,
    pub average_latency_ns: u64,
    pub peak_depth: usize,
}
```

---

## 3. RingKernelRuntime Trait

**DotCompute**: `IRingKernelRuntime`

```rust
// crates/ringkernel-core/src/runtime.rs

use async_trait::async_trait;

/// Runtime for managing ring kernel lifecycles.
#[async_trait]
pub trait RingKernelRuntime: Send + Sync {
    /// Backend identifier (e.g., "cuda", "metal", "wgpu").
    fn backend_name(&self) -> &'static str;

    /// Launch a kernel (initially inactive).
    async fn launch(
        &self,
        kernel_id: &str,
        grid_size: Dim3,
        block_size: Dim3,
        options: LaunchOptions,
    ) -> Result<KernelHandle>;

    /// Activate a launched kernel (begin processing).
    async fn activate(&self, kernel_id: &str) -> Result<()>;

    /// Deactivate a kernel (pause, preserve state).
    async fn deactivate(&self, kernel_id: &str) -> Result<()>;

    /// Terminate a kernel (cleanup resources).
    async fn terminate(&self, kernel_id: &str) -> Result<()>;

    /// Send a message to a kernel.
    async fn send<T: RingMessage>(&self, kernel_id: &str, message: T) -> Result<()>;

    /// Receive a message from a kernel.
    async fn receive<T: RingMessage>(&self, kernel_id: &str, timeout: Duration) -> Result<T>;

    /// Get kernel status.
    async fn status(&self, kernel_id: &str) -> Result<KernelStatus>;

    /// Get kernel metrics.
    async fn metrics(&self, kernel_id: &str) -> Result<KernelMetrics>;

    /// Get real-time telemetry (<1μs latency).
    async fn telemetry(&self, kernel_id: &str) -> Result<TelemetrySnapshot>;

    /// List all managed kernels.
    async fn list_kernels(&self) -> Result<Vec<String>>;
}

/// Grid/block dimensions.
#[derive(Debug, Clone, Copy, Default)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    pub fn linear(size: u32) -> Self {
        Self { x: size, y: 1, z: 1 }
    }
}
```

---

## 4. RingContext Struct

**DotCompute**: `RingKernelContext` (ref struct)

```rust
// crates/ringkernel-core/src/context.rs

/// Runtime context passed to ring kernel handlers.
///
/// Methods translate to GPU intrinsics during compilation.
pub struct RingContext<'a> {
    // Internal pointers (populated by runtime)
    control_block: &'a ControlBlock,
    input_queue: *const u8,
    output_queue: *mut u8,
    hlc: &'a mut HlcState,
}

impl<'a> RingContext<'a> {
    // ═══════════════════════════════════════════════════════════════
    // Thread Identity
    // ═══════════════════════════════════════════════════════════════

    /// Thread index within block (→ threadIdx.x)
    #[inline(always)]
    pub fn thread_id(&self) -> u32 { /* intrinsic */ 0 }

    /// Block index within grid (→ blockIdx.x)
    #[inline(always)]
    pub fn block_id(&self) -> u32 { /* intrinsic */ 0 }

    /// Warp index (thread_id / 32)
    #[inline(always)]
    pub fn warp_id(&self) -> u32 { self.thread_id() / 32 }

    /// Lane within warp (thread_id % 32)
    #[inline(always)]
    pub fn lane_id(&self) -> u32 { self.thread_id() % 32 }

    /// Global thread index (→ blockIdx.x * blockDim.x + threadIdx.x)
    #[inline(always)]
    pub fn global_thread_id(&self) -> u32 { /* intrinsic */ 0 }

    // ═══════════════════════════════════════════════════════════════
    // Synchronization Barriers
    // ═══════════════════════════════════════════════════════════════

    /// Block-level barrier (→ __syncthreads())
    #[inline(always)]
    pub fn sync_threads(&self) { /* intrinsic */ }

    /// Grid-level barrier (→ cooperative_groups::grid::sync())
    #[inline(always)]
    pub fn sync_grid(&self) { /* intrinsic */ }

    /// Warp-level barrier (→ __syncwarp(mask))
    #[inline(always)]
    pub fn sync_warp(&self, mask: u32) { /* intrinsic */ }

    /// Named barrier for cross-kernel sync
    pub fn named_barrier(&self, name: &str) { /* runtime */ }

    // ═══════════════════════════════════════════════════════════════
    // Temporal Operations (HLC)
    // ═══════════════════════════════════════════════════════════════

    /// Get current HLC timestamp (→ clock64() + logical)
    #[inline(always)]
    pub fn now(&self) -> HlcTimestamp { /* intrinsic */ HlcTimestamp::default() }

    /// Advance local clock (local event)
    #[inline(always)]
    pub fn tick(&mut self) { /* intrinsic */ }

    /// Merge received timestamp (causal ordering)
    #[inline(always)]
    pub fn update_clock(&mut self, received: HlcTimestamp) { /* intrinsic */ }

    // ═══════════════════════════════════════════════════════════════
    // Memory Ordering
    // ═══════════════════════════════════════════════════════════════

    /// Device-scope fence (→ __threadfence())
    #[inline(always)]
    pub fn thread_fence(&self) { /* intrinsic */ }

    /// Block-scope fence (→ __threadfence_block())
    #[inline(always)]
    pub fn thread_fence_block(&self) { /* intrinsic */ }

    /// System-scope fence (→ __threadfence_system())
    #[inline(always)]
    pub fn thread_fence_system(&self) { /* intrinsic */ }

    // ═══════════════════════════════════════════════════════════════
    // Atomic Operations
    // ═══════════════════════════════════════════════════════════════

    /// Atomic add (→ atomicAdd)
    #[inline(always)]
    pub fn atomic_add(&self, target: &mut i32, value: i32) -> i32 { /* intrinsic */ 0 }

    /// Atomic CAS (→ atomicCAS)
    #[inline(always)]
    pub fn atomic_cas(&self, target: &mut i32, compare: i32, value: i32) -> i32 { /* intrinsic */ 0 }

    /// Atomic exchange (→ atomicExch)
    #[inline(always)]
    pub fn atomic_exch(&self, target: &mut i32, value: i32) -> i32 { /* intrinsic */ 0 }

    // ═══════════════════════════════════════════════════════════════
    // Warp Primitives
    // ═══════════════════════════════════════════════════════════════

    /// Warp shuffle (→ __shfl_sync)
    #[inline(always)]
    pub fn warp_shuffle(&self, value: i32, src_lane: u32, mask: u32) -> i32 { /* intrinsic */ 0 }

    /// Warp ballot (→ __ballot_sync)
    #[inline(always)]
    pub fn warp_ballot(&self, predicate: bool, mask: u32) -> u32 { /* intrinsic */ 0 }

    // ═══════════════════════════════════════════════════════════════
    // Queue Operations
    // ═══════════════════════════════════════════════════════════════

    /// Enqueue output message
    pub fn enqueue_output<T: RingMessage>(&mut self, message: T) -> bool { false }

    /// Check output queue capacity
    pub fn output_queue_free_slots(&self) -> usize { 0 }

    // ═══════════════════════════════════════════════════════════════
    // K2K Messaging
    // ═══════════════════════════════════════════════════════════════

    /// Send to another kernel
    pub fn send_to_kernel<T: RingMessage>(&mut self, target: &str, message: T) -> bool { false }

    /// Receive from another kernel
    pub fn try_receive_from_kernel<T: RingMessage>(&mut self, source: &str) -> Option<T> { None }

    // ═══════════════════════════════════════════════════════════════
    // Control
    // ═══════════════════════════════════════════════════════════════

    /// Request graceful termination
    pub fn request_termination(&self) { /* write control block */ }

    /// Check if termination requested
    pub fn is_termination_requested(&self) -> bool { false }

    /// Report an error
    pub fn report_error(&self) { /* atomic increment */ }
}
```

---

## 5. HlcTimestamp

**DotCompute**: `HlcTimestamp` (Hybrid Logical Clock)

```rust
// crates/ringkernel-core/src/hlc.rs

use std::cmp::Ordering;

/// Hybrid Logical Clock timestamp for causal ordering.
///
/// Combines physical time (wall clock) with logical counter
/// to provide total ordering even when physical clocks disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct HlcTimestamp {
    /// Physical time component (nanoseconds since epoch or GPU ticks)
    physical: u64,

    /// Logical counter for events at same physical time
    logical: u32,

    /// Node/kernel identifier (for tie-breaking)
    node_id: u32,
}

impl HlcTimestamp {
    /// Create a new timestamp.
    pub fn new(physical: u64, logical: u32, node_id: u32) -> Self {
        Self { physical, logical, node_id }
    }

    /// Get physical time component.
    pub fn physical(&self) -> u64 { self.physical }

    /// Get logical counter.
    pub fn logical(&self) -> u32 { self.logical }

    /// Advance for local event (tick).
    pub fn tick(&mut self) {
        self.logical = self.logical.wrapping_add(1);
    }

    /// Merge with received timestamp (max + 1).
    pub fn merge(&mut self, other: &HlcTimestamp) {
        if other.physical > self.physical {
            self.physical = other.physical;
            self.logical = other.logical + 1;
        } else if other.physical == self.physical {
            self.logical = self.logical.max(other.logical) + 1;
        } else {
            self.logical += 1;
        }
    }

    /// Update physical time if wall clock advanced.
    pub fn update_physical(&mut self, wall_clock: u64) {
        if wall_clock > self.physical {
            self.physical = wall_clock;
            self.logical = 0;
        }
    }
}

impl Ord for HlcTimestamp {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.physical.cmp(&other.physical) {
            Ordering::Equal => match self.logical.cmp(&other.logical) {
                Ordering::Equal => self.node_id.cmp(&other.node_id),
                other => other,
            },
            other => other,
        }
    }
}

impl PartialOrd for HlcTimestamp {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
```

---

## 6. ControlBlock and TelemetryBuffer

See [Architecture Overview](./01-architecture-overview.md) for the full struct definitions.

---

## 7. Enterprise Runtime (v0.2.0+)

RingKernel provides enterprise-grade runtime features for production deployments.

### RuntimeBuilder

Fluent builder for configuring the enterprise runtime:

```rust
use ringkernel_core::runtime_context::RuntimeBuilder;

// Quick presets
let dev_runtime = RuntimeBuilder::new().development().build()?;
let prod_runtime = RuntimeBuilder::new().production().build()?;
let perf_runtime = RuntimeBuilder::new().high_performance().build()?;

// Custom configuration
let runtime = RuntimeBuilder::new()
    .with_health_check_interval(Duration::from_secs(30))
    .with_circuit_breaker_threshold(5)
    .with_prometheus_export(true)
    .build()?;
```

### RingKernelContext

Unified runtime managing all enterprise features:

```rust
use ringkernel_core::runtime_context::RingKernelContext;

let ctx = RingKernelContext::new(config)?;

// Lifecycle management
ctx.start()?;                        // Transition to Running
ctx.drain(Duration::from_secs(30))?; // Stop accepting new work
let report = ctx.complete_shutdown()?;

// Access components
let health = ctx.health_checker();
let circuit = ctx.circuit_breaker();
let metrics = ctx.prometheus_exporter();
let coordinator = ctx.multi_gpu_coordinator();
```

### LifecycleState

Runtime state machine:

```
Initializing → Running → Draining → ShuttingDown → Stopped
```

```rust
pub enum LifecycleState {
    Initializing,   // Setting up components
    Running,        // Accepting and processing work
    Draining,       // Finishing existing work, rejecting new
    ShuttingDown,   // Cleanup in progress
    Stopped,        // Terminated
}
```

### ConfigBuilder

Nested configuration system:

```rust
use ringkernel_core::config::ConfigBuilder;

let config = ConfigBuilder::new()
    .runtime(|r| r
        .max_kernels(100)
        .default_queue_capacity(4096))
    .health(|h| h
        .check_interval(Duration::from_secs(10))
        .degradation_threshold(0.8))
    .multi_gpu(|g| g
        .load_balancing(LoadBalancing::LeastLoaded)
        .enable_migration(true))
    .build()?;
```

---

## 8. Domain System (v0.3.0)

Organize messages by business domain with automatic type ID allocation:

```rust
use ringkernel_core::domain::{Domain, DomainMessage};

/// 20 predefined business domains with reserved type ID ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Domain {
    General,              // 0-99 (default)
    GraphAnalytics,       // 100-199
    StatisticalML,        // 200-299
    Compliance,           // 300-399
    RiskManagement,       // 400-499
    OrderMatching,        // 500-599
    MarketData,           // 600-699
    Settlement,           // 700-799
    Accounting,           // 800-899
    NetworkAnalysis,      // 900-999
    FraudDetection,       // 1000-1099
    TimeSeries,           // 1100-1199
    Simulation,           // 1200-1299
    Banking,              // 1300-1399
    BehavioralAnalytics,  // 1400-1499
    ProcessIntelligence,  // 1500-1599
    Clearing,             // 1600-1699
    TreasuryManagement,   // 1700-1799
    PaymentProcessing,    // 1800-1899
    FinancialAudit,       // 1900-1999
    Custom,               // 10000+ (user-defined)
}

impl Domain {
    /// Get the type ID range for this domain.
    pub fn type_id_range(&self) -> (u32, u32) { /* ... */ }

    /// Determine domain from a type ID.
    pub fn from_type_id(type_id: u32) -> Option<Self> { /* ... */ }
}
```

### Usage with RingMessage

```rust
use ringkernel::prelude::*;

#[derive(RingMessage)]
#[message(domain = "FraudDetection")]  // Auto-assigns type ID in 1000-1099 range
struct SuspiciousActivity {
    #[message(id)]
    id: MessageId,
    account_id: u64,
    risk_score: f32,
}

#[derive(RingMessage)]
#[message(domain = "ProcessIntelligence")]  // Type ID in 1500-1599 range
struct ActivityEvent {
    #[message(id)]
    id: MessageId,
    case_id: u64,
    activity: String,
    timestamp: u64,
}
```

---

## 9. PersistentMessage Trait (v0.3.0)

For type-based GPU kernel dispatch with automatic handler routing:

```rust
use ringkernel_core::persistent_message::{PersistentMessage, HandlerId, DispatchTable};

/// Trait for messages that can be dispatched to GPU kernel handlers.
pub trait PersistentMessage: RingMessage {
    /// Unique handler identifier for dispatch routing.
    const HANDLER_ID: HandlerId;

    /// Whether this message type expects a response.
    const REQUIRES_RESPONSE: bool = false;

    /// Serialize payload for GPU transfer.
    fn serialize_payload(&self) -> Vec<u8>;

    /// Deserialize response from GPU.
    fn deserialize_response(data: &[u8]) -> Option<Self::Response>
    where
        Self::Response: Sized;

    /// Associated response type (if any).
    type Response: RingMessage = ();
}

/// Handler identifier for dispatch routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HandlerId(pub u32);

/// Runtime dispatch table for message→handler mapping.
pub struct DispatchTable {
    handlers: HashMap<HandlerId, Box<dyn Handler>>,
}
```

### Derive Macro Usage

```rust
use ringkernel::prelude::*;

#[derive(PersistentMessage)]
#[persistent_message(handler_id = 1, requires_response = true)]
struct ComputeRequest {
    data: Vec<f32>,
    operation: OperationType,
}

#[derive(PersistentMessage)]
#[persistent_message(handler_id = 1)]  // Same handler as request
struct ComputeResponse {
    result: Vec<f32>,
    elapsed_ns: u64,
}
```

---

## 10. KernelDispatcher (v0.3.0)

Multi-kernel message routing component:

```rust
use ringkernel_core::dispatcher::{
    KernelDispatcher, DispatcherBuilder, DispatcherConfig, DispatcherMetrics
};

/// Routes messages to GPU kernels based on message type.
pub struct KernelDispatcher {
    broker: Arc<K2KBroker>,
    config: DispatcherConfig,
    metrics: DispatcherMetrics,
}

impl KernelDispatcher {
    /// Dispatch a message to the appropriate kernel.
    pub async fn dispatch<T: PersistentMessage>(
        &self,
        kernel_id: KernelId,
        message: T,
    ) -> Result<DispatchReceipt, DispatchError>;

    /// Dispatch with explicit priority override.
    pub async fn dispatch_priority<T: PersistentMessage>(
        &self,
        kernel_id: KernelId,
        message: T,
        priority: Priority,
    ) -> Result<DispatchReceipt, DispatchError>;

    /// Get dispatch metrics.
    pub fn metrics(&self) -> &DispatcherMetrics;
}
```

### Builder Pattern

```rust
use ringkernel_core::dispatcher::DispatcherBuilder;

let dispatcher = DispatcherBuilder::new(k2k_broker)
    .with_default_priority(Priority::Normal)
    .with_timeout(Duration::from_millis(100))
    .with_retry_policy(RetryPolicy::exponential(3))
    .build();

// Dispatch messages
let receipt = dispatcher.dispatch(kernel_id, compute_request).await?;

// Check metrics
let metrics = dispatcher.metrics();
println!("Dispatched: {}", metrics.messages_dispatched);
println!("Errors: {}", metrics.errors);
println!("Avg latency: {:?}", metrics.average_latency);
```

---

## 11. Queue Tiering (v0.3.0)

Predefined queue capacity tiers for different throughput requirements:

```rust
use ringkernel_core::queue::{QueueTier, QueueFactory, QueueMonitor, QueueMetrics};

/// Queue capacity tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueueTier {
    Small,      // 256 messages - low traffic
    Medium,     // 1024 messages - moderate traffic (default)
    Large,      // 4096 messages - high traffic
    ExtraLarge, // 16384 messages - very high traffic
}

impl QueueTier {
    /// Get capacity for this tier.
    pub fn capacity(&self) -> usize;

    /// Suggest tier based on expected throughput.
    pub fn for_throughput(messages_per_second: u64, target_headroom_ms: u64) -> Self;

    /// Upgrade to next tier.
    pub fn upgrade(&self) -> Self;

    /// Downgrade to previous tier.
    pub fn downgrade(&self) -> Self;
}
```

### Queue Factory

```rust
use ringkernel_core::queue::{QueueFactory, QueueTier};

// Create queues with specific tiers
let small_queue = QueueFactory::create_spsc(QueueTier::Small);
let large_queue = QueueFactory::create_mpsc(QueueTier::Large);

// Auto-select tier based on throughput requirements
let tier = QueueTier::for_throughput(10_000, 100);  // 10k msg/s, 100ms buffer
let queue = QueueFactory::create_spsc(tier);
```

### Queue Monitor

```rust
use ringkernel_core::queue::{QueueMonitor, QueueMonitorConfig};

let monitor = QueueMonitor::new(queue, QueueMonitorConfig {
    warning_threshold: 0.75,   // Warn at 75% capacity
    critical_threshold: 0.90,  // Critical at 90%
    sample_interval: Duration::from_millis(100),
});

// Check queue health
let health = monitor.check_health();
println!("Depth: {}/{}", health.current_depth, health.capacity);
println!("Healthy: {}", health.is_healthy);
println!("Level: {:?}", health.pressure_level);

// Get detailed metrics
let metrics = monitor.metrics();
println!("Enqueued: {}, Dequeued: {}", metrics.total_enqueued, metrics.total_dequeued);
println!("Peak depth: {}", metrics.peak_depth);
```

---

## 12. Global Reduction Primitives (v0.3.0)

Backend-agnostic reduction operations for GPU algorithms:

```rust
use ringkernel_core::reduction::{ReductionOp, ReductionScalar, GlobalReduction};

/// Supported reduction operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReductionOp {
    Sum,     // Σ values
    Min,     // minimum value
    Max,     // maximum value
    And,     // bitwise AND
    Or,      // bitwise OR
    Xor,     // bitwise XOR
    Product, // Π values
}

/// Trait for types that can participate in reductions.
pub trait ReductionScalar: Copy + Send + Sync + 'static {
    /// Identity element for the given operation.
    fn identity(op: ReductionOp) -> Self;

    /// Apply reduction operation.
    fn reduce(self, other: Self, op: ReductionOp) -> Self;
}

// Implemented for: i32, i64, u32, u64, f32, f64

/// Backend-agnostic global reduction interface.
pub trait GlobalReduction {
    /// Perform a grid-wide reduction.
    fn reduce<T: ReductionScalar>(
        &self,
        input: &[T],
        op: ReductionOp,
    ) -> Result<T, ReductionError>;

    /// Perform reduction and broadcast result to all threads.
    fn reduce_and_broadcast<T: ReductionScalar>(
        &self,
        input: &[T],
        op: ReductionOp,
    ) -> Result<T, ReductionError>;
}
```

### CUDA Reduction Buffer

For PageRank-style algorithms with inter-kernel reductions:

```rust
use ringkernel_cuda::reduction::{ReductionBuffer, ReductionBufferBuilder};

// Create reduction buffer using mapped memory (zero-copy host read)
let buffer = ReductionBufferBuilder::new()
    .with_op(ReductionOp::Sum)
    .with_slots(4)  // Multiple slots reduce contention
    .build(&device)?;

// In kernel: accumulate to buffer
// grid_reduce_sum(local_value, shared_mem, buffer.device_ptr())

// Host: read result without explicit copy
let sum = buffer.read_result(0)?;
```

### Multi-Phase Kernel Execution

For algorithms requiring synchronization between compute phases:

```rust
use ringkernel_cuda::phases::{MultiPhaseConfig, SyncMode, KernelPhase};

let config = MultiPhaseConfig::new()
    .with_sync_mode(SyncMode::Cooperative)  // Uses grid.sync() on CC 6.0+
    .with_phase(KernelPhase::new("scatter", scatter_fn))
    .with_phase(KernelPhase::new("gather", gather_fn))
    .with_reduction_between_phases(ReductionOp::Sum);

let executor = MultiPhaseExecutor::new(&device, config)?;
let stats = executor.run(iterations)?;

println!("Total time: {:?}", stats.total_time);
println!("Sync overhead: {:?}", stats.sync_overhead);
```

---

## Next: [Memory Management](./04-memory-management.md)
