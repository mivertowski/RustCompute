# Persistent Kernel Specification

> Backend-Agnostic GPU Actor Model for RingKernel

## Overview

This specification defines the persistent kernel architecture that enables GPU kernels to operate as long-lived actors with sub-microsecond message passing. The design abstracts over hardware differences while maximizing performance on each backend.

---

## Core Concepts

### 1. Persistent Kernel Lifecycle

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Kernel Lifecycle                              │
└─────────────────────────────────────────────────────────────────────┘

    ┌──────────┐    launch()    ┌──────────┐    activate()   ┌────────┐
    │ Created  │ ──────────────▶│ Launched │ ──────────────▶│ Active │
    └──────────┘                └──────────┘                 └────────┘
                                                                  │
                    ┌─────────────────────────────────────────────┤
                    │                                             │
                    ▼                                             ▼
              ┌──────────┐     resume()      ┌──────────┐    terminate()
              │  Paused  │ ◀────────────────▶│  Active  │ ─────────────▶
              └──────────┘     pause()       └──────────┘
                    │                                             │
                    │            terminate()                      │
                    └─────────────────────────────────────────────┤
                                                                  ▼
                                                           ┌─────────────┐
                                                           │ Terminated  │
                                                           └─────────────┘
```

### 2. Execution Models by Backend

| Backend | Model | Description |
|---------|-------|-------------|
| **CUDA** | True Persistent | Single kernel launch, runs for lifetime |
| **Metal** | Indirect Command Buffer | ICB-based persistence |
| **WebGPU** | Host-Driven Loop | Efficient dispatch batching |
| **CPU** | Async Task | Tokio task-based simulation |

---

## Memory Architecture

### Control Block (256 bytes, 64-byte aligned)

The control block is the shared state between host and GPU, residing in mapped/shared memory for zero-copy access.

```rust
/// Core control block for persistent kernel lifecycle management
#[repr(C, align(64))]
pub struct PersistentControlBlock {
    // ═══════════════════════════════════════════════════════════════════
    // Cache Line 0: Status and Synchronization (64 bytes)
    // ═══════════════════════════════════════════════════════════════════
    /// Kernel status (see KernelStatus enum)
    pub status: AtomicU32,              // 4 bytes
    /// Error code if status == Error
    pub error_code: AtomicU32,          // 4 bytes
    /// Barrier for grid-wide synchronization
    pub sync_barrier: AtomicU32,        // 4 bytes
    /// Number of blocks that have reached barrier
    pub sync_count: AtomicU32,          // 4 bytes
    /// Total number of thread blocks
    pub grid_size: u32,                 // 4 bytes
    /// Reserved for future use
    _pad0: [u32; 11],                   // 44 bytes

    // ═══════════════════════════════════════════════════════════════════
    // Cache Line 1: Step Counters (64 bytes)
    // ═══════════════════════════════════════════════════════════════════
    /// Current simulation step (completed)
    pub current_step: AtomicU64,        // 8 bytes
    /// Target step (host writes, kernel reads)
    pub target_step: AtomicU64,         // 8 bytes
    /// Steps executed since last progress report
    pub steps_since_report: AtomicU64,  // 8 bytes
    /// Interval for progress reporting
    pub progress_interval: u64,         // 8 bytes
    /// Reserved for future use
    _pad1: [u64; 4],                    // 32 bytes

    // ═══════════════════════════════════════════════════════════════════
    // Cache Line 2: H2K Queue Pointers (64 bytes)
    // ═══════════════════════════════════════════════════════════════════
    /// H2K queue head (host writes)
    pub h2k_head: AtomicU32,            // 4 bytes
    /// H2K queue tail (kernel reads)
    pub h2k_tail: AtomicU32,            // 4 bytes
    /// H2K queue capacity (power of 2)
    pub h2k_capacity: u32,              // 4 bytes
    /// H2K queue mask (capacity - 1)
    pub h2k_mask: u32,                  // 4 bytes
    /// K2H queue head (kernel writes)
    pub k2h_head: AtomicU32,            // 4 bytes
    /// K2H queue tail (host reads)
    pub k2h_tail: AtomicU32,            // 4 bytes
    /// K2H queue capacity
    pub k2h_capacity: u32,              // 4 bytes
    /// K2H queue mask
    pub k2h_mask: u32,                  // 4 bytes
    /// Reserved for future use
    _pad2: [u32; 8],                    // 32 bytes

    // ═══════════════════════════════════════════════════════════════════
    // Cache Line 3: HLC and Timing (64 bytes)
    // ═══════════════════════════════════════════════════════════════════
    /// HLC physical clock (nanoseconds since epoch)
    pub hlc_physical: AtomicU64,        // 8 bytes
    /// HLC logical counter
    pub hlc_logical: AtomicU32,         // 4 bytes
    /// HLC node ID for this kernel
    pub hlc_node_id: u32,               // 4 bytes
    /// Kernel launch timestamp
    pub launch_time_ns: u64,            // 8 bytes
    /// Total kernel execution time
    pub execution_time_ns: AtomicU64,   // 8 bytes
    /// Reserved for future use
    _pad3: [u64; 4],                    // 32 bytes
}

/// Kernel status values
#[repr(u32)]
pub enum KernelStatus {
    /// Kernel created but not yet running
    Created = 0,
    /// Kernel launched and initializing
    Launching = 1,
    /// Kernel active and processing
    Active = 2,
    /// Kernel paused, waiting for resume
    Paused = 3,
    /// Kernel terminating
    Terminating = 4,
    /// Kernel terminated normally
    Terminated = 5,
    /// Kernel encountered error
    Error = 6,
}
```

### Message Header (256 bytes, 64-byte aligned)

All messages use a standardized header for routing, correlation, and HLC timestamps.

```rust
/// Message header for envelope-based communication
#[repr(C, align(64))]
pub struct MessageHeader {
    // ═══════════════════════════════════════════════════════════════════
    // Cache Line 0: Identity and Routing (64 bytes)
    // ═══════════════════════════════════════════════════════════════════
    /// Magic number for validation (0x52494E474B45524E = "RINGKERN")
    pub magic: u64,                     // 8 bytes
    /// Message type ID (user-defined)
    pub type_id: u32,                   // 4 bytes
    /// Message flags (see MessageFlags)
    pub flags: u32,                     // 4 bytes
    /// Source kernel ID
    pub source_kernel: u64,             // 8 bytes
    /// Destination kernel ID
    pub dest_kernel: u64,               // 8 bytes
    /// Unique message ID
    pub message_id: u64,                // 8 bytes
    /// Correlation ID for request/response tracking
    pub correlation_id: u64,            // 8 bytes
    /// Payload length in bytes
    pub payload_len: u32,               // 4 bytes
    /// Checksum of payload (CRC32)
    pub checksum: u32,                  // 4 bytes
    /// Reserved
    _pad0: [u64; 1],                    // 8 bytes

    // ═══════════════════════════════════════════════════════════════════
    // Cache Line 1: HLC Timestamp (64 bytes)
    // ═══════════════════════════════════════════════════════════════════
    /// HLC physical clock when message was created
    pub hlc_physical: u64,              // 8 bytes
    /// HLC logical counter
    pub hlc_logical: u32,               // 4 bytes
    /// HLC node ID of sender
    pub hlc_node_id: u32,               // 4 bytes
    /// Monotonic timestamp for latency tracking
    pub mono_timestamp_ns: u64,         // 8 bytes
    /// Reserved for tracing context
    _pad1: [u64; 5],                    // 40 bytes

    // ═══════════════════════════════════════════════════════════════════
    // Cache Line 2: Priority and QoS (64 bytes)
    // ═══════════════════════════════════════════════════════════════════
    /// Message priority (0-255, higher = more important)
    pub priority: u8,                   // 1 byte
    /// Hop count for K2K routing
    pub hop_count: u8,                  // 1 byte
    /// Time-to-live in hops
    pub ttl: u8,                        // 1 byte
    /// QoS flags
    pub qos_flags: u8,                  // 1 byte
    /// Deadline timestamp (optional)
    pub deadline_ns: u64,               // 8 bytes
    /// Reserved
    _pad2: [u64; 6],                    // 48 bytes + 4 bytes alignment

    // ═══════════════════════════════════════════════════════════════════
    // Cache Line 3: Trace Context (64 bytes)
    // ═══════════════════════════════════════════════════════════════════
    /// OpenTelemetry trace ID (high 64 bits)
    pub trace_id_high: u64,             // 8 bytes
    /// OpenTelemetry trace ID (low 64 bits)
    pub trace_id_low: u64,              // 8 bytes
    /// OpenTelemetry span ID
    pub span_id: u64,                   // 8 bytes
    /// Trace flags
    pub trace_flags: u32,               // 4 bytes
    /// Reserved
    _pad3: [u8; 36],                    // 36 bytes
}

/// Message flags
bitflags::bitflags! {
    pub struct MessageFlags: u32 {
        /// Message requires acknowledgment
        const REQUIRES_ACK = 0b0000_0001;
        /// Message is a response
        const IS_RESPONSE = 0b0000_0010;
        /// Message is high priority
        const HIGH_PRIORITY = 0b0000_0100;
        /// Message has deadline
        const HAS_DEADLINE = 0b0000_1000;
        /// Message should be traced
        const TRACE_ENABLED = 0b0001_0000;
        /// Message is compressed
        const COMPRESSED = 0b0010_0000;
        /// Message is encrypted
        const ENCRYPTED = 0b0100_0000;
    }
}
```

---

## Message Passing

### H2K (Host-to-Kernel) Protocol

```
Host                                    Kernel
  │                                       │
  │  1. Acquire slot: head = h2k_head    │
  │  2. Write message to queue[head]     │
  │  3. Fence: atomic_thread_fence       │
  │  4. Publish: h2k_head = head + 1     │
  │ ─────────────────────────────────────▶│
  │                                       │  5. Check: if h2k_tail != h2k_head
  │                                       │  6. Read message from queue[tail]
  │                                       │  7. Fence: atomic_thread_fence
  │                                       │  8. Consume: h2k_tail = tail + 1
  │                                       │  9. Process message
  │                                       │
```

**H2K Message Types**:
```rust
#[repr(C)]
pub struct H2KMessage {
    pub header: MessageHeader,
    pub command: H2KCommand,
    pub payload: [u8; 192],  // Variable-length payload
}

#[repr(u32)]
pub enum H2KCommand {
    /// No operation
    Nop = 0,
    /// Run N simulation steps
    RunSteps { count: u64 } = 1,
    /// Pause kernel execution
    Pause = 2,
    /// Resume kernel execution
    Resume = 3,
    /// Inject impulse at location
    Inject { x: u32, y: u32, z: u32, value: f32 } = 4,
    /// Request progress update
    GetProgress = 5,
    /// Request statistics
    GetStats = 6,
    /// Terminate kernel
    Terminate = 7,
    /// Custom command (type_id in header)
    Custom = 255,
}
```

### K2H (Kernel-to-Host) Protocol

Same SPSC queue pattern, kernel writes, host reads.

**K2H Message Types**:
```rust
#[repr(C)]
pub struct K2HMessage {
    pub header: MessageHeader,
    pub response: K2HResponse,
    pub payload: [u8; 192],
}

#[repr(u32)]
pub enum K2HResponse {
    /// Acknowledgment of command
    Ack { command_id: u64 } = 0,
    /// Progress report
    Progress { step: u64, total: u64, rate: f32 } = 1,
    /// Statistics
    Stats { execution_time_ns: u64, messages_processed: u64 } = 2,
    /// Error occurred
    Error { code: u32, message: [u8; 128] } = 3,
    /// Kernel terminated
    Terminated { final_step: u64 } = 4,
    /// Energy/metric value
    Metric { name: [u8; 32], value: f64 } = 5,
    /// Custom response
    Custom = 255,
}
```

### K2K (Kernel-to-Kernel) Protocol

Direct GPU memory communication between thread blocks or kernels.

```
Kernel A (Block 0)                      Kernel B (Block 1)
       │                                       │
       │  1. Check route table for dest        │
       │  2. Acquire slot in dest inbox        │
       │  3. Write message to slot             │
       │  4. Memory fence (device scope)       │
       │  5. Publish: inbox_head++             │
       │ ─────────────────────────────────────▶│
       │                                       │  6. Poll inbox
       │                                       │  7. Read message
       │                                       │  8. Memory fence
       │                                       │  9. Consume: inbox_tail++
       │                                       │
```

**K2K Route Table**:
```rust
/// Routing entry for K2K communication
#[repr(C)]
pub struct K2KRouteEntry {
    /// Destination kernel/block ID
    pub dest_id: u32,
    /// Pointer to destination inbox
    pub inbox_ptr: u64,
    /// Inbox capacity
    pub inbox_capacity: u32,
    /// Current inbox head (for publishing)
    pub inbox_head: *mut AtomicU32,
    /// Current inbox tail (for reading)
    pub inbox_tail: *mut AtomicU32,
    /// Neighbor direction (for stencil patterns)
    pub direction: K2KDirection,
    /// Reserved
    _pad: [u32; 2],
}

#[repr(u8)]
pub enum K2KDirection {
    North = 0,
    South = 1,
    East = 2,
    West = 3,
    Up = 4,
    Down = 5,
    Custom = 255,
}
```

---

## Backend-Specific Implementation

### CUDA Implementation

```c
// Persistent kernel structure in CUDA
__global__ void persistent_kernel(
    PersistentControlBlock* __restrict__ ctrl,
    H2KMessage* __restrict__ h2k_queue,
    K2HMessage* __restrict__ k2h_queue,
    void* __restrict__ state,
    K2KRouteEntry* __restrict__ routes
) {
    // Initialize cooperative groups
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    // Main persistent loop
    while (atomicLoad(&ctrl->status) != TERMINATED) {
        // 1. Process H2K commands
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            while (h2k_has_messages(ctrl)) {
                H2KMessage* msg = h2k_dequeue(ctrl, h2k_queue);
                process_h2k_command(ctrl, msg, k2h_queue);
            }
        }

        // 2. Grid-wide synchronization
        grid.sync();

        // 3. Check if we should run steps
        if (atomicLoad(&ctrl->current_step) < atomicLoad(&ctrl->target_step)) {
            // 4. Execute one simulation step
            simulation_step(state, routes);

            // 5. K2K halo exchange
            k2k_exchange_halos(state, routes);

            // 6. Increment step counter
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                atomicAdd(&ctrl->current_step, 1);
            }

            // 7. Grid sync after step
            grid.sync();
        }
    }

    // Cleanup and send termination response
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        k2h_send_terminated(ctrl, k2h_queue);
    }
}
```

### Metal Implementation (Proposed)

```metal
// Metal persistent kernel using Indirect Command Buffer
kernel void persistent_kernel(
    device PersistentControlBlock* ctrl [[buffer(0)]],
    device H2KMessage* h2k_queue [[buffer(1)]],
    device K2HMessage* k2h_queue [[buffer(2)]],
    device void* state [[buffer(3)]],
    device K2KRouteEntry* routes [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]
) {
    // Use threadgroup_barrier for synchronization
    threadgroup_barrier(mem_flags::mem_device);

    // Process commands (leader thread only)
    if (lid == 0 && gid == 0) {
        while (h2k_has_messages(ctrl)) {
            device H2KMessage* msg = h2k_dequeue(ctrl, h2k_queue);
            process_h2k_command(ctrl, msg, k2h_queue);
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Execute simulation step
    if (atomic_load_explicit(&ctrl->current_step, memory_order_relaxed)
        < atomic_load_explicit(&ctrl->target_step, memory_order_relaxed)) {
        simulation_step(state, tid);
        k2k_exchange_halos(state, routes, tid, gid);

        if (lid == 0 && gid == 0) {
            atomic_fetch_add_explicit(&ctrl->current_step, 1, memory_order_release);
        }
    }
}
```

### WebGPU Implementation (Host-Driven)

```rust
// WebGPU: Host drives persistence via batched dispatches
pub struct WgpuPersistentEmulation {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    ctrl_buffer: wgpu::Buffer,
    h2k_staging: wgpu::Buffer,
    k2h_staging: wgpu::Buffer,
}

impl WgpuPersistentEmulation {
    /// Process batch of commands with single dispatch
    pub async fn process_batch(
        &self,
        commands: &[H2KCommand],
        steps: u64,
    ) -> Result<Vec<K2HResponse>> {
        // 1. Write commands to staging buffer
        self.queue.write_buffer(&self.h2k_staging, 0, bytemuck::cast_slice(commands));

        // 2. Create command encoder
        let mut encoder = self.device.create_command_encoder(&Default::default());

        // 3. Dispatch kernel for N steps
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);

            // Dispatch once per step (cannot persist across dispatches)
            for _ in 0..steps {
                pass.dispatch_workgroups(self.grid_x, self.grid_y, self.grid_z);
            }
        }

        // 4. Copy responses to staging
        encoder.copy_buffer_to_buffer(
            &self.k2h_buffer, 0,
            &self.k2h_staging, 0,
            self.k2h_size,
        );

        // 5. Submit and wait
        self.queue.submit(std::iter::once(encoder.finish()));

        // 6. Map and read responses
        let responses = self.read_responses().await?;
        Ok(responses)
    }
}
```

---

## HLC (Hybrid Logical Clock) Integration

### HLC Operations

```rust
/// GPU-side HLC operations
impl HlcClock {
    /// Tick: Increment clock before local event
    pub fn tick(&mut self) -> HlcTimestamp {
        let physical = self.wall_clock_ns();
        if physical > self.timestamp.physical {
            self.timestamp = HlcTimestamp {
                physical,
                logical: 0,
                node_id: self.node_id,
            };
        } else {
            self.timestamp.logical += 1;
        }
        self.timestamp
    }

    /// Update: Merge with received timestamp
    pub fn update(&mut self, received: HlcTimestamp) -> HlcTimestamp {
        let physical = self.wall_clock_ns();

        if physical > self.timestamp.physical && physical > received.physical {
            self.timestamp = HlcTimestamp {
                physical,
                logical: 0,
                node_id: self.node_id,
            };
        } else if self.timestamp.physical > received.physical {
            self.timestamp.logical += 1;
        } else if received.physical > self.timestamp.physical {
            self.timestamp = HlcTimestamp {
                physical: received.physical,
                logical: received.logical + 1,
                node_id: self.node_id,
            };
        } else {
            self.timestamp.logical = max(self.timestamp.logical, received.logical) + 1;
        }

        self.timestamp
    }
}
```

### GPU Intrinsics (CUDA)

```c
// HLC intrinsics for CUDA kernels
__device__ __forceinline__ void hlc_tick(
    PersistentControlBlock* ctrl,
    HlcTimestamp* out
) {
    unsigned long long physical = clock64();  // Or global timer

    unsigned long long prev_physical = atomicMax(&ctrl->hlc_physical, physical);
    if (physical > prev_physical) {
        atomicExch(&ctrl->hlc_logical, 0);
        out->physical = physical;
        out->logical = 0;
    } else {
        out->logical = atomicAdd(&ctrl->hlc_logical, 1);
        out->physical = prev_physical;
    }
    out->node_id = ctrl->hlc_node_id;
}

__device__ __forceinline__ void hlc_update(
    PersistentControlBlock* ctrl,
    const HlcTimestamp* received,
    HlcTimestamp* out
) {
    unsigned long long physical = clock64();
    unsigned long long local_phys = atomicLoad(&ctrl->hlc_physical);
    unsigned int local_log = atomicLoad(&ctrl->hlc_logical);

    if (physical > local_phys && physical > received->physical) {
        atomicMax(&ctrl->hlc_physical, physical);
        atomicExch(&ctrl->hlc_logical, 0);
        out->physical = physical;
        out->logical = 0;
    } else if (local_phys > received->physical) {
        out->physical = local_phys;
        out->logical = atomicAdd(&ctrl->hlc_logical, 1);
    } else if (received->physical > local_phys) {
        atomicMax(&ctrl->hlc_physical, received->physical);
        atomicExch(&ctrl->hlc_logical, received->logical + 1);
        out->physical = received->physical;
        out->logical = received->logical + 1;
    } else {
        unsigned int new_log = max(local_log, received->logical) + 1;
        atomicMax(&ctrl->hlc_logical, new_log);
        out->physical = local_phys;
        out->logical = new_log;
    }
    out->node_id = ctrl->hlc_node_id;
}
```

---

## Error Handling

### Error Codes

```rust
#[repr(u32)]
pub enum PersistentError {
    /// No error
    Ok = 0,
    /// H2K queue full
    H2KQueueFull = 1,
    /// K2H queue full
    K2HQueueFull = 2,
    /// Invalid message magic number
    InvalidMagic = 3,
    /// Message checksum mismatch
    ChecksumMismatch = 4,
    /// Unknown command type
    UnknownCommand = 5,
    /// Kernel already terminated
    AlreadyTerminated = 6,
    /// K2K routing failed
    K2KRoutingFailed = 7,
    /// Memory allocation failed
    OutOfMemory = 8,
    /// Deadline exceeded
    DeadlineExceeded = 9,
    /// Custom error (code in payload)
    Custom = 255,
}
```

### Error Recovery

```rust
/// Host-side error recovery
impl<H: PersistentHandle> ErrorRecovery for H {
    /// Attempt to recover from error state
    async fn recover(&mut self) -> Result<RecoveryAction> {
        let status = self.status()?;

        match status {
            KernelStatus::Error => {
                let error = self.last_error()?;
                match error.code {
                    PersistentError::H2KQueueFull => {
                        // Wait for kernel to drain queue
                        tokio::time::sleep(Duration::from_micros(100)).await;
                        Ok(RecoveryAction::Retry)
                    }
                    PersistentError::K2HQueueFull => {
                        // Drain responses
                        self.poll_responses().await?;
                        Ok(RecoveryAction::Retry)
                    }
                    PersistentError::ChecksumMismatch => {
                        // Resend command
                        Ok(RecoveryAction::Resend)
                    }
                    _ => {
                        // Unrecoverable, restart kernel
                        self.shutdown().await?;
                        Ok(RecoveryAction::Restart)
                    }
                }
            }
            _ => Ok(RecoveryAction::None),
        }
    }
}
```

---

## Performance Considerations

### Memory Alignment

- All structures: 64-byte aligned (cache line)
- Message headers: 256 bytes (4 cache lines)
- Control block: 256 bytes (4 cache lines)
- Queue capacity: Power of 2 for efficient indexing

### Latency Optimization

| Operation | Traditional | Persistent | Notes |
|-----------|-------------|------------|-------|
| Command injection | 300+ µs | <0.1 µs | No kernel launch |
| Response polling | N/A | <0.1 µs | Mapped memory read |
| Grid sync | N/A | 1-10 µs | Cooperative groups |
| K2K exchange | 50+ µs | 1-5 µs | Device memory only |

### Queue Sizing

```rust
/// Calculate optimal queue capacity
pub fn optimal_queue_capacity(
    expected_throughput: usize,  // messages/sec
    latency_budget_us: u64,      // max latency
) -> usize {
    let capacity = (expected_throughput as u64 * latency_budget_us / 1_000_000) + 1;
    capacity.next_power_of_two() as usize
}
```

---

## Testing Requirements

### Correctness Tests

1. **Lifecycle**: Created → Active → Paused → Active → Terminated
2. **H2K Delivery**: All commands delivered in order
3. **K2H Delivery**: All responses received
4. **K2K Routing**: Messages reach correct destinations
5. **HLC Monotonicity**: Timestamps always increase

### Stress Tests

1. **Queue Saturation**: Fill queues to capacity
2. **Rapid Pause/Resume**: 1000 toggles/second
3. **Maximum K2K**: All-to-all communication
4. **Long-Running**: 24-hour stability test

### Performance Benchmarks

1. **Command Latency**: Time from host write to kernel read
2. **Response Latency**: Time from kernel write to host read
3. **Step Throughput**: Simulation steps per second
4. **K2K Bandwidth**: Messages per second between blocks

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01 | Initial specification |

---

## References

1. CUDA Cooperative Groups Programming Guide
2. Metal Indirect Command Buffers Documentation
3. WebGPU Specification (W3C)
4. Hybrid Logical Clocks (Kulkarni et al., 2014)
