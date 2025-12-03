# ringkernel-core

Core traits and types for the RingKernel GPU-native actor system.

## Overview

This crate provides the foundational abstractions that all RingKernel backends implement. It defines the message passing protocols, kernel lifecycle management, and synchronization primitives.

## Core Abstractions

### Messages

```rust
use ringkernel_core::prelude::*;

// Messages must be serializable for GPU transfer
pub trait RingMessage: Archive + Serialize + Deserialize {
    fn type_id(&self) -> u32;
    fn correlation_id(&self) -> Option<CorrelationId>;
}
```

### Kernel Runtime

```rust
#[async_trait]
pub trait RingKernelRuntime: Send + Sync {
    async fn launch(&self, id: &str, options: LaunchOptions) -> Result<KernelHandle>;
    async fn shutdown(&self) -> Result<()>;
    fn is_k2k_enabled(&self) -> bool;
}
```

### Control Block

128-byte GPU-resident structure for kernel lifecycle management:

```rust
#[repr(C, align(128))]
pub struct ControlBlock {
    pub is_active: AtomicU32,
    pub should_terminate: AtomicU32,
    pub has_terminated: AtomicU32,
    pub messages_processed: AtomicU64,
    // ... HLC state, queue pointers, etc.
}
```

## Key Components

| Component | Description |
|-----------|-------------|
| `RingMessage` | Trait for GPU-transferable messages |
| `MessageQueue` | Lock-free ring buffer for message passing |
| `KernelHandle` | Handle to manage kernel lifecycle |
| `ControlBlock` | GPU-resident kernel state |
| `HlcTimestamp` | Hybrid Logical Clock for causal ordering |
| `K2KBroker` | Kernel-to-kernel messaging broker |
| `PubSubBroker` | Topic-based publish/subscribe |

## Hybrid Logical Clocks

Provides causal ordering across distributed operations:

```rust
use ringkernel_core::hlc::{HlcClock, HlcTimestamp};

let clock = HlcClock::new(node_id);
let ts1 = clock.tick();
let ts2 = clock.tick();
assert!(ts1 < ts2);

// Synchronize with remote timestamp
let synced = clock.update(&remote_ts)?;
```

## Kernel-to-Kernel Messaging

Direct communication between kernels without host involvement:

```rust
let broker = K2KBroker::new();
let endpoint = broker.register(kernel_id);
endpoint.send(destination_id, envelope).await?;
```

## Testing

```bash
cargo test -p ringkernel-core
```

The crate includes 65+ tests covering message serialization, queue operations, HLC ordering, and K2K messaging.

## License

Apache-2.0
