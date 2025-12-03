# ringkernel-cpu

CPU backend implementation for RingKernel.

## Overview

This crate provides a CPU-based implementation of the `RingKernelRuntime` trait. It serves two purposes:

1. **Development and Testing**: Test kernel logic without GPU hardware
2. **Fallback**: Run on systems without GPU support

The CPU backend implements the full RingKernel API including kernel lifecycle, message queues, HLC timestamps, and K2K messaging.

## Usage

```rust
use ringkernel_cpu::CpuRuntime;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = CpuRuntime::new().await?;

    let kernel = runtime.launch("worker", LaunchOptions::default()).await?;

    // Process messages...

    kernel.terminate().await?;
    runtime.shutdown().await?;
    Ok(())
}
```

## Features

- Full `RingKernelRuntime` implementation
- Lock-free message queues using crossbeam
- HLC timestamp generation
- K2K messaging support
- Kernel lifecycle management (Created, Active, Paused, Terminated)

## Performance

The CPU backend is optimized for correctness over performance. For production workloads requiring high throughput, use the CUDA or WebGPU backends.

Typical performance characteristics:
- Message throughput: ~10M messages/sec
- Latency: <1ms per message batch

## Testing

```bash
cargo test -p ringkernel-cpu
```

## When to Use

- Unit testing kernel logic
- Development without GPU hardware
- CI/CD pipelines
- Systems without GPU support
- Debugging message flow

## License

Apache-2.0
