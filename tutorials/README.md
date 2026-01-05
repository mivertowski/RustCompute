# RingKernel Interactive Tutorials

Welcome to RingKernel's interactive tutorial series! These tutorials guide you
through the core concepts of GPU-native persistent actors.

## Prerequisites

- Rust 1.75 or later
- Basic understanding of async Rust
- No GPU hardware required (CPU backend used for learning)

## Tutorials

### 01: Getting Started
**File:** `01-getting-started/tutorial.rs`

Learn the fundamentals:
- Creating a runtime
- Launching kernels
- Understanding lifecycle states
- Graceful shutdown

```bash
cargo run -p ringkernel-tutorials --bin tutorial-01
```

### 02: Message Passing
**File:** `02-message-passing/tutorial.rs`

Master GPU communication:
- Defining messages
- Lock-free queues
- Hybrid Logical Clocks (HLC)
- Request-response patterns

```bash
cargo run -p ringkernel-tutorials --bin tutorial-02
```

### 03: Writing GPU Kernels
**File:** `03-gpu-kernels/tutorial.rs`

Write GPU code in Rust:
- Rust DSL syntax
- GPU intrinsics
- Global, stencil, and ring kernels
- Multi-backend compilation

```bash
cargo run -p ringkernel-tutorials --bin tutorial-03
```

### 04: Enterprise Features
**File:** `04-enterprise-features/tutorial.rs`

Production-ready features:
- Health monitoring
- Circuit breakers
- Graceful degradation
- Prometheus metrics
- GPU profiling

```bash
cargo run -p ringkernel-tutorials --bin tutorial-04
```

## Running All Tutorials

```bash
# Run all tutorials in sequence
for i in 01 02 03 04; do
    cargo run -p ringkernel-tutorials --bin tutorial-$i
    echo ""
done
```

## Learning Path

For the best learning experience, complete the tutorials in order:

```
01-Getting-Started
       │
       ▼
02-Message-Passing
       │
       ▼
03-GPU-Kernels
       │
       ▼
04-Enterprise-Features
```

Each tutorial builds on concepts from previous ones.

## Exercises

Each tutorial includes exercises at the end. Try them to reinforce your learning:

1. **Getting Started**: Launch multiple kernels, observe state transitions
2. **Message Passing**: Implement ping-pong, add correlation IDs
3. **GPU Kernels**: Write vector addition, modify stencil patterns
4. **Enterprise**: Create custom health checks, export metrics

## Additional Resources

- **Examples**: `examples/` directory for complete applications
- **API Reference**: `cargo doc --open`
- **Architecture Guide**: `CLAUDE.md` in repository root
- **Showcase Apps**: `ringkernel-wavesim`, `ringkernel-txmon`, `ringkernel-procint`

## Getting Help

- GitHub Issues: [github.com/mivertowski/RustCompute/issues](https://github.com/mivertowski/RustCompute/issues)
- Documentation: [docs.rs/ringkernel](https://docs.rs/ringkernel)
