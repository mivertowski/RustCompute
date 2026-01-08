# ringkernel-ir

Unified Intermediate Representation for RingKernel GPU code generation.

## Overview

`ringkernel-ir` provides a unified IR that serves as the foundation for multi-backend GPU code generation. The IR is SSA-based and captures GPU-specific operations, enabling optimization passes before lowering to target backends (CUDA, WGSL, MSL).

## Architecture

```
Rust DSL (syn::ItemFn)
        │
        ▼
   ┌─────────┐
   │ IrBuilder │  ← Construct IR from Rust AST
   └────┬────┘
        │
        ▼
   ┌─────────┐
   │ IrModule │  ← SSA-based representation
   └────┬────┘
        │
        ▼
   ┌─────────────┐
   │ PassManager │  ← Optimization passes
   └────┬────────┘
        │
        ├──────────────┬──────────────┐
        ▼              ▼              ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │  CUDA   │    │  WGSL   │    │  MSL    │
   │ Lowering│    │ Lowering│    │ Lowering│
   └─────────┘    └─────────┘    └─────────┘
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ringkernel-ir = "0.2"
```

## Usage

### Building IR

```rust
use ringkernel_ir::{IrBuilder, IrType, Dimension};

let mut builder = IrBuilder::new("saxpy");

// Define parameters
let x = builder.parameter("x", IrType::Ptr(Box::new(IrType::F32)));
let y = builder.parameter("y", IrType::Ptr(Box::new(IrType::F32)));
let a = builder.parameter("a", IrType::F32);
let n = builder.parameter("n", IrType::I32);

// Get thread index
let idx = builder.thread_id(Dimension::X);

// Bounds check
let in_bounds = builder.lt(idx, n);
builder.if_then(in_bounds, |b| {
    let x_val = b.load(x, idx);
    let y_val = b.load(y, idx);
    let result = b.add(b.mul(a, x_val), y_val);
    b.store(y, idx, result);
});

let module = builder.build();
```

### Optimization Passes

```rust
use ringkernel_ir::{optimize, PassManager, ConstantFolding, DeadCodeElimination};

// Apply default optimizations
let optimized = optimize(&module);

// Or use PassManager for fine-grained control
let mut pm = PassManager::new();
pm.add_pass(ConstantFolding::new());
pm.add_pass(DeadCodeElimination::new());
let result = pm.run(&module);
```

Available passes:
- **ConstantFolding** - Evaluate compile-time constant expressions
- **DeadCodeElimination** - Remove unused values
- **DeadBlockElimination** - Remove unreachable basic blocks
- **AlgebraicSimplification** - Simplify arithmetic expressions (x * 1 → x, x + 0 → x)

### Lowering to Backends

```rust
use ringkernel_ir::{lower_to_cuda, lower_to_wgsl, lower_to_msl};

// Lower to CUDA C
let cuda_code = lower_to_cuda(&module)?;

// Lower to WGSL
let wgsl_code = lower_to_wgsl(&module)?;

// Lower to Metal Shading Language
let msl_code = lower_to_msl(&module)?;
```

### Backend Capabilities

Check what features each backend supports:

```rust
use ringkernel_ir::{BackendCapabilities, Capabilities, CapabilityFlag};

let caps = BackendCapabilities::cuda();

if caps.has(CapabilityFlag::AtomicFloat64) {
    // Use 64-bit atomic operations
} else {
    // Fall back to emulation
}
```

### Validation

```rust
use ringkernel_ir::{Validator, ValidationLevel};

let validator = Validator::new(ValidationLevel::Strict);
let result = validator.validate(&module);

if !result.is_valid() {
    for error in result.errors() {
        eprintln!("IR Error: {}", error);
    }
}
```

### Pretty Printing

```rust
use ringkernel_ir::IrPrinter;

let ir_text = IrPrinter::new().print(&module);
println!("{}", ir_text);
```

Output:
```
define kernel saxpy(x: ptr<f32>, y: ptr<f32>, a: f32, n: i32) {
bb0:
    %0 = thread_id.x
    %1 = lt %0, n
    br_if %1, bb1, bb2
bb1:
    %2 = load x[%0]
    %3 = load y[%0]
    %4 = mul a, %2
    %5 = add %4, %3
    store y[%0], %5
    br bb2
bb2:
    return
}
```

## IR Node Types

### Values
- **Constant** - Literal values (integers, floats, booleans)
- **Parameter** - Function parameters
- **BinaryOp** - Arithmetic and comparison operations
- **UnaryOp** - Negation, bitwise complement
- **Load/Store** - Memory access operations
- **Cast** - Type conversions

### GPU-Specific
- **ThreadId** - Thread index (X, Y, Z dimensions)
- **BlockId** - Block index
- **BlockDim** - Block size
- **GridDim** - Grid size
- **SyncThreads** - Block-level barrier
- **Atomic** - Atomic memory operations
- **SharedMemory** - Shared memory allocation

### Control Flow
- **Branch** - Unconditional branch
- **BranchIf** - Conditional branch
- **Return** - Function return

## Features

Enable optional features in `Cargo.toml`:

```toml
[dependencies]
ringkernel-ir = { version = "0.2", features = ["validation", "optimization"] }
```

- **validation** - Enable detailed IR validation
- **optimization** - Enable IR optimization passes

## License

Licensed under Apache-2.0 OR MIT.
