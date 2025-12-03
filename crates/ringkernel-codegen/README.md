# ringkernel-codegen

Template-based GPU kernel code generation for RingKernel.

## Overview

This crate generates GPU kernel source code from templates for multiple backend targets: CUDA PTX, Metal MSL, and WebGPU WGSL. It provides a simple template substitution system for embedding user kernel code.

For DSL-to-GPU transpilation, see `ringkernel-cuda-codegen` and `ringkernel-wgpu-codegen`.

## Usage

```rust
use ringkernel_codegen::{CodeGenerator, Target, KernelConfig};

let generator = CodeGenerator::new();

// Generate for a specific target
let cuda_source = generator.generate_kernel_source(
    "my_kernel",
    "// custom kernel code here",
    Target::Cuda,
)?;

// Or generate for all targets at once
let config = KernelConfig {
    id: "processor".to_string(),
    grid_size: 4,
    block_size: 256,
    shared_memory: 1024,
    ..Default::default()
};

let files = generator.generate_all_targets(&config, "// user code")?;
for file in files {
    println!("{}: {} bytes", file.filename, file.content.len());
}
```

## Targets

| Target | Extension | Description |
|--------|-----------|-------------|
| `Target::Cuda` | `.ptx` | NVIDIA CUDA (PTX assembly) |
| `Target::Metal` | `.metal` | Apple Metal (MSL) |
| `Target::Wgsl` | `.wgsl` | WebGPU (WGSL) |

## Template Variables

Custom variables can be set for template substitution:

```rust
let mut generator = CodeGenerator::new();
generator.set_variable("BLOCK_SIZE", "256");
generator.set_variable("QUEUE_CAPACITY", "1024");
```

Variables are substituted using `{{VARIABLE_NAME}}` syntax in templates.

## Intrinsic Mappings

The crate provides cross-platform intrinsic mappings:

```rust
use ringkernel_codegen::standard_intrinsics;

let intrinsics = standard_intrinsics();
for intrinsic in intrinsics {
    println!("{} -> CUDA: {}, Metal: {}, WGSL: {}",
        intrinsic.rust_name,
        intrinsic.cuda,
        intrinsic.metal,
        intrinsic.wgsl);
}
```

Standard mappings include:
- `sync_threads` - Thread synchronization barrier
- `thread_fence` / `thread_fence_block` - Memory fences
- `atomic_add` / `atomic_cas` - Atomic operations

## Testing

```bash
cargo test -p ringkernel-codegen
```

## License

Apache-2.0
