---
layout: default
title: CLI Tool
nav_order: 17
---

# RingKernel CLI Tool

The `ringkernel` CLI provides project scaffolding, kernel code generation, and compatibility checking.

## Installation

```bash
# From crates.io
cargo install ringkernel-cli

# From source
cargo install --path crates/ringkernel-cli
```

---

## Commands

### `ringkernel new` - Create Project

Create a new RingKernel project:

```bash
# Basic project
ringkernel new my-app

# With template
ringkernel new my-app --template persistent-actor

# With multiple backends
ringkernel new my-app --backends cuda,wgpu

# Custom location
ringkernel new my-app --path /projects
```

#### Templates

| Template | Description |
|----------|-------------|
| `basic` | Minimal project with single kernel (default) |
| `persistent-actor` | Persistent GPU actor with H2K/K2H messaging |
| `wavesim` | Wave simulation with stencil kernels |
| `enterprise` | Production-ready with enterprise features |

#### Generated Structure

```
my-app/
├── Cargo.toml
├── ringkernel.toml          # RingKernel configuration
├── src/
│   ├── main.rs
│   └── kernels/
│       └── mod.rs           # Kernel definitions
└── generated/               # Generated GPU code
    ├── cuda/
    └── wgsl/
```

---

### `ringkernel init` - Initialize Existing Project

Add RingKernel to an existing Rust project:

```bash
ringkernel init

# With backends
ringkernel init --backends cuda,wgpu

# Force overwrite
ringkernel init --force
```

This adds:
- RingKernel dependencies to `Cargo.toml`
- `ringkernel.toml` configuration
- Example kernel in `src/kernels/`

---

### `ringkernel codegen` - Generate GPU Code

Generate GPU kernel code from Rust DSL:

```bash
# Generate CUDA
ringkernel codegen src/kernels/processor.rs --backend cuda

# Multiple backends
ringkernel codegen src/kernels/processor.rs --backend cuda,wgsl,msl

# Custom output
ringkernel codegen src/kernels/ --backend cuda --output generated/

# Specific kernel
ringkernel codegen src/kernels/mod.rs --kernel process_messages

# Dry run (preview)
ringkernel codegen src/kernels/processor.rs --dry-run
```

#### Output

```
Generating GPU code...

  src/kernels/processor.rs:
    ✓ process_messages → generated/cuda/process_messages.cu
    ✓ process_messages → generated/wgsl/process_messages.wgsl

Generated 2 files in 0.15s
```

---

### `ringkernel check` - Validate Compatibility

Check kernel compatibility across backends:

```bash
# Check all in src/
ringkernel check

# Custom path
ringkernel check --path src/kernels

# Specific backends
ringkernel check --backends cuda,wgsl

# All backends
ringkernel check --backends all

# Detailed report
ringkernel check --detailed
```

#### Output

```
Checking kernel compatibility...

src/kernels/processor.rs:
  process_messages:
    ✓ CUDA    - Compatible
    ✓ WGSL    - Compatible (64-bit atomics emulated)
    ✗ MSL     - Incompatible (K2K not supported)

src/kernels/stencil.rs:
  fdtd_step:
    ✓ CUDA    - Compatible
    ✓ WGSL    - Compatible
    ✓ MSL     - Compatible

Summary: 3/4 kernels fully compatible across all backends
```

---

### `ringkernel completions` - Shell Completions

Generate shell completion scripts:

```bash
# Bash
ringkernel completions bash > ~/.bash_completion.d/ringkernel

# Zsh
ringkernel completions zsh > ~/.zfunc/_ringkernel

# Fish
ringkernel completions fish > ~/.config/fish/completions/ringkernel.fish

# PowerShell
ringkernel completions powershell > ringkernel.ps1
```

---

## Configuration

### `ringkernel.toml`

```toml
[project]
name = "my-gpu-app"
version = "0.1.0"

[backends]
default = "cuda"
enabled = ["cuda", "wgpu"]

[codegen]
output_dir = "generated"
optimize = true

[codegen.cuda]
arch = "sm_80"
ptx_version = "7.0"

[codegen.wgsl]
workgroup_size = [256, 1, 1]

[check]
strict = true
warn_emulation = true
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `RINGKERNEL_BACKEND` | Default backend |
| `RINGKERNEL_CUDA_ARCH` | CUDA architecture |
| `RINGKERNEL_OUTPUT_DIR` | Code generation output |

---

## Global Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable verbose output |
| `-q, --quiet` | Suppress all output except errors |
| `--version` | Show version |
| `--help` | Show help |

---

## CI/CD Integration

```yaml
# .github/workflows/check.yml
name: RingKernel Check
on: [push, pull_request]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo install ringkernel-cli
      - run: ringkernel check --backends all --detailed
```

---

## Next: [Security Module](./17-security.md)
