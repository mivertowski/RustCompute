# ringkernel-cli

Command-line tool for RingKernel project scaffolding, kernel code generation, and compatibility checking.

## Installation

### From crates.io

```bash
cargo install ringkernel-cli
```

### From source

```bash
git clone https://github.com/mivertowski/RustCompute.git
cd RustCompute
cargo install --path crates/ringkernel-cli
```

## Commands

### `ringkernel new` - Create a New Project

Create a new RingKernel project with pre-configured templates:

```bash
# Basic project
ringkernel new my-app

# With specific template
ringkernel new my-app --template persistent-actor

# Specify GPU backends
ringkernel new my-app --backends cuda,wgpu

# Custom path
ringkernel new my-app --path /path/to/projects
```

**Templates:**

| Template | Description |
|----------|-------------|
| `basic` | Minimal project with single kernel (default) |
| `persistent-actor` | Persistent GPU actor with H2K/K2H messaging |
| `wavesim` | Wave simulation with stencil kernels |
| `enterprise` | Production-ready with enterprise features |

**Options:**
- `-t, --template <TEMPLATE>` - Project template (default: `basic`)
- `-p, --path <PATH>` - Target directory
- `-b, --backends <BACKENDS>` - GPU backends (default: `cuda`)
- `--no-git` - Skip git repository initialization

### `ringkernel init` - Initialize in Existing Project

Add RingKernel to an existing Rust project:

```bash
ringkernel init

# With specific backends
ringkernel init --backends cuda,wgpu

# Force overwrite existing config
ringkernel init --force
```

This adds:
- RingKernel dependencies to `Cargo.toml`
- `ringkernel.toml` configuration file
- Example kernel in `src/kernels/`

### `ringkernel codegen` - Generate GPU Code

Generate GPU kernel code from Rust DSL:

```bash
# Generate CUDA code
ringkernel codegen src/kernels/processor.rs --backend cuda

# Generate multiple backends
ringkernel codegen src/kernels/processor.rs --backend cuda,wgsl,msl

# Custom output directory
ringkernel codegen src/kernels/ --backend cuda --output generated/

# Generate specific kernel
ringkernel codegen src/kernels/mod.rs --kernel process_messages

# Preview without writing files
ringkernel codegen src/kernels/processor.rs --dry-run
```

**Options:**
- `-b, --backend <BACKEND>` - Target backends: `cuda`, `wgsl`, `msl` (default: `cuda`)
- `-o, --output <DIR>` - Output directory for generated code
- `-k, --kernel <NAME>` - Generate specific kernel only
- `--dry-run` - Preview generated code without writing files

### `ringkernel check` - Validate Compatibility

Check kernel compatibility across GPU backends:

```bash
# Check all kernels in src/
ringkernel check

# Check specific directory
ringkernel check --path src/kernels

# Check against specific backends
ringkernel check --backends cuda,wgsl

# Check all backends
ringkernel check --backends all

# Detailed report
ringkernel check --detailed
```

**Options:**
- `-p, --path <PATH>` - Directory to scan (default: `src`)
- `-b, --backends <BACKENDS>` - Backends to check (default: `all`)
- `--detailed` - Show detailed compatibility report

Example output:
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

### `ringkernel profile` (Planned)

Profile kernel performance:

```bash
ringkernel profile my_kernel --iterations 1000 --format flamegraph
```

*Note: This command is planned for a future release.*

## Configuration

Create `ringkernel.toml` in your project root:

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
```

## Global Options

- `-v, --verbose` - Enable verbose output
- `-q, --quiet` - Suppress all output except errors
- `--version` - Show version information
- `--help` - Show help

## Examples

### Create and Build a Project

```bash
# Create new project with persistent actor template
ringkernel new my-gpu-app --template persistent-actor

# Navigate to project
cd my-gpu-app

# Generate CUDA code
ringkernel codegen src/kernels/actor.rs --backend cuda

# Build and run
cargo run --release --features cuda
```

### CI/CD Integration

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

## License

Licensed under Apache-2.0 OR MIT.
