# RingKernel VSCode Extension

GPU kernel development support for RingKernel - syntax highlighting, snippets, and GPU debugging.

## Features

### Code Snippets

The extension provides snippets for common RingKernel patterns:

| Prefix | Description |
|--------|-------------|
| `ringkernel` | Create a persistent ring kernel actor |
| `gpukernel` | Create a GPU global kernel |
| `stencilkernel` | Create a 2D stencil kernel |
| `stencilkernel3d` | Create a 3D stencil kernel |
| `ringmessage` | Create a RingMessage struct |
| `gputype` | Create a GPU-compatible type |
| `tidx`, `tidx2d`, `tidx3d` | Thread index calculations |
| `sync` | Thread synchronization |
| `atomicadd`, `atomiccas` | Atomic operations |
| `k2ksend`, `k2krecv` | Kernel-to-kernel messaging |
| `sandboxpolicy` | Create kernel sandbox |
| `memencrypt` | Setup memory encryption |
| `compliance` | Generate compliance report |

### Commands

Access commands via the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`):

- **RingKernel: Generate GPU Kernel** - Create a new kernel from template
- **RingKernel: Transpile to CUDA** - Convert Rust DSL to CUDA
- **RingKernel: Transpile to WGSL** - Convert Rust DSL to WGSL
- **RingKernel: Check Backend Compatibility** - Verify kernel backend support
- **RingKernel: Launch GPU Playground** - Open interactive kernel playground
- **RingKernel: Show GPU Memory Dashboard** - View GPU memory usage
- **RingKernel: Profile GPU Kernel** - Run kernel profiler

### Sidebar Views

The extension adds a RingKernel activity bar with:

- **GPU Kernels** - List of kernels in your project
- **Memory Usage** - Real-time GPU memory monitoring
- **Profiler** - Kernel performance metrics

### Code Lens

Kernel functions display inline actions:
- **Run Kernel** - Execute the kernel
- **Transpile** - Convert to target backend

### Hover Information

Hover over GPU intrinsics to see documentation:

```rust
let idx = block_idx_x() * block_dim_x() + thread_idx_x();
//        ^-- Hover for documentation
```

## Configuration

```json
{
  "ringkernel.defaultBackend": "cuda",
  "ringkernel.enableInlayHints": true,
  "ringkernel.showMemoryUsage": true,
  "ringkernel.cliPath": "/path/to/ringkernel-cli",
  "ringkernel.autoTranspile": false,
  "ringkernel.playground.port": 8765
}
```

## Requirements

- [ringkernel-cli](https://crates.io/crates/ringkernel-cli) for transpilation
- NVIDIA GPU + CUDA toolkit (optional, for CUDA backend)
- WebGPU support (optional, for WebGPU backend)

## Installation

### From Marketplace

Search for "RingKernel" in the VSCode Extensions view.

### From Source

```bash
cd tools/vscode-ringkernel
npm install
npm run compile
```

Then press F5 to launch a development instance.

## Development

```bash
# Install dependencies
npm install

# Compile TypeScript
npm run compile

# Watch for changes
npm run watch

# Run linter
npm run lint

# Run tests
npm test
```

## License

MIT License - see [LICENSE](../../LICENSE) for details.
