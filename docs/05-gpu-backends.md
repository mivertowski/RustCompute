---
layout: default
title: GPU Backends
nav_order: 6
---

# GPU Backends

## Backend Trait Hierarchy

```rust
// crates/ringkernel-core/src/backend.rs

use async_trait::async_trait;

/// Core GPU backend abstraction.
#[async_trait]
pub trait GpuBackend: Send + Sync + 'static {
    /// Backend identifier.
    fn name(&self) -> &'static str;

    /// Number of available devices.
    fn device_count(&self) -> usize;

    /// Get device properties.
    fn device_info(&self, device_id: usize) -> Result<DeviceInfo>;

    /// Create a new execution context.
    async fn create_context(&self, device_id: usize) -> Result<Box<dyn GpuContext>>;
}

/// GPU execution context (one per device).
#[async_trait]
pub trait GpuContext: Send + Sync {
    /// Allocate device memory.
    async fn allocate(&self, size: usize) -> Result<DevicePtr>;

    /// Free device memory.
    async fn free(&self, ptr: DevicePtr) -> Result<()>;

    /// Copy host → device.
    async fn copy_to_device(&self, src: &[u8], dst: DevicePtr) -> Result<()>;

    /// Copy device → host.
    async fn copy_to_host(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()>;

    /// Copy device → device.
    async fn copy_device_to_device(&self, src: DevicePtr, dst: DevicePtr, size: usize) -> Result<()>;

    /// Compile kernel source.
    async fn compile(&self, source: &str, options: &CompileOptions) -> Result<Box<dyn CompiledKernel>>;

    /// Create execution stream.
    fn create_stream(&self) -> Result<Box<dyn GpuStream>>;

    /// Synchronize all pending operations.
    async fn synchronize(&self) -> Result<()>;
}

/// Compiled GPU kernel.
#[async_trait]
pub trait CompiledKernel: Send + Sync {
    /// Get kernel function by name.
    fn get_function(&self, name: &str) -> Result<Box<dyn KernelFunction>>;
}

/// Executable kernel function.
#[async_trait]
pub trait KernelFunction: Send + Sync {
    /// Launch kernel.
    async fn launch(
        &self,
        grid: Dim3,
        block: Dim3,
        shared_mem: usize,
        stream: &dyn GpuStream,
        args: &[KernelArg],
    ) -> Result<()>;
}
```

---

## CUDA Backend

### Implementation with `cudarc`

```rust
// crates/ringkernel-cuda/src/lib.rs

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;

pub struct CudaBackend {
    devices: Vec<Arc<CudaDevice>>,
}

impl CudaBackend {
    pub fn new() -> Result<Self> {
        let device_count = CudaDevice::count()? as usize;
        let devices = (0..device_count)
            .map(|i| CudaDevice::new(i))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            devices: devices.into_iter().map(Arc::new).collect(),
        })
    }
}

#[async_trait]
impl GpuBackend for CudaBackend {
    fn name(&self) -> &'static str { "cuda" }

    fn device_count(&self) -> usize { self.devices.len() }

    fn device_info(&self, device_id: usize) -> Result<DeviceInfo> {
        let dev = &self.devices[device_id];
        Ok(DeviceInfo {
            name: dev.name()?,
            compute_capability: dev.compute_capability(),
            total_memory: dev.total_memory()?,
            // ...
        })
    }

    async fn create_context(&self, device_id: usize) -> Result<Box<dyn GpuContext>> {
        Ok(Box::new(CudaContext {
            device: self.devices[device_id].clone(),
        }))
    }
}

pub struct CudaContext {
    device: Arc<CudaDevice>,
}

#[async_trait]
impl GpuContext for CudaContext {
    async fn allocate(&self, size: usize) -> Result<DevicePtr> {
        let slice: CudaSlice<u8> = self.device.alloc_zeros(size)?;
        Ok(DevicePtr::from_cuda(slice))
    }

    async fn compile(&self, source: &str, options: &CompileOptions) -> Result<Box<dyn CompiledKernel>> {
        let ptx = Ptx::from_src(source)?;
        let module = self.device.load_ptx(ptx, "ring_kernel", &[])?;
        Ok(Box::new(CudaCompiledKernel { module }))
    }

    // ... other methods
}
```

### CUDA PTX Compilation (v0.3.0)

The `ringkernel-cuda` crate provides a `compile_ptx()` function for runtime CUDA compilation without directly depending on cudarc:

```rust
use ringkernel_cuda::compile_ptx;

let cuda_source = r#"
    extern "C" __global__ void add(float* out, const float* a, const float* b, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = a[idx] + b[idx];
    }
"#;

// Compile to PTX using NVRTC
let ptx = compile_ptx(cuda_source)?;
println!("Generated PTX:\n{}", ptx);
```

### cudarc 0.18.2 API (Updated)

The CUDA backend uses cudarc 0.18.2 with these patterns:

```rust
// Module loading (NEW in 0.18.2)
let module = device.inner().load_module(ptx)?;  // Returns Arc<CudaModule>
let func = module.load_function("kernel_name")?; // Load specific function

// Kernel launch with builder pattern (NEW in 0.18.2)
use cudarc::driver::PushKernelArg;
unsafe {
    stream
        .launch_builder(&func)
        .arg(&input_ptr)
        .arg(&output_ptr)
        .arg(&scalar_param)
        .launch(cfg)?;
}

// Cooperative kernel launch
use cudarc::driver::result as cuda_result;
unsafe {
    cuda_result::launch_cooperative_kernel(
        func, grid_dim, block_dim, shared_mem_bytes, stream, kernel_params
    )?;
}
```

**Old API (cudarc 0.11) - No longer works:**
- `device.load_ptx(ptx, module_name, &[func_names])` → Use `load_module()` + `load_function()`
- `device.get_func(module_name, fn_name)` → Store functions at construction time
- `func.launch(cfg, params)` → Use `stream.launch_builder(&func).arg(...).launch(cfg)`
```

### CUDA Ring Kernel Template

```cuda
// Generated CUDA C for ring kernels
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/atomic>

namespace cg = cooperative_groups;

struct ControlBlock {
    cuda::atomic<int> is_active;
    cuda::atomic<int> should_terminate;
    cuda::atomic<int> has_terminated;
    cuda::atomic<int> errors_encountered;
    cuda::atomic<long long> messages_processed;
    // ... queue pointers
};

template<typename TInput, typename TOutput>
__global__ void ring_kernel_persistent(
    ControlBlock* control,
    TInput* input_buffer,
    cuda::atomic<int>* input_head,
    cuda::atomic<int>* input_tail,
    TOutput* output_buffer,
    cuda::atomic<int>* output_head,
    cuda::atomic<int>* output_tail,
    int queue_capacity
) {
    auto grid = cg::this_grid();

    // Persistent kernel loop
    while (!control->should_terminate.load(cuda::memory_order_acquire)) {
        // Check if active
        if (!control->is_active.load(cuda::memory_order_acquire)) {
            // Yield to avoid busy-waiting
            __nanosleep(1000);
            continue;
        }

        // Try dequeue input
        int head = input_head->load(cuda::memory_order_relaxed);
        int tail = input_tail->load(cuda::memory_order_acquire);

        if (head != tail) {
            // Claim slot
            int next_head = (head + 1) & (queue_capacity - 1);
            if (input_head->compare_exchange_strong(head, next_head,
                    cuda::memory_order_acq_rel)) {

                TInput msg = input_buffer[head];

                // === USER HANDLER CODE ===
                TOutput response = process_message(msg);
                // =========================

                // Enqueue output
                enqueue_output(output_buffer, output_head, output_tail,
                              queue_capacity, response);

                control->messages_processed.fetch_add(1, cuda::memory_order_relaxed);
            }
        }

        grid.sync(); // Cooperative sync
    }

    // Mark terminated
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        control->has_terminated.store(1, cuda::memory_order_release);
    }
}
```

---

## Persistent vs Traditional Kernel Patterns

RingKernel supports two fundamentally different GPU execution models, each optimized for different workloads.

### Traditional Kernels (Launch-per-Command)

```
Host                              GPU
  │                                │
  ├─── cudaMemcpy(H→D) ───────────>│ (20-50µs)
  ├─── cudaLaunchKernel ──────────>│ (10-30µs)
  │                                ├── Compute
  │<── cudaDeviceSynchronize ──────┤ (5-20µs)
  │<── cudaMemcpy(D→H) ────────────┤ (20-50µs)
  │                                │
```

**Best for:** Batch processing, compute-bound workloads, running 1000s of steps at once.

### Persistent Kernels (Actor Model)

```
Host                              GPU
  │                                │
  ├─── Launch kernel (once) ──────>│ ← Single launch
  │                                ├── Persistent loop
  │    ┌─────────────────────────┐ │
  │    │ for each command:       │ │
  ├────┤  Write to mapped mem ───┼─┤ (~10-50ns)
  │<───┤  Poll response ─────────┼─┤ (~1-5µs)
  │    └─────────────────────────┘ │
  │                                │
  ├─── Terminate signal ──────────>│
  │                                │
```

**Best for:** Interactive applications, real-time GUIs, dynamic parameter changes, command-heavy workloads.

### Performance Comparison (RTX Ada)

| Operation | Traditional | Persistent | Speedup |
|-----------|-------------|------------|---------|
| Inject command | 317 µs | 0.03 µs | **11,327x** |
| Single compute step | 3.2 µs | 163 µs | Traditional 51x |
| Mixed workload (300 ops) | 40.5 ms | 15.3 ms | **Persistent 2.7x** |

### Decision Guide

| Your Workload | Recommended | Why |
|---------------|-------------|-----|
| Run 10,000 simulation steps | Traditional | Launch overhead amortized |
| Real-time GUI at 60 FPS | **Persistent** | 2.7x more ops per frame |
| Interactive parameter tuning | **Persistent** | 0.03µs command latency |
| Batch matrix operations | Traditional | Maximum compute throughput |
| Game physics with user input | **Persistent** | Instant response to inputs |

---

## Backend Selection

```rust
/// Automatic backend selection based on platform.
pub fn select_backend() -> Result<Box<dyn GpuBackend>> {
    #[cfg(feature = "cuda")]
    if let Ok(backend) = CudaBackend::new() {
        if backend.device_count() > 0 {
            return Ok(Box::new(backend));
        }
    }

    #[cfg(feature = "cpu")]
    return Ok(Box::new(CpuBackend::new()));

    Err(Error::NoBackendAvailable)
}
```

---

## Rust-to-CUDA Transpilation

The `ringkernel-cuda-codegen` crate enables writing CUDA kernels in a Rust DSL:

```rust
use ringkernel_cuda_codegen::{transpile_global_kernel, transpile_stencil_kernel, StencilConfig};
use ringkernel_cuda_codegen::dsl::*;
use syn::parse_quote;

// Generic kernel using block/thread indices
let kernel: syn::ItemFn = parse_quote! {
    fn vector_scale(data: &mut [f32], scale: f32, n: i32) {
        let idx = block_idx_x() * block_dim_x() + thread_idx_x();
        if idx >= n { return; }
        data[idx as usize] = data[idx as usize] * scale;
    }
};
let cuda = transpile_global_kernel(&kernel)?;

// Stencil kernel using GridPos abstraction
let stencil: syn::ItemFn = parse_quote! {
    fn laplacian(input: &[f32], output: &mut [f32], pos: GridPos) {
        let lap = pos.north(input) + pos.south(input)
                + pos.east(input) + pos.west(input)
                - 4.0 * input[pos.idx()];
        output[pos.idx()] = lap;
    }
};
let config = StencilConfig::new("laplacian").with_tile_size(16, 16).with_halo(1);
let cuda_stencil = transpile_stencil_kernel(&stencil, &config)?;
```

### DSL Functions (CPU Fallback Implementations)

The `dsl` module provides functions that compile to CUDA intrinsics but also work on CPU:

```rust
use ringkernel_cuda_codegen::dsl::*;

// Thread/block indices (return 0 on CPU, compile to CUDA intrinsics)
let tx = thread_idx_x();  // -> threadIdx.x
let bx = block_idx_x();   // -> blockIdx.x
let bd = block_dim_x();   // -> blockDim.x
let gd = grid_dim_x();    // -> gridDim.x

// Synchronization (no-op on CPU)
sync_threads();           // -> __syncthreads()
thread_fence();           // -> __threadfence()
```

---

## Next: [Serialization Strategy](./06-serialization.md)
