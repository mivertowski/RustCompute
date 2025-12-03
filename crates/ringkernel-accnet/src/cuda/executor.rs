//! GPU kernel executor using ringkernel-cuda.
//!
//! This module provides actual GPU execution of the generated CUDA kernels
//! using the ringkernel-cuda infrastructure.

use std::sync::Arc;
use std::time::Instant;

use ringkernel_cuda::{CudaDevice, StencilKernelLoader};
use cudarc::driver::LaunchAsync;

use crate::models::AccountingNetwork;

/// GPU-accelerated analysis executor.
pub struct GpuExecutor {
    /// CUDA device.
    device: CudaDevice,
    /// Kernel loader for compiling CUDA code.
    loader: StencilKernelLoader,
    /// Compiled suspense detection kernel.
    suspense_kernel: Option<CompiledKernel>,
    /// Compiled GAAP violation kernel.
    gaap_kernel: Option<CompiledKernel>,
    /// Compiled Benford analysis kernel.
    benford_kernel: Option<CompiledKernel>,
    /// Device name for reporting.
    device_name: String,
    /// Compute capability.
    compute_capability: (u32, u32),
}

/// A compiled CUDA kernel ready for execution.
struct CompiledKernel {
    /// Module name for function lookup.
    module_name: &'static str,
    /// Kernel function name.
    function_name: &'static str,
}

/// Result of GPU analysis.
#[derive(Debug, Clone, Default)]
pub struct GpuAnalysisResult {
    /// Suspense scores for each account.
    pub suspense_scores: Vec<f32>,
    /// GAAP violation flags for each flow.
    pub gaap_violations: Vec<u8>,
    /// Benford digit counts (1-9).
    pub benford_counts: [u32; 9],
    /// Total GPU execution time in microseconds.
    pub execution_time_us: u64,
    /// Number of accounts processed.
    pub accounts_processed: usize,
    /// Number of flows processed.
    pub flows_processed: usize,
}

/// Benchmark results for a single kernel.
#[derive(Debug, Clone)]
pub struct KernelBenchmark {
    /// Kernel name.
    pub name: String,
    /// Execution time in microseconds.
    pub time_us: u64,
    /// Elements processed.
    pub elements: usize,
    /// Throughput in million elements per second.
    pub throughput_meps: f64,
}

/// Combined benchmark results.
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Device name.
    pub device_name: String,
    /// Compute capability.
    pub compute_capability: (u32, u32),
    /// Individual kernel benchmarks.
    pub kernels: Vec<KernelBenchmark>,
    /// Total GPU time.
    pub total_gpu_time_us: u64,
    /// Total CPU baseline time.
    pub total_cpu_time_us: u64,
    /// Overall speedup factor.
    pub speedup: f64,
}

impl GpuExecutor {
    /// Create a new GPU executor.
    pub fn new() -> Result<Self, String> {
        // Check if CUDA is available
        if !ringkernel_cuda::is_cuda_available() {
            return Err("CUDA is not available on this system".to_string());
        }

        // Create device
        let device = CudaDevice::new(0)
            .map_err(|e| format!("Failed to create CUDA device: {}", e))?;

        let device_name = device.name().to_string();
        let compute_capability = device.compute_capability();

        let loader = StencilKernelLoader::new(device.clone());

        Ok(Self {
            device,
            loader,
            suspense_kernel: None,
            gaap_kernel: None,
            benford_kernel: None,
            device_name,
            compute_capability,
        })
    }

    /// Get device name.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Get compute capability.
    pub fn compute_capability(&self) -> (u32, u32) {
        self.compute_capability
    }

    /// Compile all analysis kernels.
    pub fn compile_kernels(&mut self) -> Result<(), String> {
        // Generate kernel code
        let kernels = super::codegen::GeneratedKernels::generate()?;

        // Compile suspense detection kernel
        self.compile_kernel("suspense_detection", &kernels.suspense_detection)?;
        self.suspense_kernel = Some(CompiledKernel {
            module_name: Box::leak("suspense_detection_module".to_string().into_boxed_str()),
            function_name: Box::leak("suspense_detection".to_string().into_boxed_str()),
        });

        // Compile GAAP violation kernel
        self.compile_kernel("gaap_violation", &kernels.gaap_violation)?;
        self.gaap_kernel = Some(CompiledKernel {
            module_name: Box::leak("gaap_violation_module".to_string().into_boxed_str()),
            function_name: Box::leak("gaap_violation".to_string().into_boxed_str()),
        });

        // Compile Benford analysis kernel
        self.compile_kernel("benford_analysis", &kernels.benford_analysis)?;
        self.benford_kernel = Some(CompiledKernel {
            module_name: Box::leak("benford_analysis_module".to_string().into_boxed_str()),
            function_name: Box::leak("benford_analysis".to_string().into_boxed_str()),
        });

        Ok(())
    }

    /// Compile a single kernel from CUDA source.
    fn compile_kernel(&self, name: &str, cuda_source: &str) -> Result<(), String> {
        let cuda_device = self.device.inner();

        // Compile CUDA source to PTX using NVRTC
        let ptx = cudarc::nvrtc::compile_ptx(cuda_source)
            .map_err(|e| format!("NVRTC compilation failed for '{}': {}", name, e))?;

        // Create static strings for module registration
        let module_name: &'static str = Box::leak(format!("{}_module", name).into_boxed_str());
        let func_name: &'static str = Box::leak(name.to_string().into_boxed_str());

        // Load the PTX module
        cuda_device
            .load_ptx(ptx, module_name, &[func_name])
            .map_err(|e| format!("Failed to load PTX for '{}': {}", name, e))?;

        Ok(())
    }

    /// Run GPU analysis on the network.
    pub fn analyze(&self, network: &AccountingNetwork) -> Result<GpuAnalysisResult, String> {
        let start = Instant::now();
        let cuda_device = self.device.inner();

        let n_accounts = network.accounts.len();
        let n_flows = network.flows.len();

        if n_accounts == 0 {
            return Ok(GpuAnalysisResult::default());
        }

        let mut result = GpuAnalysisResult {
            accounts_processed: n_accounts,
            flows_processed: n_flows,
            ..Default::default()
        };

        // === Run Suspense Detection ===
        if let Some(ref kernel) = self.suspense_kernel {
            let suspense_scores = self.run_suspense_detection(network, kernel)?;
            result.suspense_scores = suspense_scores;
        }

        // === Run GAAP Violation Detection ===
        if let Some(ref kernel) = self.gaap_kernel {
            if n_flows > 0 {
                let violations = self.run_gaap_violation(network, kernel)?;
                result.gaap_violations = violations;
            }
        }

        // === Run Benford Analysis ===
        if let Some(ref kernel) = self.benford_kernel {
            if n_flows > 0 {
                let counts = self.run_benford_analysis(network, kernel)?;
                result.benford_counts = counts;
            }
        }

        // Synchronize to ensure all GPU work is done
        cuda_device.synchronize()
            .map_err(|e| format!("GPU synchronize failed: {}", e))?;

        result.execution_time_us = start.elapsed().as_micros() as u64;

        Ok(result)
    }

    /// Run suspense detection kernel.
    fn run_suspense_detection(
        &self,
        network: &AccountingNetwork,
        kernel: &CompiledKernel,
    ) -> Result<Vec<f32>, String> {
        let cuda_device = self.device.inner();
        let n = network.accounts.len();

        // Prepare input data
        let balance_debit: Vec<f64> = network.accounts.iter()
            .map(|a| a.total_debits.to_f64())
            .collect();
        let balance_credit: Vec<f64> = network.accounts.iter()
            .map(|a| a.total_credits.to_f64())
            .collect();
        let risk_scores: Vec<f32> = network.accounts.iter()
            .map(|a| a.risk_score)
            .collect();
        let inflow_counts: Vec<u32> = network.accounts.iter()
            .map(|a| a.in_degree as u32)
            .collect();
        let outflow_counts: Vec<u32> = network.accounts.iter()
            .map(|a| a.out_degree as u32)
            .collect();

        // Allocate GPU memory
        let d_balance_debit = cuda_device.htod_copy(balance_debit)
            .map_err(|e| format!("Failed to copy balance_debit: {}", e))?;
        let d_balance_credit = cuda_device.htod_copy(balance_credit)
            .map_err(|e| format!("Failed to copy balance_credit: {}", e))?;
        let d_risk_scores = cuda_device.htod_copy(risk_scores)
            .map_err(|e| format!("Failed to copy risk_scores: {}", e))?;
        let d_inflow_counts = cuda_device.htod_copy(inflow_counts)
            .map_err(|e| format!("Failed to copy inflow_counts: {}", e))?;
        let d_outflow_counts = cuda_device.htod_copy(outflow_counts)
            .map_err(|e| format!("Failed to copy outflow_counts: {}", e))?;

        // Allocate output buffer
        let d_suspense_scores = unsafe { cuda_device.alloc::<f32>(n) }
            .map_err(|e| format!("Failed to allocate suspense_scores: {}", e))?;

        // Get kernel function
        let func = cuda_device.get_func(kernel.module_name, kernel.function_name)
            .ok_or_else(|| format!("Kernel function '{}' not found", kernel.function_name))?;

        // Calculate grid dimensions
        let block_size = 256u32;
        let grid_size = ((n as u32) + block_size - 1) / block_size;

        // Launch kernel
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (
                &d_balance_debit,
                &d_balance_credit,
                &d_risk_scores,
                &d_inflow_counts,
                &d_outflow_counts,
                &d_suspense_scores,
                n as i32,
            ))
        }.map_err(|e| format!("Kernel launch failed: {}", e))?;

        // Copy results back
        let suspense_scores = cuda_device.dtoh_sync_copy(&d_suspense_scores)
            .map_err(|e| format!("Failed to copy results: {}", e))?;

        Ok(suspense_scores)
    }

    /// Run GAAP violation detection kernel.
    fn run_gaap_violation(
        &self,
        network: &AccountingNetwork,
        kernel: &CompiledKernel,
    ) -> Result<Vec<u8>, String> {
        let cuda_device = self.device.inner();
        let n_flows = network.flows.len();

        // Prepare input data
        let flow_source: Vec<u16> = network.flows.iter()
            .map(|f| f.source_account_index)
            .collect();
        let flow_target: Vec<u16> = network.flows.iter()
            .map(|f| f.target_account_index)
            .collect();
        let account_types: Vec<u8> = network.accounts.iter()
            .map(|a| a.account_type as u8)
            .collect();

        // Allocate GPU memory
        let d_flow_source = cuda_device.htod_copy(flow_source)
            .map_err(|e| format!("Failed to copy flow_source: {}", e))?;
        let d_flow_target = cuda_device.htod_copy(flow_target)
            .map_err(|e| format!("Failed to copy flow_target: {}", e))?;
        let d_account_types = cuda_device.htod_copy(account_types)
            .map_err(|e| format!("Failed to copy account_types: {}", e))?;

        // Allocate output buffer
        let d_violation_flags = unsafe { cuda_device.alloc::<u8>(n_flows) }
            .map_err(|e| format!("Failed to allocate violation_flags: {}", e))?;

        // Get kernel function
        let func = cuda_device.get_func(kernel.module_name, kernel.function_name)
            .ok_or_else(|| format!("Kernel function '{}' not found", kernel.function_name))?;

        // Calculate grid dimensions
        let block_size = 256u32;
        let grid_size = ((n_flows as u32) + block_size - 1) / block_size;

        // Launch kernel
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (
                &d_flow_source,
                &d_flow_target,
                &d_account_types,
                &d_violation_flags,
                n_flows as i32,
            ))
        }.map_err(|e| format!("Kernel launch failed: {}", e))?;

        // Copy results back
        let violations = cuda_device.dtoh_sync_copy(&d_violation_flags)
            .map_err(|e| format!("Failed to copy results: {}", e))?;

        Ok(violations)
    }

    /// Run Benford analysis kernel.
    fn run_benford_analysis(
        &self,
        network: &AccountingNetwork,
        kernel: &CompiledKernel,
    ) -> Result<[u32; 9], String> {
        let cuda_device = self.device.inner();
        let n_flows = network.flows.len();

        // Prepare input data - extract amounts from flows
        let amounts: Vec<f64> = network.flows.iter()
            .map(|f| f.amount.to_f64().abs())
            .collect();

        // Allocate GPU memory
        let d_amounts = cuda_device.htod_copy(amounts)
            .map_err(|e| format!("Failed to copy amounts: {}", e))?;

        // Allocate and zero-initialize digit counts
        let d_digit_counts = cuda_device.htod_copy(vec![0u32; 9])
            .map_err(|e| format!("Failed to allocate digit_counts: {}", e))?;

        // Get kernel function
        let func = cuda_device.get_func(kernel.module_name, kernel.function_name)
            .ok_or_else(|| format!("Kernel function '{}' not found", kernel.function_name))?;

        // Calculate grid dimensions
        let block_size = 256u32;
        let grid_size = ((n_flows as u32) + block_size - 1) / block_size;

        // Launch kernel
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (
                &d_amounts,
                &d_digit_counts,
                n_flows as i32,
            ))
        }.map_err(|e| format!("Kernel launch failed: {}", e))?;

        // Copy results back
        let counts_vec = cuda_device.dtoh_sync_copy(&d_digit_counts)
            .map_err(|e| format!("Failed to copy results: {}", e))?;

        let mut counts = [0u32; 9];
        counts.copy_from_slice(&counts_vec);

        Ok(counts)
    }

    /// Run benchmarks comparing CPU vs GPU performance.
    pub fn run_benchmarks(&self, network: &AccountingNetwork) -> Result<BenchmarkResults, String> {
        let mut kernels = Vec::new();
        let mut total_gpu_time = 0u64;

        // Benchmark Suspense Detection
        if let Some(ref kernel) = self.suspense_kernel {
            let start = Instant::now();
            for _ in 0..10 {
                self.run_suspense_detection(network, kernel)?;
            }
            self.device.synchronize().map_err(|e| format!("Sync failed: {}", e))?;
            let elapsed = start.elapsed().as_micros() as u64 / 10;

            let n = network.accounts.len();
            kernels.push(KernelBenchmark {
                name: "Suspense Detection".to_string(),
                time_us: elapsed,
                elements: n,
                throughput_meps: if elapsed > 0 { n as f64 / elapsed as f64 } else { 0.0 },
            });
            total_gpu_time += elapsed;
        }

        // Benchmark GAAP Violation
        if let Some(ref kernel) = self.gaap_kernel {
            if !network.flows.is_empty() {
                let start = Instant::now();
                for _ in 0..10 {
                    self.run_gaap_violation(network, kernel)?;
                }
                self.device.synchronize().map_err(|e| format!("Sync failed: {}", e))?;
                let elapsed = start.elapsed().as_micros() as u64 / 10;

                let n = network.flows.len();
                kernels.push(KernelBenchmark {
                    name: "GAAP Violation".to_string(),
                    time_us: elapsed,
                    elements: n,
                    throughput_meps: if elapsed > 0 { n as f64 / elapsed as f64 } else { 0.0 },
                });
                total_gpu_time += elapsed;
            }
        }

        // Benchmark Benford Analysis
        if let Some(ref kernel) = self.benford_kernel {
            if !network.flows.is_empty() {
                let start = Instant::now();
                for _ in 0..10 {
                    self.run_benford_analysis(network, kernel)?;
                }
                self.device.synchronize().map_err(|e| format!("Sync failed: {}", e))?;
                let elapsed = start.elapsed().as_micros() as u64 / 10;

                let n = network.flows.len();
                kernels.push(KernelBenchmark {
                    name: "Benford Analysis".to_string(),
                    time_us: elapsed,
                    elements: n,
                    throughput_meps: if elapsed > 0 { n as f64 / elapsed as f64 } else { 0.0 },
                });
                total_gpu_time += elapsed;
            }
        }

        // Run CPU baseline
        let cpu_start = Instant::now();
        for _ in 0..10 {
            self.cpu_baseline(network);
        }
        let total_cpu_time = cpu_start.elapsed().as_micros() as u64 / 10;

        let speedup = if total_gpu_time > 0 {
            total_cpu_time as f64 / total_gpu_time as f64
        } else {
            0.0
        };

        Ok(BenchmarkResults {
            device_name: self.device_name.clone(),
            compute_capability: self.compute_capability,
            kernels,
            total_gpu_time_us: total_gpu_time,
            total_cpu_time_us: total_cpu_time,
            speedup,
        })
    }

    /// CPU baseline for comparison.
    fn cpu_baseline(&self, network: &AccountingNetwork) {
        // Suspense detection
        let _suspense: Vec<f32> = network.accounts.iter().map(|a| {
            let balance = a.total_debits.to_f64() - a.total_credits.to_f64();
            let mut score = 0.0f32;
            if balance.abs() > 0.0 && balance.abs() % 1000.0 < 1.0 {
                score += 0.3;
            }
            if a.risk_score > 0.5 {
                score += 0.4;
            }
            let flow_ratio = if a.out_degree > 0 {
                a.in_degree as f32 / a.out_degree as f32
            } else {
                10.0
            };
            if flow_ratio > 5.0 {
                score += 0.3;
            }
            score.min(1.0)
        }).collect();

        // GAAP violations
        let _violations: Vec<u8> = network.flows.iter().map(|f| {
            let src_type = network.accounts.get(f.source_account_index as usize)
                .map(|a| a.account_type as u8)
                .unwrap_or(0);
            let tgt_type = network.accounts.get(f.target_account_index as usize)
                .map(|a| a.account_type as u8)
                .unwrap_or(0);

            if src_type == 3 && tgt_type == 0 { 1 }
            else if src_type == 3 && tgt_type == 4 { 2 }
            else { 0 }
        }).collect();

        // Benford analysis
        let mut _digit_counts = [0u32; 9];
        for flow in &network.flows {
            let amount = flow.amount.to_f64().abs();
            if amount >= 1.0 {
                let mut value = amount;
                while value >= 10.0 { value /= 10.0; }
                let first_digit = value as usize;
                if first_digit >= 1 && first_digit <= 9 {
                    _digit_counts[first_digit - 1] += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_gpu_executor_creation() {
        // This test will only pass if CUDA is available
        if ringkernel_cuda::is_cuda_available() {
            let executor = GpuExecutor::new();
            assert!(executor.is_ok());
        }
    }
}
