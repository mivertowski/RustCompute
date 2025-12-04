//! CUDA kernel execution manager.
//!
//! Manages GPU kernel compilation and execution with CPU fallback.
//! Uses ringkernel-cuda patterns for proper GPU integration.

use super::{KernelSource, KernelType};

#[allow(unused_imports)]
use super::LaunchConfig;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig as CudaLaunchConfig};
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Kernel execution result.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Execution time in microseconds.
    pub execution_time_us: u64,
    /// Number of elements processed.
    pub elements_processed: u64,
    /// Throughput in elements per second.
    pub throughput: f64,
    /// Whether GPU was used.
    pub used_gpu: bool,
}

impl Default for ExecutionResult {
    fn default() -> Self {
        Self {
            execution_time_us: 0,
            elements_processed: 0,
            throughput: 0.0,
            used_gpu: false,
        }
    }
}

impl ExecutionResult {
    /// Create a new execution result.
    pub fn new(execution_time_us: u64, elements_processed: u64, used_gpu: bool) -> Self {
        let throughput = if execution_time_us > 0 {
            elements_processed as f64 * 1_000_000.0 / execution_time_us as f64
        } else {
            0.0
        };
        Self {
            execution_time_us,
            elements_processed,
            throughput,
            used_gpu,
        }
    }
}

/// Compiled kernel handle.
#[derive(Debug)]
pub struct CompiledKernel {
    /// Kernel name.
    pub name: String,
    /// Entry point.
    pub entry_point: String,
    /// Is compiled.
    pub is_compiled: bool,
    /// Kernel type.
    pub kernel_type: KernelType,
    /// CUDA source code.
    pub source: String,
}

/// GPU execution backend status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuStatus {
    /// No GPU available, using CPU fallback.
    CpuFallback,
    /// CUDA GPU available and initialized.
    CudaReady,
    /// GPU available but not initialized.
    CudaPending,
    /// GPU initialization failed.
    CudaError,
    /// CUDA feature not compiled in.
    CudaNotCompiled,
}

impl GpuStatus {
    /// Human-readable status string.
    pub fn as_str(&self) -> &'static str {
        match self {
            GpuStatus::CpuFallback => "CPU",
            GpuStatus::CudaReady => "CUDA",
            GpuStatus::CudaPending => "CUDA (init)",
            GpuStatus::CudaError => "CUDA (err)",
            GpuStatus::CudaNotCompiled => "CPU (no CUDA)",
        }
    }

    /// Check if CUDA feature is compiled.
    pub fn is_cuda_compiled() -> bool {
        cfg!(feature = "cuda")
    }
}

/// GPU usage statistics.
#[derive(Debug, Clone, Default)]
pub struct GpuStats {
    /// Total kernel launches.
    pub kernel_launches: u64,
    /// Total GPU execution time in microseconds.
    pub total_gpu_time_us: u64,
    /// Total elements processed on GPU.
    pub total_elements_gpu: u64,
    /// Total bytes transferred to GPU.
    pub bytes_to_gpu: u64,
    /// Total bytes transferred from GPU.
    pub bytes_from_gpu: u64,
}

impl GpuStats {
    /// Record a kernel execution.
    pub fn record(&mut self, result: &ExecutionResult, bytes_in: u64, bytes_out: u64) {
        if result.used_gpu {
            self.kernel_launches += 1;
            self.total_gpu_time_us += result.execution_time_us;
            self.total_elements_gpu += result.elements_processed;
            self.bytes_to_gpu += bytes_in;
            self.bytes_from_gpu += bytes_out;
        }
    }

    /// Get average kernel time in microseconds.
    pub fn avg_kernel_time_us(&self) -> f64 {
        if self.kernel_launches > 0 {
            self.total_gpu_time_us as f64 / self.kernel_launches as f64
        } else {
            0.0
        }
    }

    /// Get throughput in elements per second.
    pub fn throughput(&self) -> f64 {
        if self.total_gpu_time_us > 0 {
            self.total_elements_gpu as f64 * 1_000_000.0 / self.total_gpu_time_us as f64
        } else {
            0.0
        }
    }
}

/// Kernel executor for GPU operations.
pub struct KernelExecutor {
    /// GPU status.
    gpu_status: GpuStatus,
    /// Compiled kernel cache.
    kernel_cache: std::collections::HashMap<String, CompiledKernel>,
    /// GPU usage statistics.
    pub stats: GpuStats,
    /// CUDA device handle.
    #[cfg(feature = "cuda")]
    device: Option<Arc<CudaDevice>>,
}

impl std::fmt::Debug for KernelExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelExecutor")
            .field("gpu_status", &self.gpu_status)
            .field("kernel_count", &self.kernel_cache.len())
            .field("stats", &self.stats)
            .finish()
    }
}

impl Default for KernelExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelExecutor {
    /// Create a new kernel executor.
    pub fn new() -> Self {
        #[cfg(feature = "cuda")]
        {
            match CudaDevice::new(0) {
                Ok(device) => {
                    log::info!("CUDA device initialized successfully");
                    Self {
                        gpu_status: GpuStatus::CudaReady,
                        kernel_cache: std::collections::HashMap::new(),
                        stats: GpuStats::default(),
                        device: Some(device),
                    }
                }
                Err(e) => {
                    log::warn!("CUDA device initialization failed: {}", e);
                    Self {
                        gpu_status: GpuStatus::CudaError,
                        kernel_cache: std::collections::HashMap::new(),
                        stats: GpuStats::default(),
                        device: None,
                    }
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            Self {
                gpu_status: GpuStatus::CudaNotCompiled,
                kernel_cache: std::collections::HashMap::new(),
                stats: GpuStats::default(),
            }
        }
    }

    /// Get GPU status.
    pub fn gpu_status(&self) -> GpuStatus {
        self.gpu_status
    }

    /// Check if CUDA is available.
    pub fn is_cuda_available(&self) -> bool {
        self.gpu_status == GpuStatus::CudaReady
    }

    /// Get CUDA device reference.
    #[cfg(feature = "cuda")]
    pub fn device(&self) -> Option<&Arc<CudaDevice>> {
        self.device.as_ref()
    }

    /// Compile a kernel from source using NVRTC.
    pub fn compile(&mut self, source: &KernelSource) -> Result<&CompiledKernel, String> {
        // Clone the name first to avoid lifetime issues
        let kernel_name = source.name.clone();
        let entry_point = source.entry_point.clone();
        let cuda_source = source.source.clone();

        if self.kernel_cache.contains_key(&kernel_name) {
            return Ok(self.kernel_cache.get(&kernel_name).unwrap());
        }

        #[cfg(feature = "cuda")]
        if let Some(device) = &self.device {
            // Compile CUDA C to PTX using NVRTC
            let ptx = cudarc::nvrtc::compile_ptx(&cuda_source)
                .map_err(|e| format!("NVRTC compilation failed for {}: {}", kernel_name, e))?;

            // Load the PTX module into the device
            // load_ptx requires 'static strings, so we use leaked Box<str>
            let module_name: &'static str = Box::leak(kernel_name.clone().into_boxed_str());
            let func_name: &'static str = Box::leak(entry_point.clone().into_boxed_str());

            device
                .load_ptx(ptx, module_name, &[func_name])
                .map_err(|e| {
                    format!(
                        "Failed to load PTX for {} (func: {}): {}",
                        kernel_name, func_name, e
                    )
                })?;

            log::info!(
                "Compiled and loaded CUDA kernel: {} (entry: {})",
                kernel_name,
                entry_point
            );
        }

        let compiled = CompiledKernel {
            name: kernel_name.clone(),
            entry_point,
            is_compiled: true,
            kernel_type: source.kernel_type,
            source: cuda_source,
        };

        self.kernel_cache.insert(kernel_name.clone(), compiled);
        Ok(self.kernel_cache.get(&kernel_name).unwrap())
    }

    /// Get compiled kernel count.
    pub fn kernel_count(&self) -> usize {
        self.kernel_cache.len()
    }

    /// Execute DFG construction kernel on GPU.
    #[cfg(feature = "cuda")]
    pub fn execute_dfg_gpu(
        &mut self,
        events: &[crate::models::GpuObjectEvent],
    ) -> Result<(Vec<crate::models::GpuDFGEdge>, ExecutionResult), String> {
        use crate::models::GpuDFGEdge;

        let device = self.device.as_ref().ok_or("No CUDA device available")?;
        let start = std::time::Instant::now();

        let n = events.len();
        if n < 2 {
            return Ok((Vec::new(), ExecutionResult::new(0, 0, false)));
        }

        // Extract activity pairs from consecutive events per case
        let mut source_activities: Vec<u32> = Vec::new();
        let mut target_activities: Vec<u32> = Vec::new();
        let mut durations: Vec<u32> = Vec::new();

        // Group by case and extract transitions
        let mut case_events: std::collections::HashMap<u64, Vec<&crate::models::GpuObjectEvent>> =
            std::collections::HashMap::new();
        for event in events {
            case_events.entry(event.object_id).or_default().push(event);
        }

        for case_evts in case_events.values() {
            let mut sorted: Vec<_> = case_evts.iter().collect();
            sorted.sort_by_key(|e| e.timestamp.physical_ms);

            for window in sorted.windows(2) {
                source_activities.push(window[0].activity_id);
                target_activities.push(window[1].activity_id);
                durations.push(window[0].duration_ms);
            }
        }

        let pair_count = source_activities.len();
        if pair_count == 0 {
            return Ok((Vec::new(), ExecutionResult::new(0, 0, false)));
        }

        // Max activities for edge matrix (32x32)
        let max_activities = 32usize;
        let edge_count = max_activities * max_activities;

        // Host-to-Device transfers
        let d_sources = device
            .htod_sync_copy(&source_activities)
            .map_err(|e| format!("HtoD sources failed: {}", e))?;
        let d_targets = device
            .htod_sync_copy(&target_activities)
            .map_err(|e| format!("HtoD targets failed: {}", e))?;
        let d_durations = device
            .htod_sync_copy(&durations)
            .map_err(|e| format!("HtoD durations failed: {}", e))?;

        // Allocate output buffers (initialized to zeros)
        let edge_frequencies = vec![0u32; edge_count];
        let edge_durations = vec![0u64; edge_count];

        let d_edge_freq = device
            .htod_sync_copy(&edge_frequencies)
            .map_err(|e| format!("HtoD edge_freq failed: {}", e))?;
        let d_edge_dur = device
            .htod_sync_copy(&edge_durations)
            .map_err(|e| format!("HtoD edge_dur failed: {}", e))?;

        // Get the compiled kernel function
        let func = device
            .get_func("dfg_construction", "dfg_construction_kernel")
            .ok_or("DFG kernel not loaded - call compile() with generate_dfg_kernel() first")?;

        // Launch configuration
        let block_size = 256u32;
        let grid_size = (pair_count as u32 + block_size - 1) / block_size;

        let config = CudaLaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        unsafe {
            func.launch(
                config,
                (
                    &d_sources,
                    &d_targets,
                    &d_durations,
                    &d_edge_freq,
                    &d_edge_dur,
                    max_activities as i32,
                    pair_count as i32,
                ),
            )
            .map_err(|e| format!("Kernel launch failed: {}", e))?;
        }

        // Synchronize - wait for kernel completion
        device
            .synchronize()
            .map_err(|e| format!("Device synchronize failed: {}", e))?;

        // Device-to-Host transfers
        let mut result_frequencies = vec![0u32; edge_count];
        let mut result_durations = vec![0u64; edge_count];

        device
            .dtoh_sync_copy_into(&d_edge_freq, &mut result_frequencies)
            .map_err(|e| format!("DtoH frequencies failed: {}", e))?;
        device
            .dtoh_sync_copy_into(&d_edge_dur, &mut result_durations)
            .map_err(|e| format!("DtoH durations failed: {}", e))?;

        let elapsed = start.elapsed().as_micros() as u64;

        // Build edge results from frequency matrix
        let mut edges = Vec::new();
        for src in 0..max_activities {
            for tgt in 0..max_activities {
                let idx = src * max_activities + tgt;
                let freq = result_frequencies[idx];
                if freq > 0 {
                    let total_dur = result_durations[idx];
                    let avg_dur = total_dur as f32 / freq as f32;

                    let mut edge = GpuDFGEdge::default();
                    edge.source_activity = src as u32;
                    edge.target_activity = tgt as u32;
                    edge.frequency = freq;
                    edge.avg_duration_ms = avg_dur;
                    edges.push(edge);
                }
            }
        }

        // Record stats
        let bytes_in = (pair_count * 3 * 4) as u64;
        let bytes_out = (edge_count * 12) as u64;
        let result = ExecutionResult::new(elapsed, pair_count as u64, true);
        self.stats.record(&result, bytes_in, bytes_out);

        log::debug!(
            "DFG GPU kernel: {} pairs -> {} edges in {}us",
            pair_count,
            edges.len(),
            elapsed
        );

        Ok((edges, result))
    }

    /// Execute pattern detection kernel on GPU.
    #[cfg(feature = "cuda")]
    pub fn execute_pattern_gpu(
        &mut self,
        nodes: &[crate::models::GpuDFGNode],
        bottleneck_threshold: f32,
        duration_threshold: f32,
    ) -> Result<(Vec<crate::models::GpuPatternMatch>, ExecutionResult), String> {
        use crate::models::{GpuPatternMatch, PatternSeverity, PatternType};

        let device = self.device.as_ref().ok_or("No CUDA device available")?;
        let start = std::time::Instant::now();

        let n = nodes.len();
        if n == 0 {
            return Ok((Vec::new(), ExecutionResult::new(0, 0, false)));
        }

        // Extract node data into Structure-of-Arrays format for GPU
        let event_counts: Vec<u32> = nodes.iter().map(|n| n.event_count).collect();
        let avg_durations: Vec<f32> = nodes.iter().map(|n| n.avg_duration_ms).collect();
        let incoming_counts: Vec<u16> = nodes.iter().map(|n| n.incoming_count).collect();
        let outgoing_counts: Vec<u16> = nodes.iter().map(|n| n.outgoing_count).collect();

        // HtoD transfers
        let d_event_counts = device
            .htod_sync_copy(&event_counts)
            .map_err(|e| format!("HtoD event_counts failed: {}", e))?;
        let d_avg_durations = device
            .htod_sync_copy(&avg_durations)
            .map_err(|e| format!("HtoD avg_durations failed: {}", e))?;
        let d_incoming = device
            .htod_sync_copy(&incoming_counts)
            .map_err(|e| format!("HtoD incoming failed: {}", e))?;
        let d_outgoing = device
            .htod_sync_copy(&outgoing_counts)
            .map_err(|e| format!("HtoD outgoing failed: {}", e))?;

        // Output buffers
        let pattern_types = vec![0u8; n];
        let pattern_confidences = vec![0.0f32; n];

        let d_pattern_types = device
            .htod_sync_copy(&pattern_types)
            .map_err(|e| format!("HtoD pattern_types failed: {}", e))?;
        let d_pattern_conf = device
            .htod_sync_copy(&pattern_confidences)
            .map_err(|e| format!("HtoD pattern_conf failed: {}", e))?;

        // Get kernel function
        let func = device
            .get_func("pattern_detection", "pattern_detection_kernel")
            .ok_or(
                "Pattern kernel not loaded - call compile() with generate_pattern_kernel() first",
            )?;

        // Launch configuration
        let block_size = 256u32;
        let grid_size = (n as u32 + block_size - 1) / block_size;

        let config = CudaLaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        unsafe {
            func.launch(
                config,
                (
                    &d_event_counts,
                    &d_avg_durations,
                    &d_incoming,
                    &d_outgoing,
                    &d_pattern_types,
                    &d_pattern_conf,
                    bottleneck_threshold,
                    duration_threshold,
                    n as i32,
                ),
            )
            .map_err(|e| format!("Pattern kernel launch failed: {}", e))?;
        }

        device
            .synchronize()
            .map_err(|e| format!("Device synchronize failed: {}", e))?;

        // DtoH transfers
        let mut result_types = vec![0u8; n];
        let mut result_confidences = vec![0.0f32; n];

        device
            .dtoh_sync_copy_into(&d_pattern_types, &mut result_types)
            .map_err(|e| format!("DtoH pattern_types failed: {}", e))?;
        device
            .dtoh_sync_copy_into(&d_pattern_conf, &mut result_confidences)
            .map_err(|e| format!("DtoH pattern_conf failed: {}", e))?;

        let elapsed = start.elapsed().as_micros() as u64;

        // Build pattern results
        let mut patterns = Vec::new();
        for (i, &ptype) in result_types.iter().enumerate() {
            if ptype != 0 {
                let pattern_type = match ptype {
                    6 => PatternType::LongRunning,
                    7 => PatternType::Bottleneck,
                    _ => continue,
                };

                let severity = if ptype == 7 {
                    PatternSeverity::Critical
                } else {
                    PatternSeverity::Warning
                };

                let mut pattern = GpuPatternMatch::new(pattern_type, severity);
                pattern.add_activity(nodes[i].activity_id);
                pattern.confidence = result_confidences[i];
                pattern.frequency = nodes[i].event_count;
                pattern.avg_duration_ms = nodes[i].avg_duration_ms;
                patterns.push(pattern);
            }
        }

        // Record stats
        let bytes_in = (n * 12) as u64;
        let bytes_out = (n * 5) as u64;
        let result = ExecutionResult::new(elapsed, n as u64, true);
        self.stats.record(&result, bytes_in, bytes_out);

        log::debug!(
            "Pattern GPU kernel: {} nodes -> {} patterns in {}us",
            n,
            patterns.len(),
            elapsed
        );

        Ok((patterns, result))
    }

    /// Execute partial order derivation kernel on GPU.
    #[cfg(feature = "cuda")]
    pub fn execute_partial_order_gpu(
        &mut self,
        events: &[crate::models::GpuObjectEvent],
    ) -> Result<(Vec<crate::models::GpuPartialOrderTrace>, ExecutionResult), String> {
        use crate::models::{GpuPartialOrderTrace, HybridTimestamp};
        use std::collections::HashMap;

        let device = self.device.as_ref().ok_or("No CUDA device available")?;
        let start = std::time::Instant::now();

        // Group events by case
        let mut case_events: HashMap<u64, Vec<&crate::models::GpuObjectEvent>> = HashMap::new();
        for event in events {
            case_events.entry(event.object_id).or_default().push(event);
        }

        // For each case, we'll run a GPU kernel to compute the precedence matrix
        let mut all_traces = Vec::with_capacity(case_events.len());
        let mut total_kernel_time_us = 0u64;

        for (case_id, case_evts) in case_events {
            if case_evts.len() < 2 {
                continue;
            }

            let mut sorted: Vec<_> = case_evts.into_iter().collect();
            sorted.sort_by_key(|e| e.timestamp.physical_ms);

            let n = sorted.len().min(16);

            // Extract timing data
            let start_times: Vec<u64> = sorted.iter().map(|e| e.timestamp.physical_ms).collect();
            let end_times: Vec<u64> = sorted
                .iter()
                .map(|e| e.timestamp.physical_ms + e.duration_ms as u64)
                .collect();

            // Pad to 16 elements for consistent GPU buffer sizes
            let mut start_times_padded = vec![0u64; 16];
            let mut end_times_padded = vec![0u64; 16];
            for i in 0..n {
                start_times_padded[i] = start_times[i];
                end_times_padded[i] = end_times[i];
            }

            // HtoD transfers
            let d_start_times = device
                .htod_sync_copy(&start_times_padded)
                .map_err(|e| format!("HtoD start_times failed: {}", e))?;
            let d_end_times = device
                .htod_sync_copy(&end_times_padded)
                .map_err(|e| format!("HtoD end_times failed: {}", e))?;

            // Output buffer (16x16 = 256 elements)
            let precedence_flat = vec![0u32; 256];
            let d_precedence = device
                .htod_sync_copy(&precedence_flat)
                .map_err(|e| format!("HtoD precedence failed: {}", e))?;

            // Get kernel function
            let func = device
                .get_func("partial_order", "partial_order_kernel")
                .ok_or("Partial order kernel not loaded")?;

            // Launch configuration (16x16 grid for pairwise comparison)
            let config = CudaLaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };

            let kernel_start = std::time::Instant::now();

            // Launch kernel
            unsafe {
                func.launch(
                    config,
                    (&d_start_times, &d_end_times, &d_precedence, 16i32, 16i32),
                )
                .map_err(|e| format!("Partial order kernel launch failed: {}", e))?;
            }

            device
                .synchronize()
                .map_err(|e| format!("Device synchronize failed: {}", e))?;

            total_kernel_time_us += kernel_start.elapsed().as_micros() as u64;

            // DtoH transfer
            let mut result_precedence = vec![0u32; 256];
            device
                .dtoh_sync_copy_into(&d_precedence, &mut result_precedence)
                .map_err(|e| format!("DtoH precedence failed: {}", e))?;

            // Convert flat precedence to 16x16 bit matrix
            let mut precedence_matrix = [0u16; 16];
            for i in 0..n {
                for j in 0..n {
                    if result_precedence[i * 16 + j] != 0 {
                        precedence_matrix[i] |= 1u16 << j;
                    }
                }
            }

            // Build activity IDs array
            let mut activity_ids = [u32::MAX; 16];
            for (i, event) in sorted.iter().take(n).enumerate() {
                activity_ids[i] = event.activity_id;
            }

            // Compute timing in seconds
            let trace_start_ms = sorted.first().map(|e| e.timestamp.physical_ms).unwrap_or(0);
            let mut activity_start_secs = [0u16; 16];
            let mut activity_duration_secs = [0u16; 16];
            for (i, event) in sorted.iter().take(n).enumerate() {
                let rel_start = event.timestamp.physical_ms.saturating_sub(trace_start_ms);
                activity_start_secs[i] = (rel_start / 1000).min(u16::MAX as u64) as u16;
                activity_duration_secs[i] =
                    (event.duration_ms / 1000).max(1).min(u16::MAX as u32) as u16;
            }

            // Build trace
            let trace = GpuPartialOrderTrace {
                trace_id: all_traces.len() as u64 + 1,
                case_id,
                event_count: sorted.len() as u32,
                activity_count: n as u32,
                start_time: sorted.first().map(|e| e.timestamp).unwrap_or_default(),
                end_time: HybridTimestamp::new(
                    sorted
                        .last()
                        .map(|e| e.timestamp.physical_ms + e.duration_ms as u64)
                        .unwrap_or(0),
                    0,
                ),
                max_width: Self::compute_width_from_matrix(&precedence_matrix, n),
                flags: 0x01, // Has transitive closure flag
                precedence_matrix,
                activity_ids,
                activity_start_secs,
                activity_duration_secs,
                _reserved: [0u8; 32],
            };

            all_traces.push(trace);
        }

        let elapsed = start.elapsed().as_micros() as u64;
        let result = ExecutionResult::new(elapsed, events.len() as u64, true);

        // Record stats
        let bytes_in = (events.len() * 16) as u64;
        let bytes_out = (all_traces.len() * 256) as u64;
        self.stats.record(&result, bytes_in, bytes_out);

        log::debug!(
            "Partial order GPU kernel: {} events -> {} traces in {}us (kernel: {}us)",
            events.len(),
            all_traces.len(),
            elapsed,
            total_kernel_time_us
        );

        Ok((all_traces, result))
    }

    #[cfg(feature = "cuda")]
    fn compute_width_from_matrix(precedence: &[u16; 16], n: usize) -> u32 {
        let mut max_width = 1u32;

        for i in 0..n {
            let mut concurrent = 1u32;
            for j in (i + 1)..n {
                let i_precedes_j = (precedence[i] & (1u16 << j)) != 0;
                let j_precedes_i = (precedence[j] & (1u16 << i)) != 0;
                if !i_precedes_j && !j_precedes_i {
                    concurrent += 1;
                }
            }
            max_width = max_width.max(concurrent);
        }

        max_width
    }

    /// Execute conformance checking kernel on GPU.
    #[cfg(feature = "cuda")]
    pub fn execute_conformance_gpu(
        &mut self,
        events: &[crate::models::GpuObjectEvent],
        model: &crate::models::ProcessModel,
    ) -> Result<(Vec<crate::models::ConformanceResult>, ExecutionResult), String> {
        use crate::models::{ComplianceLevel, ConformanceResult, ConformanceStatus};
        use std::collections::HashMap;

        let device = self.device.as_ref().ok_or("No CUDA device available")?;
        let start = std::time::Instant::now();

        // Group events by case
        let mut case_events: HashMap<u64, Vec<&crate::models::GpuObjectEvent>> = HashMap::new();
        for event in events {
            case_events.entry(event.object_id).or_default().push(event);
        }

        // Flatten traces for GPU processing
        let mut all_activities: Vec<u32> = Vec::new();
        let mut trace_starts: Vec<i32> = Vec::new();
        let mut trace_lengths: Vec<i32> = Vec::new();
        let mut trace_case_ids: Vec<u64> = Vec::new();

        for (case_id, case_evts) in &case_events {
            let mut sorted: Vec<_> = case_evts.iter().collect();
            sorted.sort_by_key(|e| e.timestamp.physical_ms);

            trace_starts.push(all_activities.len() as i32);
            trace_lengths.push(sorted.len() as i32);
            trace_case_ids.push(*case_id);

            for event in sorted {
                all_activities.push(event.activity_id);
            }
        }

        let num_traces = trace_starts.len();
        if num_traces == 0 {
            return Ok((Vec::new(), ExecutionResult::new(0, 0, false)));
        }

        // Extract model transitions
        let model_sources: Vec<u32> = model.transitions.iter().map(|(s, _)| *s).collect();
        let model_targets: Vec<u32> = model.transitions.iter().map(|(_, t)| *t).collect();
        let num_transitions = model_sources.len() as i32;

        // HtoD transfers
        let d_activities = device
            .htod_sync_copy(&all_activities)
            .map_err(|e| format!("HtoD activities failed: {}", e))?;
        let d_trace_starts = device
            .htod_sync_copy(&trace_starts)
            .map_err(|e| format!("HtoD trace_starts failed: {}", e))?;
        let d_trace_lengths = device
            .htod_sync_copy(&trace_lengths)
            .map_err(|e| format!("HtoD trace_lengths failed: {}", e))?;
        let d_model_sources = device
            .htod_sync_copy(&model_sources)
            .map_err(|e| format!("HtoD model_sources failed: {}", e))?;
        let d_model_targets = device
            .htod_sync_copy(&model_targets)
            .map_err(|e| format!("HtoD model_targets failed: {}", e))?;

        // Output buffer
        let fitness_scores = vec![0.0f32; num_traces];
        let d_fitness = device
            .htod_sync_copy(&fitness_scores)
            .map_err(|e| format!("HtoD fitness failed: {}", e))?;

        // Get kernel function
        let func = device
            .get_func("conformance", "conformance_kernel")
            .ok_or("Conformance kernel not loaded")?;

        // Launch configuration
        let block_size = 256u32;
        let grid_size = (num_traces as u32 + block_size - 1) / block_size;

        let config = CudaLaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        unsafe {
            func.launch(
                config,
                (
                    &d_activities,
                    &d_trace_starts,
                    &d_trace_lengths,
                    &d_model_sources,
                    &d_model_targets,
                    num_transitions,
                    &d_fitness,
                    num_traces as i32,
                ),
            )
            .map_err(|e| format!("Conformance kernel launch failed: {}", e))?;
        }

        device
            .synchronize()
            .map_err(|e| format!("Device synchronize failed: {}", e))?;

        // DtoH transfer
        let mut result_fitness = vec![0.0f32; num_traces];
        device
            .dtoh_sync_copy_into(&d_fitness, &mut result_fitness)
            .map_err(|e| format!("DtoH fitness failed: {}", e))?;

        let elapsed = start.elapsed().as_micros() as u64;

        // Build conformance results
        let mut results = Vec::with_capacity(num_traces);
        for (i, &fitness) in result_fitness.iter().enumerate() {
            let status = if fitness >= 1.0 {
                ConformanceStatus::Conformant
            } else if fitness >= 0.8 {
                ConformanceStatus::Deviation
            } else {
                ConformanceStatus::MissingActivity
            };

            results.push(ConformanceResult {
                trace_id: trace_case_ids[i],
                model_id: model.id,
                status: status as u8,
                compliance_level: ComplianceLevel::from_fitness(fitness) as u8,
                fitness,
                precision: fitness, // Simplified: same as fitness
                generalization: 0.8,
                simplicity: 1.0,
                missing_count: ((1.0 - fitness) * trace_lengths[i] as f32) as u16,
                extra_count: 0,
                alignment_cost: ((1.0 - fitness) * trace_lengths[i] as f32) as u32,
                alignment_length: trace_lengths[i] as u32,
                _padding1: [0; 2],
                _reserved: [0; 16],
            });
        }

        // Record stats
        let bytes_in = (all_activities.len() * 4 + model_sources.len() * 8) as u64;
        let bytes_out = (num_traces * 4) as u64;
        let result = ExecutionResult::new(elapsed, events.len() as u64, true);
        self.stats.record(&result, bytes_in, bytes_out);

        log::debug!(
            "Conformance GPU kernel: {} traces in {}us, avg fitness: {:.2}",
            num_traces,
            elapsed,
            result_fitness.iter().sum::<f32>() / num_traces as f32
        );

        Ok((results, result))
    }
}

/// CPU fallback executor for when CUDA is not available.
pub struct CpuFallbackExecutor;

impl CpuFallbackExecutor {
    /// Execute DFG construction on CPU.
    pub fn execute_dfg_construction(
        events: &[crate::models::GpuObjectEvent],
        edges: &mut [crate::models::GpuDFGEdge],
        max_activities: usize,
    ) -> ExecutionResult {
        let start = std::time::Instant::now();

        let mut case_events: std::collections::HashMap<u64, Vec<&crate::models::GpuObjectEvent>> =
            std::collections::HashMap::new();

        for event in events {
            case_events.entry(event.object_id).or_default().push(event);
        }

        for events in case_events.values() {
            let mut sorted_events: Vec<_> = events.iter().collect();
            sorted_events.sort_by_key(|e| e.timestamp.physical_ms);

            for window in sorted_events.windows(2) {
                let source = window[0].activity_id as usize;
                let target = window[1].activity_id as usize;

                if source < max_activities && target < max_activities {
                    let edge_idx = source * max_activities + target;
                    if edge_idx < edges.len() {
                        edges[edge_idx].frequency += 1;
                        edges[edge_idx].source_activity = source as u32;
                        edges[edge_idx].target_activity = target as u32;
                    }
                }
            }
        }

        let elapsed = start.elapsed().as_micros() as u64;
        ExecutionResult::new(elapsed, events.len() as u64, false)
    }

    /// Execute pattern detection on CPU.
    pub fn execute_pattern_detection(
        nodes: &[crate::models::GpuDFGNode],
        patterns: &mut Vec<crate::models::GpuPatternMatch>,
        bottleneck_threshold: f32,
        duration_threshold: f32,
    ) -> ExecutionResult {
        use crate::models::{GpuPatternMatch, PatternSeverity, PatternType};

        let start = std::time::Instant::now();

        for node in nodes {
            if node.event_count == 0 {
                continue;
            }

            let incoming_count = node.incoming_count as f32;
            let outgoing_count = node.outgoing_count as f32;
            if incoming_count > bottleneck_threshold && outgoing_count < incoming_count * 0.5 {
                let mut pattern =
                    GpuPatternMatch::new(PatternType::Bottleneck, PatternSeverity::Critical);
                pattern.add_activity(node.activity_id);
                pattern.confidence = incoming_count / bottleneck_threshold;
                pattern.frequency = node.event_count;
                pattern.avg_duration_ms = node.avg_duration_ms;
                patterns.push(pattern);
            }

            if node.avg_duration_ms > duration_threshold {
                let mut pattern =
                    GpuPatternMatch::new(PatternType::LongRunning, PatternSeverity::Warning);
                pattern.add_activity(node.activity_id);
                pattern.confidence = node.avg_duration_ms / duration_threshold;
                pattern.frequency = node.event_count;
                pattern.avg_duration_ms = node.avg_duration_ms;
                patterns.push(pattern);
            }
        }

        let elapsed = start.elapsed().as_micros() as u64;
        ExecutionResult::new(elapsed, nodes.len() as u64, false)
    }

    /// Execute partial order derivation on CPU.
    pub fn execute_partial_order(
        events: &[crate::models::GpuObjectEvent],
        traces: &mut Vec<crate::models::GpuPartialOrderTrace>,
    ) -> ExecutionResult {
        use crate::models::{GpuPartialOrderTrace, HybridTimestamp};
        use std::collections::HashMap;

        let start = std::time::Instant::now();

        let mut case_events: HashMap<u64, Vec<&crate::models::GpuObjectEvent>> = HashMap::new();
        for event in events {
            case_events.entry(event.object_id).or_default().push(event);
        }

        for (case_id, case_evts) in &case_events {
            if case_evts.len() < 2 {
                continue;
            }

            let mut sorted: Vec<_> = case_evts.iter().collect();
            sorted.sort_by_key(|e| e.timestamp.physical_ms);

            let n = sorted.len().min(16);
            let mut precedence = [0u16; 16];
            let mut activity_ids = [0u32; 16];
            let mut activity_start_secs = [0u16; 16];
            let mut activity_duration_secs = [0u16; 16];

            // Get trace start time for relative timing
            let trace_start_ms = sorted.first().map(|e| e.timestamp.physical_ms).unwrap_or(0);

            for i in 0..n {
                activity_ids[i] = sorted[i].activity_id;

                // Store timing info in SECONDS (relative to trace start)
                let rel_start_ms = sorted[i]
                    .timestamp
                    .physical_ms
                    .saturating_sub(trace_start_ms);
                activity_start_secs[i] = (rel_start_ms / 1000).min(u16::MAX as u64) as u16;
                activity_duration_secs[i] =
                    (sorted[i].duration_ms / 1000).max(1).min(u16::MAX as u32) as u16;

                for j in (i + 1)..n {
                    let i_end = sorted[i].timestamp.physical_ms + sorted[i].duration_ms as u64;
                    let j_start = sorted[j].timestamp.physical_ms;
                    if i_end <= j_start {
                        precedence[i] |= 1u16 << j;
                    }
                }
            }

            // Calculate trace end time (last event end)
            let trace_end_ms = sorted
                .iter()
                .take(n)
                .map(|e| e.timestamp.physical_ms + e.duration_ms as u64)
                .max()
                .unwrap_or(0);

            let trace = GpuPartialOrderTrace {
                trace_id: traces.len() as u64 + 1,
                case_id: *case_id,
                event_count: sorted.len() as u32,
                activity_count: n as u32,
                start_time: sorted.first().map(|e| e.timestamp).unwrap_or_default(),
                end_time: HybridTimestamp::new(trace_end_ms, 0),
                max_width: Self::compute_width(&precedence, n),
                flags: 0,
                precedence_matrix: precedence,
                activity_ids,
                activity_start_secs,
                activity_duration_secs,
                _reserved: [0u8; 32],
            };

            traces.push(trace);
        }

        let elapsed = start.elapsed().as_micros() as u64;
        ExecutionResult::new(elapsed, events.len() as u64, false)
    }

    fn compute_width(precedence: &[u16; 16], n: usize) -> u32 {
        let mut max_width = 1u32;

        for i in 0..n {
            let mut concurrent = 1u32;
            for j in (i + 1)..n {
                let i_precedes_j = (precedence[i] & (1u16 << j)) != 0;
                let j_precedes_i = (precedence[j] & (1u16 << i)) != 0;
                if !i_precedes_j && !j_precedes_i {
                    concurrent += 1;
                }
            }
            max_width = max_width.max(concurrent);
        }

        max_width
    }

    /// Execute conformance checking on CPU.
    pub fn execute_conformance(
        events: &[crate::models::GpuObjectEvent],
        model: &crate::models::ProcessModel,
    ) -> (Vec<crate::models::ConformanceResult>, ExecutionResult) {
        use crate::models::{ComplianceLevel, ConformanceResult, ConformanceStatus};
        use std::collections::HashMap;

        let start = std::time::Instant::now();
        let mut results = Vec::new();

        let mut case_events: HashMap<u64, Vec<&crate::models::GpuObjectEvent>> = HashMap::new();
        for event in events {
            case_events.entry(event.object_id).or_default().push(event);
        }

        for (case_id, case_evts) in &case_events {
            if case_evts.is_empty() {
                continue;
            }

            let mut sorted: Vec<_> = case_evts.iter().collect();
            sorted.sort_by_key(|e| e.timestamp.physical_ms);

            let mut valid_moves = 0u32;
            let total_moves = (sorted.len() - 1) as u32;

            for window in sorted.windows(2) {
                let source = window[0].activity_id;
                let target = window[1].activity_id;

                let is_valid = model
                    .transitions
                    .iter()
                    .any(|(s, t)| *s == source && *t == target);

                if is_valid {
                    valid_moves += 1;
                }
            }

            let fitness = if total_moves > 0 {
                valid_moves as f32 / total_moves as f32
            } else {
                1.0
            };

            let compliance = ComplianceLevel::from_fitness(fitness);

            let status = if fitness >= 0.95 {
                ConformanceStatus::Conformant
            } else if total_moves > valid_moves {
                ConformanceStatus::ExtraActivity
            } else {
                ConformanceStatus::WrongSequence
            };

            let extra = total_moves.saturating_sub(valid_moves) as u16;

            results.push(ConformanceResult {
                trace_id: *case_id,
                model_id: model.id,
                status: status as u8,
                compliance_level: compliance as u8,
                _padding1: [0; 2],
                fitness,
                precision: 1.0 - extra as f32 / (total_moves + 1) as f32,
                generalization: 0.8,
                simplicity: 1.0,
                missing_count: 0,
                extra_count: extra,
                alignment_cost: extra as u32,
                alignment_length: sorted.len() as u32,
                _reserved: [0u8; 16],
            });
        }

        let elapsed = start.elapsed().as_micros() as u64;
        let exec_result = ExecutionResult::new(elapsed, events.len() as u64, false);

        (results, exec_result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::codegen::generate_dfg_kernel;

    #[test]
    fn test_executor_creation() {
        let executor = KernelExecutor::new();
        assert_eq!(executor.kernel_count(), 0);
    }

    #[test]
    fn test_kernel_compilation() {
        let mut executor = KernelExecutor::new();
        let gpu_status = executor.gpu_status();
        let source = generate_dfg_kernel();
        let result = executor.compile(&source);

        // Convert result to bool to release borrow
        let is_ok = result.is_ok();
        let err_msg = if let Err(ref e) = result {
            Some(e.clone())
        } else {
            None
        };

        // Print error for debugging if it fails
        if let Some(e) = err_msg {
            eprintln!("Kernel compilation error: {}", e);
        }

        // Skip GPU compilation tests when no CUDA available
        if gpu_status == GpuStatus::CpuFallback {
            assert!(is_ok); // CPU fallback should still cache the kernel
            assert_eq!(executor.kernel_count(), 1);
        } else {
            // With CUDA, might fail due to NVRTC issues in test env
            // Either way, kernel should be in cache (compiled or pending)
            assert!(is_ok || gpu_status == GpuStatus::CudaError);
        }
    }

    #[test]
    fn test_execution_result() {
        let result = ExecutionResult::new(1000, 1_000_000, true);
        assert_eq!(result.throughput, 1_000_000_000.0);
        assert!(result.used_gpu);
    }

    #[test]
    fn test_gpu_status() {
        let executor = KernelExecutor::new();
        let status = executor.gpu_status();
        assert!(matches!(
            status,
            GpuStatus::CpuFallback
                | GpuStatus::CudaReady
                | GpuStatus::CudaError
                | GpuStatus::CudaNotCompiled
        ));
    }

    #[test]
    fn test_gpu_stats() {
        let mut stats = GpuStats::default();
        let result = ExecutionResult::new(1000, 10000, true);
        stats.record(&result, 4000, 2000);

        assert_eq!(stats.kernel_launches, 1);
        assert_eq!(stats.total_gpu_time_us, 1000);
        assert_eq!(stats.bytes_to_gpu, 4000);
        assert_eq!(stats.bytes_from_gpu, 2000);
    }
}
