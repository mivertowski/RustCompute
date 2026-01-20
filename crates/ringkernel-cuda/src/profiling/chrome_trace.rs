//! Chrome Trace format GPU timeline export.
//!
//! This module provides GPU timeline visualization in chrome://tracing format,
//! with support for kernel events, memory transfers, and NVTX ranges.
//!
//! # Chrome Trace Format
//!
//! The Chrome Trace format is a JSON-based format supported by:
//! - Chrome DevTools (chrome://tracing)
//! - Perfetto UI (ui.perfetto.dev)
//! - Nsight Systems (can import)
//!
//! # Example
//!
//! ```ignore
//! use ringkernel_cuda::profiling::{GpuChromeTraceBuilder, ProfilingSession, KernelMetrics};
//!
//! let mut session = ProfilingSession::new();
//! // ... record events ...
//!
//! let trace = GpuChromeTraceBuilder::from_session(&session).build();
//! std::fs::write("gpu_trace.json", trace)?;
//! ```

use std::collections::HashMap;

use serde::Serialize;

use ringkernel_core::observability::ProfilerColor;

use super::metrics::{KernelMetrics, ProfilingSession, TransferDirection, TransferMetrics};

/// A single event in the Chrome Trace format.
#[derive(Debug, Clone, Serialize)]
pub struct GpuTraceEvent {
    /// Event name.
    pub name: String,
    /// Category (kernel, transfer, memory, nvtx).
    pub cat: String,
    /// Phase: "X" (complete), "B" (begin), "E" (end), "i" (instant).
    pub ph: String,
    /// Timestamp in microseconds.
    pub ts: u64,
    /// Duration in microseconds (for "X" events).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dur: Option<u64>,
    /// Process ID (device index).
    pub pid: u32,
    /// Thread ID (stream ID).
    pub tid: u32,
    /// Additional arguments.
    #[serde(skip_serializing_if = "GpuEventArgs::is_empty")]
    pub args: GpuEventArgs,
    /// Color name (for visualization).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cname: Option<String>,
}

/// Additional arguments for trace events.
#[derive(Debug, Clone, Default, Serialize)]
pub struct GpuEventArgs {
    /// Grid dimensions [x, y, z].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grid_dim: Option<[u32; 3]>,
    /// Block dimensions [x, y, z].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_dim: Option<[u32; 3]>,
    /// Shared memory bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shared_mem_bytes: Option<u32>,
    /// Registers per thread.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub registers: Option<u32>,
    /// Occupancy (0.0-1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub occupancy: Option<f32>,
    /// Transfer size in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<usize>,
    /// Transfer bandwidth in GB/s.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bandwidth_gbps: Option<f64>,
    /// Transfer direction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direction: Option<String>,
    /// Total threads launched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_threads: Option<u64>,
    /// Throughput in M threads/sec.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub throughput_mthreads: Option<f64>,
    /// Memory pointer address.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ptr: Option<String>,
    /// Allocation label.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Generic message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

impl GpuEventArgs {
    /// Check if args are empty (all None).
    pub fn is_empty(&self) -> bool {
        self.grid_dim.is_none()
            && self.block_dim.is_none()
            && self.shared_mem_bytes.is_none()
            && self.registers.is_none()
            && self.occupancy.is_none()
            && self.size_bytes.is_none()
            && self.bandwidth_gbps.is_none()
            && self.direction.is_none()
            && self.total_threads.is_none()
            && self.throughput_mthreads.is_none()
            && self.ptr.is_none()
            && self.label.is_none()
            && self.message.is_none()
    }
}

/// Builder for Chrome Trace GPU timeline.
#[derive(Debug, Default)]
pub struct GpuChromeTraceBuilder {
    /// Trace events.
    events: Vec<GpuTraceEvent>,
    /// Metadata for the trace.
    metadata: HashMap<String, String>,
    /// Process names (device index -> name).
    process_names: HashMap<u32, String>,
    /// Thread names (stream ID -> name).
    thread_names: HashMap<u32, String>,
}

impl GpuChromeTraceBuilder {
    /// Create a new trace builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a trace builder from a profiling session.
    pub fn from_session(session: &ProfilingSession) -> Self {
        let mut builder = Self::new();

        // Add metadata
        if !session.name().is_empty() {
            builder.add_metadata("session_name", session.name());
        }

        // Add kernel events
        for (ts, metrics) in session.kernel_events() {
            builder.add_kernel_event(metrics, *ts, metrics.stream_id.unwrap_or(0) as u32);
        }

        // Add transfer events
        for (ts, metrics) in session.transfer_events() {
            builder.add_transfer_event(metrics, *ts, metrics.stream_id.unwrap_or(0) as u32);
        }

        builder
    }

    /// Add metadata to the trace.
    pub fn add_metadata(&mut self, key: &str, value: &str) -> &mut Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Set a process name (device name).
    pub fn set_process_name(&mut self, device_index: u32, name: &str) -> &mut Self {
        self.process_names.insert(device_index, name.to_string());
        self
    }

    /// Set a thread name (stream name).
    pub fn set_thread_name(&mut self, stream_id: u32, name: &str) -> &mut Self {
        self.thread_names.insert(stream_id, name.to_string());
        self
    }

    /// Add a kernel execution event.
    pub fn add_kernel_event(
        &mut self,
        metrics: &KernelMetrics,
        start_us: u64,
        stream_id: u32,
    ) -> &mut Self {
        let dur_us = (metrics.gpu_time_ms * 1000.0) as u64;

        let args = GpuEventArgs {
            grid_dim: Some([metrics.grid_dim.0, metrics.grid_dim.1, metrics.grid_dim.2]),
            block_dim: Some([
                metrics.block_dim.0,
                metrics.block_dim.1,
                metrics.block_dim.2,
            ]),
            shared_mem_bytes: if metrics.shared_mem_bytes > 0 {
                Some(metrics.shared_mem_bytes)
            } else {
                None
            },
            registers: metrics.registers_per_thread,
            occupancy: metrics.occupancy,
            total_threads: Some(metrics.total_threads()),
            throughput_mthreads: metrics.throughput_mthreads_per_sec(),
            ..Default::default()
        };

        self.events.push(GpuTraceEvent {
            name: metrics.kernel_name.clone(),
            cat: "kernel".to_string(),
            ph: "X".to_string(),
            ts: start_us,
            dur: Some(dur_us),
            pid: metrics.device_index,
            tid: stream_id,
            args,
            cname: Some("thread_state_running".to_string()),
        });

        self
    }

    /// Add a memory transfer event.
    pub fn add_transfer_event(
        &mut self,
        metrics: &TransferMetrics,
        start_us: u64,
        stream_id: u32,
    ) -> &mut Self {
        let dur_us = (metrics.gpu_time_ms * 1000.0) as u64;

        let args = GpuEventArgs {
            size_bytes: Some(metrics.size_bytes),
            bandwidth_gbps: Some(metrics.bandwidth_gbps()),
            direction: Some(metrics.direction.label().to_string()),
            ..Default::default()
        };

        let color = match metrics.direction {
            TransferDirection::HostToDevice => "cq_build_passed",
            TransferDirection::DeviceToHost => "cq_build_failed",
            TransferDirection::DeviceToDevice => "cq_build_attempt",
        };

        let device = match metrics.direction {
            TransferDirection::HostToDevice => metrics.dst_device.max(0) as u32,
            TransferDirection::DeviceToHost => metrics.src_device.max(0) as u32,
            TransferDirection::DeviceToDevice => metrics.src_device.max(0) as u32,
        };

        self.events.push(GpuTraceEvent {
            name: format!(
                "{} ({} bytes)",
                metrics.direction.label(),
                metrics.size_bytes
            ),
            cat: "transfer".to_string(),
            ph: "X".to_string(),
            ts: start_us,
            dur: Some(dur_us),
            pid: device,
            tid: stream_id,
            args,
            cname: Some(color.to_string()),
        });

        self
    }

    /// Add an NVTX range event.
    pub fn add_nvtx_range(
        &mut self,
        name: &str,
        start_us: u64,
        dur_us: u64,
        color: ProfilerColor,
        device_index: u32,
    ) -> &mut Self {
        let cname = color_to_chrome_color(color);

        self.events.push(GpuTraceEvent {
            name: name.to_string(),
            cat: "nvtx".to_string(),
            ph: "X".to_string(),
            ts: start_us,
            dur: Some(dur_us),
            pid: device_index,
            tid: 0, // NVTX ranges are typically on main thread
            args: GpuEventArgs::default(),
            cname: Some(cname),
        });

        self
    }

    /// Add an instant marker event.
    pub fn add_marker(
        &mut self,
        name: &str,
        ts_us: u64,
        color: ProfilerColor,
        device_index: u32,
    ) -> &mut Self {
        let cname = color_to_chrome_color(color);

        self.events.push(GpuTraceEvent {
            name: name.to_string(),
            cat: "marker".to_string(),
            ph: "i".to_string(), // instant event
            ts: ts_us,
            dur: None,
            pid: device_index,
            tid: 0,
            args: GpuEventArgs::default(),
            cname: Some(cname),
        });

        self
    }

    /// Add a memory allocation event.
    pub fn add_alloc_event(
        &mut self,
        ptr: u64,
        size: usize,
        label: &str,
        ts_us: u64,
        device_index: u32,
    ) -> &mut Self {
        let args = GpuEventArgs {
            ptr: Some(format!("0x{:x}", ptr)),
            size_bytes: Some(size),
            label: Some(label.to_string()),
            ..Default::default()
        };

        self.events.push(GpuTraceEvent {
            name: format!("Alloc: {} ({} bytes)", label, size),
            cat: "memory".to_string(),
            ph: "i".to_string(),
            ts: ts_us,
            dur: None,
            pid: device_index,
            tid: 0,
            args,
            cname: Some("good".to_string()),
        });

        self
    }

    /// Add a memory deallocation event.
    pub fn add_free_event(&mut self, ptr: u64, ts_us: u64, device_index: u32) -> &mut Self {
        let args = GpuEventArgs {
            ptr: Some(format!("0x{:x}", ptr)),
            ..Default::default()
        };

        self.events.push(GpuTraceEvent {
            name: format!("Free: 0x{:x}", ptr),
            cat: "memory".to_string(),
            ph: "i".to_string(),
            ts: ts_us,
            dur: None,
            pid: device_index,
            tid: 0,
            args,
            cname: Some("bad".to_string()),
        });

        self
    }

    /// Add a raw trace event.
    pub fn add_event(&mut self, event: GpuTraceEvent) -> &mut Self {
        self.events.push(event);
        self
    }

    /// Get the number of events.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Build the Chrome Trace JSON output.
    pub fn build(&self) -> String {
        let mut trace_events = Vec::new();

        // Add process name events
        for (pid, name) in &self.process_names {
            trace_events.push(serde_json::json!({
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "args": { "name": name }
            }));
        }

        // Add thread name events
        for (tid, name) in &self.thread_names {
            trace_events.push(serde_json::json!({
                "name": "thread_name",
                "ph": "M",
                "tid": tid,
                "args": { "name": name }
            }));
        }

        // Add all trace events
        for event in &self.events {
            trace_events.push(serde_json::to_value(event).unwrap_or_default());
        }

        // Build the final JSON
        let trace = serde_json::json!({
            "traceEvents": trace_events,
            "metadata": self.metadata,
            "displayTimeUnit": "ms"
        });

        serde_json::to_string_pretty(&trace).unwrap_or_else(|_| "{}".to_string())
    }

    /// Build compact JSON output (no pretty printing).
    pub fn build_compact(&self) -> String {
        let mut trace_events = Vec::new();

        for (pid, name) in &self.process_names {
            trace_events.push(serde_json::json!({
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "args": { "name": name }
            }));
        }

        for (tid, name) in &self.thread_names {
            trace_events.push(serde_json::json!({
                "name": "thread_name",
                "ph": "M",
                "tid": tid,
                "args": { "name": name }
            }));
        }

        for event in &self.events {
            trace_events.push(serde_json::to_value(event).unwrap_or_default());
        }

        let trace = serde_json::json!({
            "traceEvents": trace_events,
            "metadata": self.metadata,
            "displayTimeUnit": "ms"
        });

        serde_json::to_string(&trace).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Convert ProfilerColor to Chrome trace color name.
fn color_to_chrome_color(color: ProfilerColor) -> String {
    // Chrome trace supports these color names:
    // good, bad, terrible, generic_work, thread_state_running, etc.
    // We'll map based on RGB values

    if color.r == 255 && color.g == 0 && color.b == 0 {
        "bad".to_string() // Red
    } else if color.r == 0 && color.g == 255 && color.b == 0 {
        "good".to_string() // Green
    } else if color.r == 0 && color.g == 0 && color.b == 255 {
        "vsync_highlight_color".to_string() // Blue
    } else if color.r == 255 && color.g == 255 && color.b == 0 {
        "yellow".to_string() // Yellow
    } else if color.r == 0 && color.g == 255 && color.b == 255 {
        "cq_build_attempt".to_string() // Cyan
    } else if color.r == 255 && color.g == 0 && color.b == 255 {
        "rail_animation".to_string() // Magenta
    } else if color.r == 255 && color.g == 165 && color.b == 0 {
        "heap_dump_stack_frame".to_string() // Orange
    } else {
        "generic_work".to_string() // Default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_event_args_empty() {
        let args = GpuEventArgs::default();
        assert!(args.is_empty());

        let args = GpuEventArgs {
            grid_dim: Some([128, 1, 1]),
            ..Default::default()
        };
        assert!(!args.is_empty());
    }

    #[test]
    fn test_builder_basic() {
        let builder = GpuChromeTraceBuilder::new();
        assert_eq!(builder.event_count(), 0);
    }

    #[test]
    fn test_builder_add_kernel_event() {
        let mut builder = GpuChromeTraceBuilder::new();

        let metrics = KernelMetrics::new("test_kernel")
            .with_grid_dim(128, 1, 1)
            .with_block_dim(256, 1, 1)
            .with_gpu_time(0.5);

        builder.add_kernel_event(&metrics, 1000, 0);
        assert_eq!(builder.event_count(), 1);

        let json = builder.build();
        assert!(json.contains("test_kernel"));
        assert!(json.contains("kernel"));
        assert!(json.contains("grid_dim"));
    }

    #[test]
    fn test_builder_add_transfer_event() {
        let mut builder = GpuChromeTraceBuilder::new();

        let metrics =
            TransferMetrics::new(TransferDirection::HostToDevice, 1_000_000).with_gpu_time(1.0);

        builder.add_transfer_event(&metrics, 2000, 0);
        assert_eq!(builder.event_count(), 1);

        let json = builder.build();
        assert!(json.contains("H2D"));
        assert!(json.contains("transfer"));
        assert!(json.contains("bandwidth_gbps"));
    }

    #[test]
    fn test_builder_add_nvtx_range() {
        let mut builder = GpuChromeTraceBuilder::new();

        builder.add_nvtx_range("compute_phase", 0, 5000, ProfilerColor::CYAN, 0);
        assert_eq!(builder.event_count(), 1);

        let json = builder.build();
        assert!(json.contains("compute_phase"));
        assert!(json.contains("nvtx"));
    }

    #[test]
    fn test_builder_add_marker() {
        let mut builder = GpuChromeTraceBuilder::new();

        builder.add_marker("checkpoint", 1000, ProfilerColor::GREEN, 0);
        assert_eq!(builder.event_count(), 1);

        let json = builder.build();
        assert!(json.contains("checkpoint"));
        // Pretty printing adds spaces, use compact for exact match
        let compact = builder.build_compact();
        assert!(compact.contains("\"ph\":\"i\"")); // instant event
    }

    #[test]
    fn test_builder_add_memory_events() {
        let mut builder = GpuChromeTraceBuilder::new();

        builder.add_alloc_event(0x1000, 4096, "buffer", 500, 0);
        builder.add_free_event(0x1000, 1500, 0);

        assert_eq!(builder.event_count(), 2);

        let json = builder.build();
        assert!(json.contains("Alloc"));
        assert!(json.contains("Free"));
        assert!(json.contains("0x1000"));
    }

    #[test]
    fn test_builder_metadata() {
        let mut builder = GpuChromeTraceBuilder::new();

        builder.add_metadata("device", "RTX 4090");
        builder.add_metadata("driver", "550.78");

        let json = builder.build();
        assert!(json.contains("RTX 4090"));
        assert!(json.contains("550.78"));
    }

    #[test]
    fn test_builder_process_thread_names() {
        let mut builder = GpuChromeTraceBuilder::new();

        builder.set_process_name(0, "GPU 0: RTX 4090");
        builder.set_thread_name(0, "Default Stream");
        builder.set_thread_name(1, "Compute Stream");

        let json = builder.build();
        assert!(json.contains("GPU 0: RTX 4090"));
        assert!(json.contains("Default Stream"));
        assert!(json.contains("Compute Stream"));
    }

    #[test]
    fn test_builder_from_session() {
        let mut session = ProfilingSession::with_name("test_session");

        session.record_kernel(
            KernelMetrics::new("kernel1")
                .with_grid_dim(100, 1, 1)
                .with_block_dim(256, 1, 1)
                .with_gpu_time(0.5),
        );

        session.record_transfer(
            TransferMetrics::new(TransferDirection::HostToDevice, 1000).with_gpu_time(0.1),
        );

        let builder = GpuChromeTraceBuilder::from_session(&session);
        assert_eq!(builder.event_count(), 2);

        let json = builder.build();
        assert!(json.contains("kernel1"));
        assert!(json.contains("H2D"));
    }

    #[test]
    fn test_builder_compact_output() {
        let mut builder = GpuChromeTraceBuilder::new();

        builder.add_kernel_event(&KernelMetrics::new("test").with_gpu_time(1.0), 0, 0);

        let pretty = builder.build();
        let compact = builder.build_compact();

        // Compact should be shorter (no whitespace)
        assert!(compact.len() < pretty.len());
        // Both should contain the same data
        assert!(compact.contains("test"));
    }

    #[test]
    fn test_color_mapping() {
        assert_eq!(color_to_chrome_color(ProfilerColor::RED), "bad");
        assert_eq!(color_to_chrome_color(ProfilerColor::GREEN), "good");
        assert_eq!(
            color_to_chrome_color(ProfilerColor::BLUE),
            "vsync_highlight_color"
        );
        assert_eq!(color_to_chrome_color(ProfilerColor::YELLOW), "yellow");
        assert_eq!(
            color_to_chrome_color(ProfilerColor::CYAN),
            "cq_build_attempt"
        );
        assert_eq!(
            color_to_chrome_color(ProfilerColor::MAGENTA),
            "rail_animation"
        );
        assert_eq!(
            color_to_chrome_color(ProfilerColor::ORANGE),
            "heap_dump_stack_frame"
        );
    }

    #[test]
    fn test_trace_event_serialization() {
        let event = GpuTraceEvent {
            name: "test".to_string(),
            cat: "kernel".to_string(),
            ph: "X".to_string(),
            ts: 1000,
            dur: Some(500),
            pid: 0,
            tid: 0,
            args: GpuEventArgs {
                grid_dim: Some([128, 1, 1]),
                ..Default::default()
            },
            cname: Some("good".to_string()),
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"name\":\"test\""));
        assert!(json.contains("\"ph\":\"X\""));
        assert!(json.contains("\"ts\":1000"));
        assert!(json.contains("\"dur\":500"));
    }

    #[test]
    fn test_event_args_skip_none() {
        let args = GpuEventArgs {
            grid_dim: Some([1, 1, 1]),
            // All others are None
            ..Default::default()
        };

        let json = serde_json::to_string(&args).unwrap();
        assert!(json.contains("grid_dim"));
        assert!(!json.contains("block_dim"));
        assert!(!json.contains("bandwidth_gbps"));
    }
}
